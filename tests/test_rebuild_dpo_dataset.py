import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import label_stability_audit as audit
import rebuild_dpo_dataset as rebuild


def _score(diff, timed_out=False):
    return {
        "chosen_value": diff,
        "rejected_value": 0.0,
        "timed_out": timed_out,
    }


def _agg(diffs):
    return audit.aggregate([_score(diff) for diff in diffs], len(diffs))


def test_retention_keeps_exactly_twelve_of_sixteen():
    agg = _agg([1.0] * 12 + [-1.0] * 4)
    assert agg["fraction_preferring_chosen"] == 0.75
    assert rebuild.drop_reason(agg, 0.75) is None


def test_retention_drops_eleven_of_sixteen():
    agg = _agg([1.0] * 11 + [-1.0] * 5)
    assert agg["label"] == "RETAINED"
    assert rebuild.drop_reason(agg, 0.75) == "low_fraction"


def test_retention_drops_mean_reversal_despite_high_fraction():
    agg = _agg([0.01] * 15 + [-10.0])
    assert agg["fraction_preferring_chosen"] == 15 / 16
    assert rebuild.drop_reason(agg, 0.75) == "reversed"


def test_retention_drops_exact_mean_tie():
    agg = _agg([1.0] * 8 + [-1.0] * 8)
    assert rebuild.drop_reason(agg, 0.75) == "tie"


def test_world_seeds_are_the_audit_stream():
    assert rebuild.world_seeds is audit.world_seeds
    seeds = rebuild.world_seeds(42, "g1", 7, 16)
    assert len(set(seeds)) == 16
    assert seeds[0] == 657765618
    assert seeds[:4] == audit.world_seeds(42, "g1", 7, 4)
    assert seeds == rebuild.world_seeds(42, "g1", 7, 16)


def _pair_line(game_id, decision, chosen, move_type, spaced=False):
    pair = {
        "prompt": [{"role": "user", "content": "board"}],
        "chosen": [{"role": "assistant", "content": f"answer: {chosen}"}],
        "rejected": [{"role": "assistant", "content": "answer: end_turn"}],
        "game_id": game_id,
        "seed": 10_000,
        "decision": decision,
        "move_type": move_type,
        "teacher_value_gap": 1.0,
    }
    separators = (", ", ": ") if spaced else (",", ":")
    return json.dumps(pair, separators=separators).encode()


# Per-world value diffs keyed by chosen move; worlds=4, min_fraction=0.75.
DIFFS = {
    "road:1-2": [1.0, 1.0, 1.0, 1.0],  # kept, unanimous
    "robber:5:BLUE": [1.0, 1.0, -0.5, -0.5],  # dropped, low fraction
    "robber:9:RED": [0.5, 0.5, 0.5, -9.0],  # dropped, mean reversed
    "city:3": [2.0, 2.0, 2.0, -1.0],  # kept, 3/4 fraction
}


def _write_inputs(tmp_path):
    # gB before gA so processing order (sorted by game) differs from
    # file order; the second line uses non-compact JSON separators to
    # catch any re-serialization of kept lines.
    train_lines = [
        _pair_line("gB", 4, "road:1-2", "BUILD_ROAD"),
        _pair_line("gB", 9, "robber:5:BLUE", "MOVE_ROBBER", spaced=True),
        _pair_line("gA", 2, "robber:9:RED", "MOVE_ROBBER"),
    ]
    val_lines = [_pair_line("gC", 1, "city:3", "BUILD_CITY")]
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_bytes(b"".join(line + b"\n" for line in train_lines))
    val.write_bytes(b"".join(line + b"\n" for line in val_lines))
    return train, val, train_lines, val_lines


def _fake_scoring(monkeypatch):
    monkeypatch.setattr(
        rebuild,
        "replay_decisions",
        lambda path, wanted: dict.fromkeys(wanted, "state"),
    )
    monkeypatch.setattr(
        rebuild,
        "score_pair_worlds",
        lambda replayed, chosen_id, rejected_id, seeds, seconds: [
            _score(diff) for diff in DIFFS[chosen_id][: len(seeds)]
        ],
    )


def test_rebuild_filters_membership_only(monkeypatch, tmp_path):
    train, val, train_lines, val_lines = _write_inputs(tmp_path)
    _fake_scoring(monkeypatch)
    out = tmp_path / "repaired"
    metadata = rebuild.rebuild(
        train, val, tmp_path, out, 4, 0.75, 42, 5.0
    )

    # Kept pairs are byte-identical to their originals, in input order.
    assert (out / "train.jsonl").read_bytes() == train_lines[0] + b"\n"
    assert (out / "val.jsonl").read_bytes() == val_lines[0] + b"\n"

    dropped = [
        json.loads(line)
        for line in (out / "dropped_pairs.jsonl").read_text().splitlines()
    ]
    assert [(row["game_id"], row["decision"]) for row in dropped] == [
        ("gB", 9),
        ("gA", 2),
    ]
    low, reversed_row = dropped
    assert low["reason"] == "low_fraction"
    assert low["fraction_preferring_chosen"] == 0.5
    assert low["label"] == "RETAINED"
    assert reversed_row["reason"] == "reversed"
    assert reversed_row["label"] == "REVERSED"
    assert reversed_row["split"] == "train"
    assert all(row["class"] == "MOVE_ROBBER" for row in dropped)

    assert metadata["parameters"]["worlds"] == 4
    assert metadata["parameters"]["min_fraction"] == 0.75
    assert metadata["parameters"]["seed"] == 42
    train_counts = metadata["splits"]["train"]
    assert train_counts["input"] == 3
    assert train_counts["kept"] == 1
    assert train_counts["dropped"] == 2
    assert train_counts["kept"] + train_counts["dropped"] == 3
    assert train_counts["dropped_low_fraction"] == 1
    assert train_counts["dropped_reversed"] == 1
    assert metadata["splits"]["val"] == {
        "input": 1,
        "kept": 1,
        "dropped": 0,
        "dropped_reversed": 0,
        "dropped_tie": 0,
        "dropped_low_fraction": 0,
        "dropped_failed": 0,
        "deadline_hits": 0,
    }
    assert metadata["composition"]["train"] == {
        "BUILD_ROAD": {"input": 1, "kept": 1, "dropped": 0},
        "MOVE_ROBBER": {"input": 2, "kept": 0, "dropped": 2},
    }
    assert metadata["reversed_pairs"] == 1
    assert metadata["failed_pairs"] == 0
    assert metadata["deadline_hits"] == 0
    assert not metadata["warnings"]

    written = json.loads((out / "metadata.json").read_text())
    assert written == metadata


def test_rebuild_counts_failed_replay_as_dropped(monkeypatch, tmp_path):
    train, val, _, val_lines = _write_inputs(tmp_path)

    def broken_replay(path, wanted):
        if path.name == "gA.jsonl":
            raise ValueError(f"{path.name}: legal actions diverged")
        return dict.fromkeys(wanted, "state")

    monkeypatch.setattr(rebuild, "replay_decisions", broken_replay)
    monkeypatch.setattr(
        rebuild,
        "score_pair_worlds",
        lambda replayed, chosen_id, rejected_id, seeds, seconds: [
            _score(diff) for diff in DIFFS[chosen_id][: len(seeds)]
        ],
    )
    out = tmp_path / "repaired"
    metadata = rebuild.rebuild(train, val, tmp_path, out, 4, 0.75, 42, 5.0)

    assert metadata["splits"]["train"]["dropped_failed"] == 1
    assert metadata["failed_pairs"] == 1
    failed = [
        json.loads(line)
        for line in (out / "dropped_pairs.jsonl").read_text().splitlines()
        if line and json.loads(line)["reason"] == "failed"
    ]
    assert [(row["game_id"], row["label"]) for row in failed] == [
        ("gA", "FAILED")
    ]
    assert "diverged" in failed[0]["error"]
    assert (out / "val.jsonl").read_bytes() == val_lines[0] + b"\n"


def test_rebuild_refuses_to_overwrite_inputs(tmp_path):
    train, val, _, _ = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        rebuild.rebuild(train, val, tmp_path, tmp_path, 4, 0.75, 42, 5.0)


def test_parser_defaults():
    args = rebuild.build_parser().parse_args([])
    assert args.train == "data/dpo/r3pairs/train.jsonl"
    assert args.val == "data/dpo/r3pairs/val.jsonl"
    assert args.traces == "data/dagger_traces/r3pairs"
    assert args.out == "data/dpo/r3pairs_repaired"
    assert args.worlds == 16
    assert args.min_fraction == 0.75
    assert args.seed == 42
    assert args.seconds_per_action == 5.0
