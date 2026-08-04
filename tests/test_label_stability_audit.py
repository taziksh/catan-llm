import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import label_stability_audit as audit


def _pair(game_id, decision, chosen, rejected="end_turn", gap=1.0):
    return {
        "chosen": [{"role": "assistant", "content": f"answer: {chosen}"}],
        "rejected": [{"role": "assistant", "content": f"answer: {rejected}"}],
        "game_id": game_id,
        "seed": 10_000,
        "decision": decision,
        "teacher_value_gap": gap,
    }


def _pairs(robber_count, other_count):
    robber = [
        _pair(f"g{i}", i, f"robber:{i}:BLUE") for i in range(robber_count)
    ]
    other = [
        _pair(f"g{robber_count + i}", i, f"road:{i}-{i + 1}")
        for i in range(other_count)
    ]
    return robber + other


def test_sampling_is_deterministic_and_stratified():
    pairs = _pairs(30, 70)
    first = audit.sample_pairs(pairs, 20, seed=42)
    second = audit.sample_pairs(pairs, 20, seed=42)
    assert first == second
    assert len(first) == len(set(first)) == 20
    assert sum(audit.is_robber(pairs[i]) for i in first) == 10
    assert audit.sample_pairs(pairs, 20, seed=43) != first


def test_sampling_fills_from_other_when_robber_scarce():
    pairs = _pairs(3, 50)
    sampled = audit.sample_pairs(pairs, 20, seed=42)
    assert len(sampled) == 20
    assert sum(audit.is_robber(pairs[i]) for i in sampled) == 3


def test_sampling_caps_at_available_pairs():
    pairs = _pairs(2, 3)
    assert audit.sample_pairs(pairs, 60, seed=42) == [0, 1, 2, 3, 4]


def test_world_seeds_nest_across_counts():
    sixteen = audit.world_seeds(42, "g1", 7, 16)
    assert len(set(sixteen)) == 16
    assert audit.world_seeds(42, "g1", 7, 8) == sixteen[:8]
    assert audit.world_seeds(42, "g1", 7, 4) == sixteen[:4]
    assert audit.world_seeds(42, "g1", 7, 16) == sixteen


def test_world_seeds_differ_by_pair_and_seed():
    base = audit.world_seeds(42, "g1", 7, 4)
    assert audit.world_seeds(42, "g1", 8, 4) != base
    assert audit.world_seeds(42, "g2", 7, 4) != base
    assert audit.world_seeds(43, "g1", 7, 4) != base


def _score(diff, timed_out=False):
    return {
        "chosen_value": diff,
        "rejected_value": 0.0,
        "timed_out": timed_out,
    }


def test_aggregate_retained_unanimous():
    result = audit.aggregate([_score(1.0), _score(2.0)], 2)
    assert result["label"] == "RETAINED"
    assert result["unanimous"]
    assert result["mean_value_diff"] == 1.5
    assert result["fraction_preferring_chosen"] == 1.0
    assert result["deadline_hits"] == 0


def test_aggregate_reversed_non_unanimous():
    result = audit.aggregate([_score(1.0), _score(-3.0)], 2)
    assert result["label"] == "REVERSED"
    assert not result["unanimous"]
    assert result["fraction_preferring_chosen"] == 0.5


def test_aggregate_tie_and_deadline_hits():
    result = audit.aggregate([_score(1.0, timed_out=True), _score(-1.0)], 2)
    assert result["label"] == "TIE"
    assert not result["unanimous"]
    assert result["deadline_hits"] == 1


def test_aggregate_uses_world_prefix():
    scores = [_score(1.0), _score(1.0), _score(-100.0), _score(-100.0)]
    assert audit.aggregate(scores, 2)["label"] == "RETAINED"
    assert audit.aggregate(scores, 4)["label"] == "REVERSED"
    with pytest.raises(ValueError):
        audit.aggregate(scores, 8)


def test_run_audit_with_stubbed_scoring(monkeypatch, tmp_path):
    pairs = _pairs(2, 2)
    monkeypatch.setattr(
        audit,
        "replay_decisions",
        lambda path, wanted: dict.fromkeys(wanted, "state"),
    )
    monkeypatch.setattr(
        audit,
        "score_pair_worlds",
        lambda replayed, chosen_id, rejected_id, seeds, seconds: [
            _score(1.0) for _ in seeds
        ],
    )
    rows, failures = audit.run_audit(
        pairs, tmp_path, [2, 4], sample=4, seed=42, seconds_per_action=5.0
    )
    assert not failures
    assert len(rows) == 4
    assert all(
        row["per_worlds"][count]["label"] == "RETAINED"
        for row in rows
        for count in (2, 4)
    )

    report = audit.build_report(rows, failures, [2, 4], {"seed": 42})
    assert report["sampled_by_class"] == {"robber": 2, "other": 2}
    assert report["convergence"]["all"] == {"pairs": 4, "agreement": 1.0}
    assert report["deadline_hits"] == {2: 0, 4: 0}
    assert not report["warnings"]
    by_key = {
        (entry["worlds"], entry["class"]): entry
        for entry in report["summary"]
    }
    assert by_key[(4, "robber")]["retained"] == 2
    assert by_key[(4, "other")]["non_unanimous"] == 0
    projected = report["timing"]["projected_full_rebuild_seconds"]
    assert set(projected) == {2, 4}


def test_run_audit_records_failures(monkeypatch, tmp_path):
    pairs = _pairs(1, 1)

    def broken_replay(path, wanted):
        raise ValueError(f"{path.name}: legal actions diverged")

    monkeypatch.setattr(audit, "replay_decisions", broken_replay)
    rows, failures = audit.run_audit(
        pairs, tmp_path, [4], sample=2, seed=42, seconds_per_action=5.0
    )
    assert not rows
    assert {failure["class"] for failure in failures} == {"robber", "other"}
    assert all("diverged" in failure["error"] for failure in failures)


def test_deadline_hits_produce_warning():
    scores = [_score(1.0, timed_out=True), _score(1.0)]
    row = {
        "game_id": "g0",
        "decision": 0,
        "class": "other",
        "chosen_id": "road:0-1",
        "rejected_id": "end_turn",
        "original_gap": 1.0,
        "per_worlds": {2: audit.aggregate(scores, 2)},
        "replay_seconds": 0.0,
        "score_seconds": 0.0,
    }
    report = audit.build_report([row], [], [2], {"seed": 42})
    assert report["deadline_hits"] == {2: 1}
    assert len(report["warnings"]) == 1


def test_parser_defaults():
    args = audit.build_parser().parse_args(["--out", "report.json"])
    assert args.pairs == "data/dpo/r3pairs/train.jsonl"
    assert args.traces == "data/dagger_traces/r3pairs"
    assert args.sample == 60
    assert args.worlds == "4,8,16"
    assert args.seed == 42
    assert args.seconds_per_action == 5.0


def test_parse_worlds():
    assert audit.parse_worlds("16,4,8") == [4, 8, 16]
    for bad in ("", "4,x", "0,4", "4,4"):
        with pytest.raises(ValueError):
            audit.parse_worlds(bad)
