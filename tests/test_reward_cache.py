import json
import sys
from pathlib import Path

import pytest
from catanatron import Game

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_reward_cache
from build_reward_cache import (
    build_cache,
    checked_cache,
    continuation_reward,
    continuation_seed,
    index_path,
    load_cache,
    append_rows,
    playable_move_ids,
    row_key,
    scenario_seeds,
    score_moves,
    write_index,
)
from label_stability_audit import world_seeds
from run_grpo import state_action_scores

from catan_llm.bots import BOTS, COLORS
from catan_llm.extract import TrajectoryAccumulator, deterministic_game_id
from catan_llm.replay import replay_model_decisions
from catan_llm.schema import Player
from catan_llm.simulation import GameOutcome

TRAIN_SEED = 10_019


def record_llm_game(out_dir, seed=TRAIN_SEED):
    """Log a bot game with one seat marked llm, using a training-range seed."""
    players = [BOTS["value_function"](color) for color in COLORS]
    game = Game(players, seed=seed)
    game.id = deterministic_game_id(game)
    accumulator = TrajectoryAccumulator(out_dir)
    accumulator.before(game)
    llm_color = COLORS[0]
    while game.winning_color() is None and game.state.num_turns < 300:
        current = game.state.current_player()
        if current.color == llm_color:
            action = game.playable_actions[0]
        else:
            action = current.decide(game, game.playable_actions)
        accumulator.step(game, action)
        game.execute(action)
    accumulator.after(game)

    lines = accumulator.path.read_text().splitlines()
    header = json.loads(lines[0])
    header["seats"][llm_color.value] = "llm"
    accumulator.path.write_text(
        "\n".join([json.dumps(header)] + lines[1:]) + "\n"
    )
    return accumulator.path


@pytest.fixture(scope="module")
def llm_trace(tmp_path_factory):
    return record_llm_game(tmp_path_factory.mktemp("llm_trace"))


@pytest.fixture(scope="module")
def replayed_states(llm_trace):
    return list(replay_model_decisions(llm_trace))


def test_scenario_seeds_match_audit_derivation():
    assert scenario_seeds(42, "game-a", 17, 8) == world_seeds(42, "game-a", 17, 8)
    assert scenario_seeds(42, "game-a", 17, 4) == scenario_seeds(
        42, "game-a", 17, 8
    )[:4]
    assert len(set(scenario_seeds(42, "game-a", 17, 8))) == 8
    assert scenario_seeds(42, "game-a", 18, 8) != scenario_seeds(
        42, "game-a", 17, 8
    )


def test_continuation_reward_formula():
    outcome = GameOutcome(
        winner=Player.RED,
        victory_points={Player.RED: 11, Player.BLUE: 4},
        turns=100,
    )
    assert continuation_reward(outcome, Player.RED) == pytest.approx(1.1)
    assert continuation_reward(outcome, Player.BLUE) == pytest.approx(0.04)

    truncated = GameOutcome(
        winner=None, victory_points={Player.RED: 8}, turns=300
    )
    assert truncated.truncated
    assert continuation_reward(truncated, Player.RED) == pytest.approx(0.08)


def test_score_moves_is_paired_and_deterministic(replayed_states):
    for replayed in replayed_states[-2:]:
        decision = replayed.decision
        moves = playable_move_ids(replayed)[:2]
        seeds = scenario_seeds(42, decision.game_id, decision.i, 2)

        rows = score_moves(
            replayed, moves, seeds, "value_function", "value_function"
        )
        again = score_moves(
            replayed, moves, seeds, "value_function", "value_function"
        )

        assert rows == again
        assert {(row["move_id"], row["scenario_seed"]) for row in rows} == {
            (move, seed) for seed in seeds for move in moves
        }
        for row in rows:
            assert row["reward"] == pytest.approx(
                float(row["won"]) + 0.1 * min(row["hero_vp"], 10) / 10
            )
            assert 0.0 <= row["reward"] <= 1.1


def test_score_moves_pairs_worlds_and_separates_continuation_stream(
    replayed_states, monkeypatch
):
    replayed = replayed_states[-1]
    decision = replayed.decision
    moves = playable_move_ids(replayed)[:2]
    seeds = scenario_seeds(42, decision.game_id, decision.i, 2)
    calls = []

    def record_rollout(world, index, policies, seed=None):
        calls.append((id(world), seed))
        return GameOutcome(
            winner=None,
            victory_points={color: 0 for color in policies},
            turns=0,
        )

    monkeypatch.setattr(build_reward_cache, "rollout_action", record_rollout)
    score_moves(replayed, moves, seeds, "value_function", "value_function")

    worlds = [world for world, _ in calls]
    continuations = [seed for _, seed in calls]
    assert worlds[0] == worlds[1] and worlds[2] == worlds[3]
    assert continuations == [
        continuation_seed(seeds[0]),
        continuation_seed(seeds[0]),
        continuation_seed(seeds[1]),
        continuation_seed(seeds[1]),
    ]
    assert set(continuations).isdisjoint(seeds)
    assert continuation_seed(seeds[0]) != continuation_seed(seeds[1])


def test_score_moves_refuses_illegal_move(replayed_states):
    replayed = replayed_states[-1]
    with pytest.raises(ValueError, match="not legal"):
        score_moves(
            replayed,
            ["settlement:999"],
            [7],
            "value_function",
            "value_function",
        )


def _row(game_id, decision, move, seed, reward):
    return {
        "game_id": game_id,
        "decision": decision,
        "move_id": move,
        "scenario_seed": seed,
        "won": False,
        "hero_vp": 5,
        "truncated": False,
        "turns": 50,
        "reward": reward,
    }


CONFIG = {
    "trajectories": "traces",
    "scenarios": 2,
    "seed": 42,
    "hero_policy": "alpha_beta",
    "opponent_policy": "value_function",
}


def test_cache_roundtrip_and_checked_load(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    rows = [_row("g", 1, "end_turn", 7, 0.05), _row("g", 1, "roll", 7, 1.1)]
    append_rows(cache_path, rows)

    loaded = load_cache(cache_path)
    assert set(loaded) == {("g", 1, "end_turn", 7), ("g", 1, "roll", 7)}
    assert loaded[("g", 1, "roll", 7)]["reward"] == 1.1

    index = write_index(cache_path, CONFIG)
    assert index["rows"] == 2
    assert index["games"] == 1
    assert index["decisions"] == 1

    reloaded, checked_index = checked_cache(cache_path)
    assert reloaded == loaded
    assert checked_index["cache_sha256"] == index["cache_sha256"]

    append_rows(cache_path, [_row("g", 2, "roll", 9, 0.06)])
    with pytest.raises(RuntimeError, match="hash"):
        checked_cache(cache_path)


def _fake_score_moves(
    replayed, move_ids, seeds, hero_policy=None, opponent_policy=None
):
    return [
        _row(
            replayed.game_record.game_id,
            replayed.decision.i,
            move,
            seed,
            ((seed + replayed.decision.i + len(move)) % 100) / 100,
        )
        for seed in seeds
        for move in move_ids
    ]


def test_build_cache_is_append_safe(llm_trace, replayed_states, monkeypatch, tmp_path):
    monkeypatch.setattr(build_reward_cache, "score_moves", _fake_score_moves)
    out = tmp_path / "cache.jsonl"

    index = build_cache(
        llm_trace.parent, out, scenarios=2, seed=42, workers=1, max_decisions=2
    )
    expected_rows = sum(
        2 * len(playable_move_ids(replayed))
        for replayed in replayed_states[:2]
    )
    assert index["rows_added"] == index["rows"] == expected_rows
    assert index["decisions"] == 2

    again = build_cache(
        llm_trace.parent, out, scenarios=2, seed=42, workers=1, max_decisions=2
    )
    assert again["rows_added"] == 0
    assert again["cache_sha256"] == index["cache_sha256"]

    with pytest.raises(RuntimeError, match="seed"):
        build_cache(
            llm_trace.parent, out,
            scenarios=2, seed=7, workers=1, max_decisions=2,
        )


def test_build_cache_skips_val_games(
    llm_trace, replayed_states, monkeypatch, tmp_path
):
    monkeypatch.setattr(build_reward_cache, "score_moves", _fake_score_moves)
    (tmp_path / llm_trace.name).write_text(llm_trace.read_text())
    lines = llm_trace.read_text().splitlines()
    header = json.loads(lines[0])
    header["seed"] = 10_020
    (tmp_path / "val-game.jsonl").write_text(
        "\n".join([json.dumps(header)] + lines[1:]) + "\n"
    )

    index = build_cache(
        tmp_path, tmp_path / "cache.jsonl",
        scenarios=1, seed=42, workers=1, max_decisions=1,
    )
    assert index["games"] == 1
    assert index["val_every"] == 10
    assert index["rows_added"] == len(playable_move_ids(replayed_states[0]))


def test_build_cache_refuses_eval_seeds(llm_trace, tmp_path):
    lines = llm_trace.read_text().splitlines()
    header = json.loads(lines[0])
    header["seed"] = 42
    (tmp_path / llm_trace.name).write_text(
        "\n".join([json.dumps(header)] + lines[1:]) + "\n"
    )
    with pytest.raises(ValueError, match="eval"):
        build_cache(
            tmp_path, tmp_path / "cache.jsonl",
            scenarios=1, seed=42, workers=1, max_decisions=1,
        )


def test_state_action_scores_hits_misses_and_crn(llm_trace, replayed_states):
    replayed = replayed_states[-1]
    decision = replayed.decision
    first, second = playable_move_ids(replayed)[:2]
    seeds = scenario_seeds(42, decision.game_id, decision.i, 2)
    direct = score_moves(
        replayed, [first, second], seeds, "value_function", "value_function"
    )
    cache = {
        row_key(row): row for row in direct if row["move_id"] == first
    }

    scores, new_rows, hits, misses = state_action_scores(
        cache, llm_trace, replayed, [first, second], seeds, None,
        "value_function", "value_function",
    )

    assert (hits, misses) == (2, 2)
    assert new_rows == [row for row in direct if row["move_id"] == second]
    expected = {
        move: sum(
            row["reward"] for row in direct if row["move_id"] == move
        ) / len(seeds)
        for move in (first, second)
    }
    assert scores == pytest.approx(expected)

    warm = state_action_scores(
        cache, llm_trace, replayed, [first, second], seeds, None,
        "value_function", "value_function",
    )
    assert warm == (scores, [], 4, 0)


def test_build_cache_refuses_duplicate_game_ids(llm_trace, monkeypatch, tmp_path):
    monkeypatch.setattr(build_reward_cache, "score_moves", _fake_score_moves)
    (tmp_path / llm_trace.name).write_text(llm_trace.read_text())
    (tmp_path / "backup-copy.jsonl").write_text(llm_trace.read_text())
    with pytest.raises(ValueError, match="duplicate"):
        build_cache(
            tmp_path, tmp_path / "cache.jsonl",
            scenarios=1, seed=42, workers=1, max_decisions=1,
        )


def test_build_cache_rerun_skips_rows_of_renamed_trajectory(
    llm_trace, monkeypatch, tmp_path
):
    monkeypatch.setattr(build_reward_cache, "score_moves", _fake_score_moves)
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "renamed.jsonl").write_text(llm_trace.read_text())
    out = tmp_path / "cache.jsonl"

    build_cache(traces, out, scenarios=1, seed=42, workers=1, max_decisions=1)
    again = build_cache(traces, out, scenarios=1, seed=42, workers=1, max_decisions=1)

    assert again["rows_added"] == 0


def _rebuild_index_module():
    import rebuild_reward_cache_index

    return rebuild_reward_cache_index


def test_repair_drops_partial_trailing_line_only(tmp_path):
    cache = tmp_path / "cache.jsonl"
    append_rows(cache, [_row("g", 1, "end_turn", 7, 0.05)])
    intact = cache.read_bytes()
    cache.write_bytes(intact + b'{"game_id": "g", "decis')

    removed = _rebuild_index_module().repair_truncated_tail(cache)

    assert removed == len(b'{"game_id": "g", "decis')
    assert cache.read_bytes() == intact


def test_repair_keeps_complete_rows(tmp_path):
    cache = tmp_path / "cache.jsonl"
    append_rows(cache, [_row("g", 1, "end_turn", 7, 0.05)])
    intact = cache.read_bytes()
    tail_missing_newline = json.dumps(_row("g", 2, "roll", 9, 0.06)).encode()
    cache.write_bytes(intact + tail_missing_newline)

    assert _rebuild_index_module().repair_truncated_tail(cache) == 0
    assert cache.read_bytes() == intact + tail_missing_newline + b"\n"

    assert _rebuild_index_module().repair_truncated_tail(cache) == 0
    assert cache.read_bytes() == intact + tail_missing_newline + b"\n"


EXISTING_INDEX = {
    **CONFIG,
    "val_every": 10,
    "scorer_sha256": "a" * 64,
    "catanatron_version": "3.3.0",
    "cache_sha256": "irrelevant",
    "rows": 2,
}


def test_resolve_config_fills_unset_keys_from_existing_index():
    resolved = _rebuild_index_module().resolve_config(
        EXISTING_INDEX, {"seed": None}, force=False
    )
    assert resolved["seed"] == 42
    assert resolved["scorer_sha256"] == "a" * 64
    assert set(resolved) == set(_rebuild_index_module().CONFIG_KEYS)


def test_resolve_config_refuses_silent_drift_without_force():
    module = _rebuild_index_module()
    with pytest.raises(ValueError, match="seed"):
        module.resolve_config(EXISTING_INDEX, {"seed": 7}, force=False)
    assert module.resolve_config(EXISTING_INDEX, {"seed": 7}, force=True)["seed"] == 7


def test_build_cache_refuses_indexless_cache(llm_trace, tmp_path):
    out = tmp_path / "cache.jsonl"
    out.write_text("")
    with pytest.raises(RuntimeError, match="no index"):
        build_cache(
            llm_trace.parent, out, scenarios=2, seed=42, workers=1,
            max_decisions=1,
        )


def test_build_cache_refuses_scorer_mismatch(llm_trace, monkeypatch, tmp_path):
    monkeypatch.setattr(build_reward_cache, "score_moves", _fake_score_moves)
    out = tmp_path / "cache.jsonl"
    build_cache(
        llm_trace.parent, out, scenarios=1, seed=42, workers=1, max_decisions=1
    )
    index = json.loads(index_path(out).read_text())
    index["scorer_sha256"] = "0" * 64
    index_path(out).write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="scorer_sha256"):
        build_cache(
            llm_trace.parent, out,
            scenarios=1, seed=42, workers=1, max_decisions=1,
        )
