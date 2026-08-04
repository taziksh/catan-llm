import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_reward_cache import scenario_seeds
from preflight_init_check import paired_records, summarize


def _rows(count):
    return [
        {
            "game_id": f"game-{index % 3}",
            "decision": index,
            "legal_moves": ["roll", "end_turn"],
        }
        for index in range(count)
    ]


def _cache(rows, seed, scenarios):
    cache = {}
    for row in rows:
        seeds = scenario_seeds(seed, row["game_id"], row["decision"], scenarios)
        for move in row["legal_moves"]:
            for scenario in seeds:
                cache[(row["game_id"], row["decision"], move, scenario)] = {
                    "reward": 0.4 if move == "roll" else 0.1
                }
    return cache


def test_paired_records_value_both_arms_and_zero_invalid():
    rows = _rows(4)
    cache = _cache(rows, 42, 2)
    records = paired_records(
        rows, ["roll", "roll", None, "end_turn"],
        ["end_turn", "roll", "roll", None], cache, 42, 2,
    )
    assert [record["base_value"] for record in records] == pytest.approx(
        [0.4, 0.4, 0.0, 0.1]
    )
    assert [record["adapter_value"] for record in records] == pytest.approx(
        [0.1, 0.4, 0.4, 0.0]
    )


def test_summarize_reports_paired_gap_with_ci():
    rows = _rows(30)
    cache = _cache(rows, 42, 2)
    records = paired_records(
        rows, ["end_turn"] * 30, ["roll"] * 30, cache, 42, 2
    )
    summary = summarize(records, 42)
    assert summary == summarize(records, 42)
    assert summary["decisions"] == 30
    assert summary["games"] == 3
    assert summary["mean_diff"] == pytest.approx(0.3)
    assert summary["ci95_low"] == pytest.approx(0.3)
    assert summary["ci95_high"] == pytest.approx(0.3)
    assert summary["base_invalid_rate"] == 0.0
    assert summary["adapter_invalid_rate"] == 0.0
