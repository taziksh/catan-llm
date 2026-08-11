import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from pwin_sweep import TABLES, existing_rows, run_cell, summarize, wilson
from catan_llm.whole_game import SampledCompletion, make_rollouts


def fake_sampler(prompts):
    return [SampledCompletion("answer: nonsense", [1], [2]) for _ in prompts]


def test_opponent_tables_reach_players():
    rollout = make_rollouts([20001], 1, 42, opponents=TABLES["3xvp"])[0]
    names = sorted(type(player).__name__ for player in rollout.game.state.players)
    assert names == ["PolicyPlayer"] + ["VictoryPointPlayer"] * 3


def test_run_cell_writes_rows_and_reruns_skip(tmp_path):
    out = tmp_path / "games.jsonl"
    seeds = [20001, 20002]
    run_cell(fake_sampler, 0.7, "3xvp", seeds, 42, out, done=set())
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["seed"] for row in rows] == seeds
    assert all(row["table"] == "3xvp" and row["temperature"] == 0.7 for row in rows)
    assert all("win" in row and "vp_margin" in row for row in rows)

    run_cell(fake_sampler, 0.7, "3xvp", seeds, 42, out, done=existing_rows(out))
    assert len(out.read_text().splitlines()) == len(seeds)


def test_summarize_counts_cells(tmp_path):
    out = tmp_path / "games.jsonl"
    run_cell(fake_sampler, 1.0, "3xvp", [20001, 20002, 20003], 42, out, done=set())
    summary = summarize(out)
    cell = summary["t1_3xvp"]
    assert cell["games"] == 3
    lo, hi = wilson(cell["wins"], cell["games"])
    assert cell["win_rate_ci95"] == [lo, hi]
    assert 0.0 <= lo <= cell["win_rate"] <= hi <= 1.0
