"""Sweep win rate over sampling temperature and opponent table.

Writes one row per game to games.jsonl and per-cell win rate and VP margin
to summary.json. Rerun skips games already recorded.
"""

import argparse
import json
import math
from pathlib import Path

from catan_llm.determinism import check_fixed_hashseed
from catan_llm.whole_game import (
    hero_vps,
    make_rollouts,
    rollout_games,
    training_seeds,
)
from run_dpo import load_text_only_qwen35
from run_whole_game_grpo import ModelSampler

TABLES = {
    "3xvf": ("value_function",) * 3,
    "mixed": ("value_function", "victory_point", "victory_point"),
    "3xvp": ("victory_point",) * 3,
}
DEFAULT_TEMPS = (0.0, 0.7, 1.0)
DEFAULT_SEED_START = 20_001
DEFAULT_GAMES = 100
BATCH = 32


def margin(rollout) -> int:
    best_opponent = max(
        vp
        for player, vp in rollout.outcome.victory_points.items()
        if player != rollout.hero
    )
    return hero_vps(rollout) - best_opponent


def wilson(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """wilson_interval from plots.py, which needs matplotlib the node lacks."""
    p = wins / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    spread = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games**2)) / denom
    return center - spread, center + spread


def existing_rows(path: Path) -> set[tuple[float, str, int]]:
    if not path.exists():
        return set()
    return {
        (row["temperature"], row["table"], row["seed"])
        for row in map(json.loads, path.read_text().splitlines())
    }


def game_row(temperature: float, table: str, rollout) -> dict:
    if rollout.failed is not None or rollout.outcome is None:
        return {
            "temperature": temperature,
            "table": table,
            "seed": rollout.seed,
            "failed": rollout.failed or "no outcome",
        }
    return {
        "temperature": temperature,
        "table": table,
        "seed": rollout.seed,
        "win": rollout.outcome.winner == rollout.hero,
        "hero_vp": hero_vps(rollout),
        "vp_margin": margin(rollout),
        "truncated": rollout.outcome.truncated,
        "invalid_replies": rollout.invalid_replies,
        "decisions": rollout.decision_states,
    }


def run_cell(sampler, temperature, table, seeds, run_seed, out_path, done):
    todo = [seed for seed in seeds if (temperature, table, seed) not in done]
    for start in range(0, len(todo), BATCH):
        chunk = todo[start : start + BATCH]
        rollouts = make_rollouts(chunk, 1, run_seed, opponents=TABLES[table])
        rollout_games(rollouts, sampler)
        with open(out_path, "a") as handle:
            for rollout in rollouts:
                handle.write(json.dumps(game_row(temperature, table, rollout)) + "\n")
        print(f"t={temperature:g} {table}: {start + len(chunk)}/{len(todo)}", flush=True)


def summarize(out_path: Path) -> dict:
    cells = {}
    for row in map(json.loads, out_path.read_text().splitlines()):
        if "win" not in row:
            continue
        cell = cells.setdefault(
            (row["temperature"], row["table"]),
            {"games": 0, "wins": 0, "margins": [], "invalid": 0, "decisions": 0},
        )
        cell["games"] += 1
        cell["wins"] += bool(row["win"])
        cell["margins"].append(row["vp_margin"])
        cell["invalid"] += row["invalid_replies"]
        cell["decisions"] += row["decisions"]
    summary = {}
    for (temperature, table), cell in sorted(cells.items()):
        lo, hi = wilson(cell["wins"], cell["games"])
        summary[f"t{temperature:g}_{table}"] = {
            "games": cell["games"],
            "wins": cell["wins"],
            "win_rate": cell["wins"] / cell["games"],
            "win_rate_ci95": [lo, hi],
            "vp_margin_mean": sum(cell["margins"]) / len(cell["margins"]),
            "invalid_rate": cell["invalid"] / max(cell["decisions"], 1),
        }
    return summary


def main() -> None:
    check_fixed_hashseed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temps", type=float, nargs="+", default=list(DEFAULT_TEMPS))
    parser.add_argument(
        "--tables", nargs="+", default=list(TABLES), choices=list(TABLES)
    )
    parser.add_argument("--games-per-cell", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--run-seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_text_only_qwen35(args.model).to("cuda")

    args.output.mkdir(parents=True, exist_ok=True)
    games_path = args.output / "games.jsonl"
    seeds = training_seeds(args.seed_start, 0, args.games_per_cell)
    done = existing_rows(games_path)
    for temperature in args.temps:
        sampler = ModelSampler(model, tokenizer, temperature)
        for table in args.tables:
            run_cell(sampler, temperature, table, seeds, args.run_seed, games_path, done)
    summary = summarize(games_path)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
