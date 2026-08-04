"""Score every legal move at logged LLM decisions with the alpha-beta teacher.

The output is raw evidence, not a preference dataset: one JSON object per
non-forced LLM decision with every move id and its teacher value. Pair
selection, filtering, and weighting belong in a separate downstream step.
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from catanatron.players.minimax import DebugStateNode
from catanatron.players.tree_search_utils import expand_spectrum

from catan_llm.bots import BOTS
from catan_llm.determinism import check_fixed_hashseed
from catan_llm.determinize import determinize
from catan_llm.extract import to_action
from catan_llm.schema import ActionType
from catan_llm.serialize import move_id
from teacher_agreement import replay_game


SCHEMA_VERSION = 1
ALPHA_BETA_BOTS = sorted(name for name in BOTS if "alpha_beta" in name)


def _action_id(action, catan_map) -> str:
    action_type, payload = to_action(action, catan_map)
    return move_id(ActionType(action_type), payload)


def score_options(teacher, game, playable, seconds_per_action: float):
    """Return independently searched values for every root action.

    Each root action starts with a fresh alpha-beta window. This avoids carrying
    bounds from one sibling into the next, which is necessary when callers need
    all action values rather than only the argmax.
    """
    action_outcomes = expand_spectrum(game.copy(), playable)
    if list(action_outcomes) != list(playable):
        raise RuntimeError("teacher search reordered or dropped legal actions")

    rows = []
    for action, outcomes in action_outcomes.items():
        deadline = time.time() + seconds_per_action
        expected_value = 0.0
        for outcome, probability in outcomes:
            node = DebugStateNode("score", outcome.state.current_color())
            _, value = teacher.alphabeta(
                outcome,
                teacher.depth - 1,
                float("-inf"),
                float("inf"),
                deadline,
                node,
            )
            expected_value += probability * value
        rows.append(
            {
                "id": _action_id(action, game.state.board.map),
                "value": expected_value,
                "timed_out": time.time() >= deadline,
            }
        )
    return rows


def score_options_fair(
    teacher, game, playable, seconds_per_action: float, worlds: int, world_seed: int
):
    """Return per-action values averaged over resampled hidden worlds.

    Each world redraws the opponents' hidden cards from the actor's
    information set, so values never condition on hidden state.
    """
    hero_index = game.state.colors.index(game.state.current_color())
    seeds = random.Random(
        f"{world_seed}:{game.id}:{len(game.state.action_records)}"
    )
    totals = None
    for _ in range(worlds):
        world = determinize(game, hero_index, seeds.randrange(2**63))
        rows = score_options(
            teacher, world, world.playable_actions, seconds_per_action
        )
        if totals is None:
            totals = rows
            continue
        for total, row in zip(totals, rows):
            if total["id"] != row["id"]:
                raise RuntimeError("world resampling changed the legal actions")
            total["value"] += row["value"]
            total["timed_out"] = total["timed_out"] or row["timed_out"]
    for total in totals:
        total["value"] /= worlds
    return totals


def score_game(
    path: str,
    teacher_name: str,
    seconds_per_action: float,
    worlds: int = 0,
    world_seed: int = 42,
) -> list[dict]:
    """Replay and score one trajectory. Safe to call in a worker process."""
    trajectory = Path(path)

    def probe(teacher, game, playable):
        if worlds:
            return score_options_fair(
                teacher, game, playable, seconds_per_action, worlds, world_seed
            )
        return score_options(teacher, game, playable, seconds_per_action)

    rows = []
    for replayed, options in replay_game(
        trajectory, BOTS[teacher_name], probe=probe
    ):
        decision = replayed.decision
        chosen_index = decision.chosen_action
        chosen = options[chosen_index]
        best_value = max(option["value"] for option in options)
        best_index = next(
            index for index, option in enumerate(options) if option["value"] == best_value
        )
        best = options[best_index]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "game_id": decision.game_id,
                "seed": replayed.game_record.seed,
                "decision": decision.i,
                "turn": decision.turn,
                "phase": decision.phase.value,
                "actor": decision.actor.value,
                "move_type": decision.legal_actions[chosen_index][0].value,
                "chosen_index": chosen_index,
                "chosen_id": chosen["id"],
                "chosen_value": chosen["value"],
                "best_index": best_index,
                "best_id": best["id"],
                "best_value": best["value"],
                "strict_preference": best["value"] > chosen["value"],
                "any_timeout": any(option["timed_out"] for option in options),
                "options": options,
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate best-pick and strict-preference rates by move type."""
    counts = defaultdict(
        lambda: {
            "decisions": 0,
            "usable": 0,
            "timed_out": 0,
            "best_picks": 0,
            "strict_pairs": 0,
        }
    )
    for row in rows:
        bucket = counts[row["move_type"]]
        bucket["decisions"] += 1
        if row["any_timeout"]:
            bucket["timed_out"] += 1
            continue
        bucket["usable"] += 1
        bucket["best_picks"] += not row["strict_preference"]
        bucket["strict_pairs"] += row["strict_preference"]

    summary = []
    for move_type, bucket in sorted(counts.items()):
        usable = bucket["usable"]
        summary.append(
            {
                "move_type": move_type,
                **bucket,
                "best_pick_rate": bucket["best_picks"] / usable if usable else None,
                "strict_pair_rate": bucket["strict_pairs"] / usable if usable else None,
            }
        )
    return summary


def _print_summary(rows: list[dict]) -> None:
    timeouts = sum(row["any_timeout"] for row in rows)
    pairs = sum(row["strict_preference"] and not row["any_timeout"] for row in rows)
    print(f"decisions={len(rows)} strict_pairs={pairs} timed_out={timeouts}")
    print(f"{'move type':<22} {'decisions':>9} {'timeouts':>9} {'best':>8} {'pairs':>8}")
    for item in summarize(rows):
        best = f"{item['best_pick_rate']:.1%}" if item["best_pick_rate"] is not None else "n/a"
        pairs = (
            f"{item['strict_pair_rate']:.1%}" if item["strict_pair_rate"] is not None else "n/a"
        )
        print(
            f"{item['move_type']:<22} {item['decisions']:>9} {item['timed_out']:>9} "
            f"{best:>8} {pairs:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher", default="alpha_beta", choices=ALPHA_BETA_BOTS)
    parser.add_argument(
        "--seconds-per-action",
        required=True,
        type=float,
        help="search deadline for each legal action; recorded in OUT.meta.json",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-games", type=int)
    parser.add_argument(
        "--worlds",
        type=int,
        default=0,
        help="hidden worlds averaged per decision; 0 scores the true state",
    )
    parser.add_argument("--world-seed", type=int, default=42)
    args = parser.parse_args()

    check_fixed_hashseed()
    if not math.isfinite(args.seconds_per_action) or args.seconds_per_action <= 0:
        parser.error("--seconds-per-action must be a positive finite number")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    paths = sorted(Path(args.trajectories).glob("*.jsonl"))
    if args.max_games is not None:
        paths = paths[: args.max_games]
    if not paths:
        parser.error(f"no .jsonl trajectories found in {args.trajectories}")

    worker_args = [
        (str(path), args.teacher, args.seconds_per_action, args.worlds, args.world_seed)
        for path in paths
    ]
    if args.workers == 1:
        games = [score_game(*worker_arg) for worker_arg in worker_args]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            games = list(pool.map(_score_game_star, worker_args))
    rows = [row for game_rows in games for row in game_rows]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "trajectories": str(Path(args.trajectories)),
        "teacher": args.teacher,
        "seconds_per_action": args.seconds_per_action,
        "worlds": args.worlds,
        "world_seed": args.world_seed,
        "workers": args.workers,
        "games": len(paths),
        "decisions": len(rows),
        "strict_pairs_without_timeouts": sum(
            row["strict_preference"] and not row["any_timeout"] for row in rows
        ),
        "timed_out_decisions": sum(row["any_timeout"] for row in rows),
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    _print_summary(rows)
    print(f"-> {out}")


def _score_game_star(args):
    return score_game(*args)


if __name__ == "__main__":
    main()
