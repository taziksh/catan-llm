"""Audit label stability of DPO pairs across determinized world counts.

Rescores a stratified sample of strict pairs with the alpha-beta teacher
in K resampled hidden worlds. World seeds nest, so smaller world counts
reuse the first worlds of larger ones (common random numbers across K).
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from catan_llm.bots import BOTS
from catan_llm.determinism import check_fixed_hashseed
from catan_llm.determinize import determinize_pair
from catan_llm.replay import replay_model_decisions
from teacher_values import _action_id, score_options


SCHEMA_VERSION = 1
TEACHER = "alpha_beta"
FULL_REBUILD_PAIRS = 4848
CLASSES = ("robber", "other")


def answer_move(pair: dict, role: str) -> str:
    content = pair[role][0]["content"]
    move = content.removeprefix("answer: ")
    if move == content:
        raise ValueError(f"malformed {role} completion: {content!r}")
    return move


def is_robber(pair: dict) -> bool:
    return answer_move(pair, "chosen").startswith("robber:")


def parse_worlds(text: str) -> list[int]:
    try:
        worlds = [int(part) for part in text.split(",")]
    except ValueError:
        raise ValueError(f"invalid --worlds value: {text!r}") from None
    if not worlds or any(count <= 0 for count in worlds):
        raise ValueError("--worlds needs positive world counts")
    if len(set(worlds)) != len(worlds):
        raise ValueError("--worlds has duplicate world counts")
    return sorted(worlds)


def sample_pairs(pairs: list[dict], sample: int, seed: int) -> list[int]:
    """Sample half from robber pairs and half uniformly from the rest."""
    robber = [index for index, pair in enumerate(pairs) if is_robber(pair)]
    other = [index for index, pair in enumerate(pairs) if not is_robber(pair)]
    take_robber = min(len(robber), sample // 2)
    take_other = min(len(other), sample - take_robber)
    if take_robber + take_other < sample:
        take_robber = min(len(robber), sample - take_other)
    rng = random.Random(seed)
    return sorted(
        rng.sample(robber, take_robber) + rng.sample(other, take_other)
    )


def world_seeds(seed: int, game_id: str, decision: int, count: int) -> list[int]:
    """Draw unique world seeds from one per-pair stream.

    Lower counts are prefixes of higher ones, and string seeding does not
    depend on hash randomization.
    """
    rng = random.Random(f"{seed}:{game_id}:{decision}")
    seeds = []
    seen = set()
    while len(seeds) < count:
        candidate = rng.getrandbits(32)
        if candidate not in seen:
            seen.add(candidate)
            seeds.append(candidate)
    return seeds


def replay_decisions(path: Path, wanted: list[int]) -> dict:
    """Collect the wanted decision states in one replay pass."""
    remaining = set(wanted)
    found = {}
    for replayed in replay_model_decisions(path):
        if replayed.decision.i in remaining:
            found[replayed.decision.i] = replayed
            remaining.remove(replayed.decision.i)
            if not remaining:
                break
    if remaining:
        raise ValueError(
            f"{path.name}: decisions not replayed: {sorted(remaining)}"
        )
    return found


def score_pair_worlds(
    replayed, chosen_id: str, rejected_id: str, seeds, seconds_per_action: float
) -> list[dict]:
    """Score both moves of a pair in every determinized world."""
    game = replayed.game
    llm_color = next(
        color for color, kind in replayed.game_record.seats.items()
        if kind == "llm"
    )
    hero = [color.value for color in game.state.colors].index(llm_color.value)
    teacher = BOTS[TEACHER](game.state.colors[hero])
    results = []
    for world in determinize_pair(game, hero, seeds):
        catan_map = world.state.board.map
        actions = {
            _action_id(action, catan_map): action
            for action in world.playable_actions
        }
        missing = {chosen_id, rejected_id} - actions.keys()
        if missing:
            raise ValueError(
                f"moves not legal in sampled world: {sorted(missing)}"
            )
        chosen_row, rejected_row = score_options(
            teacher,
            world,
            [actions[chosen_id], actions[rejected_id]],
            seconds_per_action,
        )
        results.append(
            {
                "chosen_value": chosen_row["value"],
                "rejected_value": rejected_row["value"],
                "timed_out": chosen_row["timed_out"] or rejected_row["timed_out"],
            }
        )
    return results


def aggregate(world_scores: list[dict], worlds: int) -> dict:
    """Label the pair from its first ``worlds`` world scores."""
    prefix = world_scores[:worlds]
    if len(prefix) != worlds:
        raise ValueError(f"need {worlds} world scores, have {len(prefix)}")
    diffs = [
        score["chosen_value"] - score["rejected_value"] for score in prefix
    ]
    mean_diff = sum(diffs) / worlds
    if mean_diff > 0:
        label = "RETAINED"
    elif mean_diff < 0:
        label = "REVERSED"
    else:
        label = "TIE"
    return {
        "worlds": worlds,
        "mean_value_diff": mean_diff,
        "fraction_preferring_chosen": sum(d > 0 for d in diffs) / worlds,
        "label": label,
        "unanimous": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
        "deadline_hits": sum(score["timed_out"] for score in prefix),
    }


def run_audit(
    pairs: list[dict],
    traces: Path,
    worlds: list[int],
    sample: int,
    seed: int,
    seconds_per_action: float,
) -> tuple[list[dict], list[dict]]:
    """Score the sampled pairs; per-pair errors become failure rows."""
    sampled = [pairs[index] for index in sample_pairs(pairs, sample, seed)]
    by_game = defaultdict(list)
    for pair in sampled:
        by_game[pair["game_id"]].append(pair)

    max_worlds = worlds[-1]
    rows, failures = [], []
    for game_id in sorted(by_game):
        game_pairs = by_game[game_id]

        def fail(pair, error):
            failures.append(
                {
                    "game_id": pair["game_id"],
                    "decision": pair["decision"],
                    "class": "robber" if is_robber(pair) else "other",
                    "error": str(error),
                }
            )

        started = time.perf_counter()
        try:
            replayed = replay_decisions(
                traces / f"{game_id}.jsonl",
                [pair["decision"] for pair in game_pairs],
            )
        except (OSError, ValueError, RuntimeError) as error:
            for pair in game_pairs:
                fail(pair, error)
            continue
        replay_seconds = (time.perf_counter() - started) / len(game_pairs)

        for pair in game_pairs:
            chosen_id = answer_move(pair, "chosen")
            rejected_id = answer_move(pair, "rejected")
            seeds = world_seeds(seed, game_id, pair["decision"], max_worlds)
            started = time.perf_counter()
            try:
                scores = score_pair_worlds(
                    replayed[pair["decision"]],
                    chosen_id,
                    rejected_id,
                    seeds,
                    seconds_per_action,
                )
            except (ValueError, RuntimeError) as error:
                fail(pair, error)
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "decision": pair["decision"],
                    "class": "robber" if is_robber(pair) else "other",
                    "chosen_id": chosen_id,
                    "rejected_id": rejected_id,
                    "original_gap": pair.get("teacher_value_gap"),
                    "per_worlds": {
                        count: aggregate(scores, count) for count in worlds
                    },
                    "replay_seconds": replay_seconds,
                    "score_seconds": time.perf_counter() - started,
                }
            )
    return rows, failures


def label_summary(rows: list[dict], worlds: list[int]) -> list[dict]:
    """Aggregate label survival by world count and action class."""
    summary = []
    for count in worlds:
        for klass in CLASSES:
            stats = [
                row["per_worlds"][count]
                for row in rows
                if row["class"] == klass
            ]
            summary.append(
                {
                    "worlds": count,
                    "class": klass,
                    "pairs": len(stats),
                    "retained": sum(s["label"] == "RETAINED" for s in stats),
                    "reversed": sum(s["label"] == "REVERSED" for s in stats),
                    "ties": sum(s["label"] == "TIE" for s in stats),
                    "non_unanimous": sum(not s["unanimous"] for s in stats),
                    "deadline_hits": sum(s["deadline_hits"] for s in stats),
                    "mean_abs_value_gap": (
                        sum(abs(s["mean_value_diff"]) for s in stats)
                        / len(stats)
                        if stats
                        else None
                    ),
                }
            )
    return summary


def convergence(rows: list[dict], worlds: list[int]) -> dict | None:
    """Compare labels between the lowest and highest world count."""
    if len(worlds) < 2:
        return None
    low, high = worlds[0], worlds[-1]
    result = {"low_worlds": low, "high_worlds": high}
    for klass in (*CLASSES, "all"):
        subset = [
            row for row in rows if klass == "all" or row["class"] == klass
        ]
        agreements = sum(
            row["per_worlds"][low]["label"] == row["per_worlds"][high]["label"]
            for row in subset
        )
        result[klass] = {
            "pairs": len(subset),
            "agreement": agreements / len(subset) if subset else None,
        }
    return result


def timing_stats(rows: list[dict], worlds: list[int]) -> dict | None:
    """Project full-rebuild wall time from the measured per-pair costs."""
    if not rows:
        return None
    replay = sum(row["replay_seconds"] for row in rows) / len(rows)
    world = sum(row["score_seconds"] for row in rows) / len(rows) / worlds[-1]
    return {
        "mean_replay_seconds_per_pair": replay,
        "mean_score_seconds_per_world": world,
        "mean_seconds_per_pair_at_max_worlds": replay + world * worlds[-1],
        "projected_full_rebuild_seconds": {
            count: FULL_REBUILD_PAIRS * (replay + world * count)
            for count in worlds
        },
    }


def build_report(
    rows: list[dict], failures: list[dict], worlds: list[int], config: dict
) -> dict:
    deadline_hits = {
        count: sum(row["per_worlds"][count]["deadline_hits"] for row in rows)
        for count in worlds
    }
    warnings = []
    if deadline_hits[worlds[-1]]:
        warnings.append(
            f"{deadline_hits[worlds[-1]]} world scorings hit the search "
            "deadline; deadline-bound values depend on wall-clock speed, so "
            "affected labels may differ across runs"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "pairs_scored": len(rows),
        "pairs_failed": len(failures),
        "sampled_by_class": {
            klass: sum(record["class"] == klass for record in rows + failures)
            for klass in CLASSES
        },
        "summary": label_summary(rows, worlds),
        "convergence": convergence(rows, worlds),
        "deadline_hits": deadline_hits,
        "timing": timing_stats(rows, worlds),
        "warnings": warnings,
        "failures": failures,
        "pairs": rows,
    }


def _print_report(report: dict) -> None:
    print(
        f"{'worlds':>6} {'class':<7} {'pairs':>5} {'retained':>8} "
        f"{'reversed':>8} {'ties':>5} {'nonunan':>7} {'deadline':>8} "
        f"{'mean|gap|':>10}"
    )
    for entry in report["summary"]:
        gap = (
            f"{entry['mean_abs_value_gap']:.3f}"
            if entry["mean_abs_value_gap"] is not None
            else "n/a"
        )
        print(
            f"{entry['worlds']:>6} {entry['class']:<7} {entry['pairs']:>5} "
            f"{entry['retained']:>8} {entry['reversed']:>8} "
            f"{entry['ties']:>5} {entry['non_unanimous']:>7} "
            f"{entry['deadline_hits']:>8} {gap:>10}"
        )
    conv = report["convergence"]
    if conv is not None:
        parts = [
            f"{klass}={conv[klass]['agreement']:.0%}"
            if conv[klass]["agreement"] is not None
            else f"{klass}=n/a"
            for klass in (*CLASSES, "all")
        ]
        print(
            f"label agreement K={conv['low_worlds']} vs "
            f"K={conv['high_worlds']}: " + " ".join(parts)
        )
    timing = report["timing"]
    if timing is not None:
        print(
            f"per pair: replay {timing['mean_replay_seconds_per_pair']:.2f}s "
            f"+ {timing['mean_score_seconds_per_world']:.2f}s/world"
        )
        for count, seconds in timing["projected_full_rebuild_seconds"].items():
            print(
                f"projected {FULL_REBUILD_PAIRS}-pair rebuild at K={count}: "
                f"{seconds / 3600:.1f}h"
            )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    if report["pairs_failed"]:
        print(f"failed pairs: {report['pairs_failed']}")


def load_pairs(path: Path) -> list[dict]:
    pairs = [json.loads(line) for line in path.read_text().splitlines()]
    if not pairs:
        raise ValueError(f"{path}: no pairs")
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/dpo/r3pairs/train.jsonl")
    parser.add_argument("--traces", default="data/dagger_traces/r3pairs")
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument(
        "--worlds",
        default="4,8,16",
        help="comma-separated world counts; seeds nest across counts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument(
        "--seconds-per-action",
        type=float,
        default=5.0,
        help="search deadline for each scored action, as in teacher_values",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    check_fixed_hashseed()
    try:
        worlds = parse_worlds(args.worlds)
    except ValueError as error:
        parser.error(str(error))
    if args.sample <= 0:
        parser.error("--sample must be positive")
    if not math.isfinite(args.seconds_per_action) or args.seconds_per_action <= 0:
        parser.error("--seconds-per-action must be a positive finite number")

    pairs = load_pairs(Path(args.pairs))
    started = time.perf_counter()
    rows, failures = run_audit(
        pairs,
        Path(args.traces),
        worlds,
        args.sample,
        args.seed,
        args.seconds_per_action,
    )
    config = {
        "pairs": str(args.pairs),
        "traces": str(args.traces),
        "sample": args.sample,
        "worlds": worlds,
        "seed": args.seed,
        "seconds_per_action": args.seconds_per_action,
        "teacher": TEACHER,
        "total_pairs": len(pairs),
        "audit_seconds": time.perf_counter() - started,
    }
    report = build_report(rows, failures, worlds, config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(report)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
