"""Rebuild the DPO dataset keeping only information-set-stable pairs.

Rescores every strict pair with the alpha-beta teacher in K determinized
hidden worlds, reusing the replay, world-seed, and scoring machinery of
``label_stability_audit``. A pair is kept only when the K-world mean value
difference prefers the original chosen (teacher) move and at least
``min_fraction`` of worlds agree. Retained pairs are copied byte for byte
in their original order, so membership is the only change; the train/val
game-boundary split is inherited from the source files.
"""

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from catan_llm.determinism import check_fixed_hashseed

from build_dpo_dataset import _sha256
from label_stability_audit import (
    TEACHER,
    aggregate,
    answer_move,
    replay_decisions,
    score_pair_worlds,
    world_seeds,
)


SCHEMA_VERSION = 1
SPLITS = ("train", "val")
WORLD_SEED_STREAM = 'random.Random(f"{seed}:{game_id}:{decision}")'
COUNT_KEYS = (
    "input",
    "kept",
    "dropped",
    "dropped_reversed",
    "dropped_tie",
    "dropped_low_fraction",
    "dropped_failed",
    "deadline_hits",
)


def drop_reason(result: dict, min_fraction: float) -> str | None:
    """Why the retention rule drops a scored pair, or None to keep it.

    A pair survives only when the mean world value difference prefers the
    original chosen move and at least ``min_fraction`` of worlds agree.
    """
    if result["label"] == "REVERSED":
        return "reversed"
    if result["label"] == "TIE":
        return "tie"
    if result["fraction_preferring_chosen"] < min_fraction:
        return "low_fraction"
    return None


def rescore_pairs(
    pairs: list[dict],
    traces: Path,
    worlds: int,
    seed: int,
    seconds_per_action: float,
    log,
) -> list[dict]:
    """Score each pair in ``worlds`` hidden worlds, one replay per game.

    Returns one result per pair in input order: the aggregate row from
    ``label_stability_audit.aggregate``, or ``{"error": ...}`` when replay
    or scoring failed.
    """
    by_game = defaultdict(list)
    for index, pair in enumerate(pairs):
        by_game[pair["game_id"]].append(index)

    results: list[dict | None] = [None] * len(pairs)
    for done, game_id in enumerate(sorted(by_game), 1):
        indices = by_game[game_id]
        try:
            replayed = replay_decisions(
                traces / f"{game_id}.jsonl",
                [pairs[index]["decision"] for index in indices],
            )
        except (OSError, ValueError, RuntimeError) as error:
            for index in indices:
                results[index] = {"error": str(error)}
            log(done, len(by_game), game_id)
            continue
        for index in indices:
            pair = pairs[index]
            seeds = world_seeds(seed, game_id, pair["decision"], worlds)
            try:
                scores = score_pair_worlds(
                    replayed[pair["decision"]],
                    answer_move(pair, "chosen"),
                    answer_move(pair, "rejected"),
                    seeds,
                    seconds_per_action,
                )
            except (ValueError, RuntimeError) as error:
                results[index] = {"error": str(error)}
                continue
            results[index] = aggregate(scores, worlds)
        log(done, len(by_game), game_id)
    return results


def apply_retention(
    split: str,
    lines: list[bytes],
    pairs: list[dict],
    results: list[dict],
    min_fraction: float,
) -> tuple[list[bytes], dict, dict, list[dict]]:
    """Filter one split's raw lines by the retention rule.

    Returns the kept lines (byte-identical, original order), the count
    table, the per-move-type composition table, and the dropped-pair
    evidence rows.
    """
    kept_lines = []
    counts = dict.fromkeys(COUNT_KEYS, 0)
    composition = defaultdict(Counter)
    dropped = []
    for line, pair, result in zip(lines, pairs, results, strict=True):
        klass = pair["move_type"]
        counts["input"] += 1
        composition[klass]["input"] += 1
        if "error" in result:
            reason = "failed"
        else:
            counts["deadline_hits"] += result["deadline_hits"]
            reason = drop_reason(result, min_fraction)
        if reason is None:
            counts["kept"] += 1
            composition[klass]["kept"] += 1
            kept_lines.append(line)
            continue
        counts["dropped"] += 1
        counts[f"dropped_{reason}"] += 1
        composition[klass]["dropped"] += 1
        dropped.append(
            {
                "split": split,
                "game_id": pair["game_id"],
                "decision": pair["decision"],
                "class": klass,
                "label": result.get("label", "FAILED"),
                "mean_value_diff": result.get("mean_value_diff"),
                "fraction_preferring_chosen": result.get(
                    "fraction_preferring_chosen"
                ),
                "deadline_hits": result.get("deadline_hits"),
                "reason": reason,
                **({"error": result["error"]} if "error" in result else {}),
            }
        )
    if counts["kept"] + counts["dropped"] != counts["input"]:
        raise AssertionError(f"{split}: kept + dropped != input")
    return kept_lines, counts, dict(composition), dropped


def rebuild(
    train: Path,
    val: Path,
    traces: Path,
    out: Path,
    worlds: int,
    min_fraction: float,
    seed: int,
    seconds_per_action: float,
) -> dict:
    """Rescore both splits, write the filtered dataset, return metadata."""
    started = time.perf_counter()
    inputs = {"train": train, "val": val}
    out.mkdir(parents=True, exist_ok=True)
    for split, path in inputs.items():
        if (out / f"{split}.jsonl").resolve() == path.resolve():
            raise ValueError(f"--out would overwrite the input {path}")

    split_counts = {}
    composition = {}
    dropped_rows = []
    for split, path in inputs.items():
        lines = path.read_bytes().splitlines()
        pairs = [json.loads(line) for line in lines]
        if not pairs:
            raise ValueError(f"{path}: no pairs")

        def log(done, total, game_id, split=split):
            print(
                f"[{split}] {done}/{total} games ({game_id}) "
                f"{time.perf_counter() - started:.0f}s elapsed",
                flush=True,
            )

        results = rescore_pairs(
            pairs, traces, worlds, seed, seconds_per_action, log
        )
        kept_lines, counts, split_composition, dropped = apply_retention(
            split, lines, pairs, results, min_fraction
        )
        (out / f"{split}.jsonl").write_bytes(
            b"".join(line + b"\n" for line in kept_lines)
        )
        split_counts[split] = counts
        composition[split] = {
            klass: {key: stats[key] for key in ("input", "kept", "dropped")}
            for klass, stats in sorted(split_composition.items())
        }
        dropped_rows.extend(dropped)

    dropped_path = out / "dropped_pairs.jsonl"
    dropped_path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in dropped_rows
        )
    )

    deadline_hits = sum(
        counts["deadline_hits"] for counts in split_counts.values()
    )
    warnings = []
    if deadline_hits:
        warnings.append(
            f"{deadline_hits} world scorings hit the search deadline; "
            "deadline-bound values depend on wall-clock speed, so affected "
            "labels may differ across runs"
        )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "parameters": {
            "worlds": worlds,
            "min_fraction": min_fraction,
            "seed": seed,
            "world_seed_stream": WORLD_SEED_STREAM,
            "seconds_per_action": seconds_per_action,
            "teacher": TEACHER,
        },
        "inputs": {
            "train": str(train),
            "train_sha256": _sha256(train),
            "val": str(val),
            "val_sha256": _sha256(val),
            "trajectories": str(traces),
        },
        "outputs": {
            "train_sha256": _sha256(out / "train.jsonl"),
            "val_sha256": _sha256(out / "val.jsonl"),
            "dropped_pairs_sha256": _sha256(dropped_path),
        },
        "splits": split_counts,
        "composition": composition,
        "reversed_pairs": sum(
            counts["dropped_reversed"] for counts in split_counts.values()
        ),
        "failed_pairs": sum(
            counts["dropped_failed"] for counts in split_counts.values()
        ),
        "deadline_hits": deadline_hits,
        "warnings": warnings,
        "wall_seconds": time.perf_counter() - started,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/dpo/r3pairs/train.jsonl")
    parser.add_argument("--val", default="data/dpo/r3pairs/val.jsonl")
    parser.add_argument("--traces", default="data/dagger_traces/r3pairs")
    parser.add_argument("--out", default="data/dpo/r3pairs_repaired")
    parser.add_argument(
        "--worlds",
        type=int,
        default=16,
        help="determinized hidden worlds sampled per pair",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.75,
        help="minimum fraction of worlds preferring the chosen move",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    if args.worlds <= 0:
        parser.error("--worlds must be positive")
    if not 0 < args.min_fraction <= 1:
        parser.error("--min-fraction must be in (0, 1]")
    if not math.isfinite(args.seconds_per_action) or args.seconds_per_action <= 0:
        parser.error("--seconds-per-action must be a positive finite number")
    metadata = rebuild(
        Path(args.train),
        Path(args.val),
        Path(args.traces),
        Path(args.out),
        args.worlds,
        args.min_fraction,
        args.seed,
        args.seconds_per_action,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
