"""Builds an SFT chat dataset from logged teacher trajectories.

One sample per non-forced teacher decision: the eval prompt as the user
message, "answer: <chosen index>" as the assistant target. Games are split
train/val by seed so decisions from one game never span both.
"""

import argparse
import json
from pathlib import Path

from catan_llm.determinism import EVAL_SEED_LIMIT
from catan_llm.schema import DecisionRecord, GameRecord
from catan_llm.serialize import decision_to_prompt
from catan_v1.taskset import SYSTEM_PROMPT


def game_samples(game, lines, teacher):
    color = next((c for c, name in game.seats.items() if name == teacher), None)
    if color is None:
        return
    for line in lines:
        decision = DecisionRecord.model_validate_json(line)
        if decision.actor != color or len(decision.legal_actions) < 2:
            continue
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": decision_to_prompt(game, decision)},
                {
                    "role": "assistant",
                    "content": f"answer: {decision.chosen_action}",
                },
            ]
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="data/games")
    parser.add_argument("--teacher", default="alpha_beta")
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="games with seed %% VAL_EVERY == 0 go to val (default: %(default)s)",
    )
    parser.add_argument("--out", default="data/sft")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="at least this many train samples, whole games only",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    writers = {name: open(out / f"{name}.jsonl", "w") for name in counts}
    games = 0
    for path in sorted(Path(args.games).glob("*.jsonl")):
        if args.samples is not None and counts["train"] >= args.samples:
            break
        lines = path.read_text().splitlines()
        game = GameRecord.model_validate_json(lines[0])
        if game.seed is None or game.seed < EVAL_SEED_LIMIT:
            continue
        split = "val" if game.seed % args.val_every == 0 else "train"
        wrote = 0
        for sample in game_samples(game, lines[1:], args.teacher):
            writers[split].write(json.dumps(sample) + "\n")
            wrote += 1
        counts[split] += wrote
        games += bool(wrote)
    for writer in writers.values():
        writer.close()
    if args.samples is not None and counts["train"] < args.samples:
        raise SystemExit(
            f"asked for {args.samples} train samples, only {counts['train']} available"
        )
    print(f"{games} games -> train={counts['train']} val={counts['val']} -> {out}")


if __name__ == "__main__":
    main()
