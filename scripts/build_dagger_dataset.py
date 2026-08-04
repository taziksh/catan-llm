"""Builds an SFT chat dataset by labeling the model's own states with the teacher's move.

One sample per non-forced LLM decision in a logged game: the eval prompt as the
user message, "answer: <teacher move id>" as the assistant target. Games are
split train/val by seed so decisions from one game never span both.
"""

import argparse
import json
from pathlib import Path

from teacher_agreement import replay_game

from catan_llm.bots import BOTS
from catan_llm.determinism import EVAL_SEED_LIMIT, check_fixed_hashseed
from catan_llm.schema import GameRecord
from catan_llm.serialize import decision_to_prompt, move_id
from catan_v1.taskset import SYSTEM_PROMPT


def game_samples(game, path, teacher_cls):
    for replayed, action in replay_game(path, teacher_cls):
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": decision_to_prompt(game, replayed.decision),
                },
                {"role": "assistant", "content": f"answer: {move_id(*action)}"},
            ]
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True, help="dir of logged llm games")
    parser.add_argument("--teacher", default="alpha_beta")
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="games with seed %% VAL_EVERY == 0 go to val (default: %(default)s)",
    )
    parser.add_argument("--out", default="data/dagger")
    args = parser.parse_args()
    check_fixed_hashseed()

    teacher_cls = BOTS[args.teacher]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    writers = {name: open(out / f"{name}.jsonl", "w") for name in counts}
    games = skipped = 0
    for path in sorted(Path(args.trajectories).glob("*.jsonl")):
        game = GameRecord.model_validate_json(path.read_text().splitlines()[0])
        if game.seed is None or game.seed < EVAL_SEED_LIMIT:
            skipped += 1
            continue
        split = "val" if game.seed % args.val_every == 0 else "train"
        wrote = 0
        for sample in game_samples(game, path, teacher_cls):
            writers[split].write(json.dumps(sample) + "\n")
            wrote += 1
        counts[split] += wrote
        games += bool(wrote)
    for writer in writers.values():
        writer.close()
    print(f"{games} games -> train={counts['train']} val={counts['val']} -> {out}")
    if skipped:
        print(f"skipped {skipped} games with seeds inside the eval range")


if __name__ == "__main__":
    main()
