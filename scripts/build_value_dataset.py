"""Builds an SFT dataset where the answer is the best move by playout reward."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_reward_cache import load_cache
from catan_llm.schema import DecisionRecord, GameRecord
from catan_llm.serialize import decision_to_prompt, move_id
from catan_v1.taskset import SYSTEM_PROMPT


def best_moves(cache_path):
    """(game_id, decision) -> (best move, margin over runner-up)."""
    rewards = defaultdict(lambda: defaultdict(list))
    for (game_id, decision, move, _), row in load_cache(Path(cache_path)).items():
        rewards[game_id, decision][move].append(row["reward"])
    best = {}
    for key, moves in rewards.items():
        means = sorted(((sum(r) / len(r), m) for m, r in moves.items()), reverse=True)
        margin = means[0][0] - means[1][0] if len(means) > 1 else float("inf")
        best[key] = (means[0][1], margin)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="data/dagger_traces/r3pairs")
    parser.add_argument("--cache", default="data/reward_cache/local_fill.jsonl")
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--val-every", type=int, default=10, help="every Nth game goes to val")
    parser.add_argument("--out", default="data/sft_value")
    args = parser.parse_args()

    labels = best_moves(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    types = Counter()
    writers = {name: open(out / f"{name}.jsonl", "w") for name in ("train", "val")}
    games = 0
    for path in sorted(Path(args.games).glob("*.jsonl")):
        lines = path.read_text().splitlines()
        game = GameRecord.model_validate_json(lines[0])
        hero = next((c for c, name in game.seats.items() if name == "llm"), None)
        if hero is None:
            continue
        decisions = [
            DecisionRecord.model_validate_json(line)
            for line in lines[1:]
            if json.loads(line).get("type") == "decision"
        ]
        cached = [d for d in decisions if d.actor == hero and (game.game_id, d.i) in labels]
        if not cached:
            continue
        split = "val" if games % args.val_every == 0 else "train"
        games += 1
        for decision in cached:
            best, margin = labels[game.game_id, decision.i]
            counts["seen"] += 1
            if margin < args.min_margin:
                counts["near_tie"] += 1
                continue
            played = move_id(*decision.legal_actions[decision.chosen_action])
            counts["corrected" if best != played else "kept"] += 1
            types[best.split(":")[0]] += 1
            counts[split] += 1
            writers[split].write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": decision_to_prompt(game, decision)},
                    {"role": "assistant", "content": f"answer: {best}"},
                ]
            }) + "\n")
    for writer in writers.values():
        writer.close()

    print(f"{games} games, {counts['seen']} cached decisions")
    print(f"near-ties dropped: {counts['near_tie']}")
    print(f"label same as played: {counts['kept']}, corrected: {counts['corrected']}")
    print("labels by move type:", dict(types.most_common()))
    print(f"train={counts['train']} val={counts['val']} -> {out}")


if __name__ == "__main__":
    main()
