"""Compares best-move labels between two independently-seeded reward caches."""

import argparse
from collections import defaultdict
from pathlib import Path

from build_reward_cache import load_cache

BINS = (0.005, 0.01, 0.02, 0.05, 0.1)


def best_moves(path):
    """(game_id, decision) -> (best move, margin over runner-up)."""
    rewards = defaultdict(lambda: defaultdict(list))
    for (game_id, decision, move, _), row in load_cache(Path(path)).items():
        rewards[game_id, decision][move].append(row["reward"])
    best = {}
    for key, moves in rewards.items():
        means = sorted(((sum(r) / len(r), m) for m, r in moves.items()), reverse=True)
        margin = means[0][0] - means[1][0] if len(means) > 1 else float("inf")
        best[key] = (means[0][1], margin)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-a", required=True)
    parser.add_argument("--bank-b", required=True)
    args = parser.parse_args()

    a, b = best_moves(args.bank_a), best_moves(args.bank_b)
    by_margin = defaultdict(lambda: [0, 0])
    by_type = defaultdict(lambda: [0, 0])
    for key, (move, margin) in a.items():
        flipped = move != b[key][0]
        bin_label = next((f"<{edge}" for edge in BINS if margin < edge), f">={BINS[-1]}")
        for table, label in ((by_margin, bin_label), (by_type, move.split(":")[0])):
            table[label][0] += flipped
            table[label][1] += 1

    flips = sum(f for f, _ in by_margin.values())
    total = sum(n for _, n in by_margin.values())
    print(f"TOTAL: {flips}/{total} flips = {flips / total:.1%}\n")
    for name, table in (("margin", by_margin), ("move type", by_type)):
        print(f"by {name}:")
        for label, (f, n) in sorted(table.items()):
            print(f"  {label:>8}: {f}/{n} = {f / n:.1%}")
        print()


if __name__ == "__main__":
    main()
