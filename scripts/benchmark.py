"""Round-robin benchmark of scripted bots, one JSON results file per run."""

import argparse
import itertools
import json
import time
from pathlib import Path

from catanatron import Game

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinism import require_fixed_hashseed
from catan_llm.schema import ENGINE_VERSION

PAIR_SEED_STRIDE = 10_000


def play_pair(a, b, games, seed_base):
    wins = {a: 0, b: 0, None: 0}
    turns = []
    for g in range(games):
        order = [a, b] if g % 2 == 0 else [b, a]
        players = [BOTS[name](color) for name, color in zip(order, COLORS)]
        game = Game(players, seed=seed_base + g)
        winner_color = game.play()
        winner = order[COLORS.index(winner_color)] if winner_color else None
        wins[winner] += 1
        turns.append(game.state.num_turns)
    return {
        "bots": [a, b],
        "wins": {a: wins[a], b: wins[b], "truncated": wins[None]},
        "avg_turns": sum(turns) / len(turns),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bots",
        default="random,weighted_random,victory_point,value_function,alpha_beta,same_turn_alpha_beta",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/benchmarks")
    args = parser.parse_args()

    bots = args.bots.split(",")
    pairs = list(itertools.combinations(bots, 2))
    results = []
    for pair_index, (a, b) in enumerate(pairs):
        start = time.time()
        result = play_pair(a, b, args.games, args.seed + pair_index * PAIR_SEED_STRIDE)
        results.append(result)
        print(
            f"[{pair_index + 1}/{len(pairs)}] {a} {result['wins'][a]}"
            f" - {result['wins'][b]} {b} ({time.time() - start:.0f}s)",
            flush=True,
        )

    totals = {bot: {"wins": 0, "games": 0} for bot in bots}
    for result in results:
        a, b = result["bots"]
        for bot in (a, b):
            totals[bot]["wins"] += result["wins"][bot]
            totals[bot]["games"] += args.games

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"roundrobin_s{args.seed}_n{args.games}.json"
    out_path.write_text(
        json.dumps(
            {
                "engine_version": ENGINE_VERSION,
                "bots": bots,
                "games_per_pair": args.games,
                "seed": args.seed,
                "pairs": results,
                "totals": totals,
            },
            indent=2,
        )
    )
    print(f"-> {out_path}")

    for bot, t in sorted(totals.items(), key=lambda kv: -kv[1]["wins"]):
        print(f"{bot:22s} {t['wins']:4d}/{t['games']} ({100 * t['wins'] / t['games']:.0f}%)")


if __name__ == "__main__":
    require_fixed_hashseed()
    main()
