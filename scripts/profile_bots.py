"""Profiles per-decision latency of bots against a fixed opponent."""

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median, quantiles

from catanatron import Game

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinism import require_fixed_hashseed


def profile(bot, opponent, games, seed):
    latencies, wins, turns = [], 0, []
    for g in range(games):
        order = [bot, opponent] if g % 2 == 0 else [opponent, bot]
        players = [BOTS[name](color) for name, color in zip(order, COLORS)]
        target = COLORS[order.index(bot)]

        def decide_fn(player, game, playable_actions):
            start = time.perf_counter()
            action = player.decide(game, playable_actions)
            if player.color == target:
                latencies.append(
                    (time.perf_counter() - start, len(playable_actions))
                )
            return action

        game = Game(players, seed=seed + g)
        winner = game.play(decide_fn=decide_fn)
        wins += winner == target
        turns.append(game.state.num_turns)

    real = [lat for lat, choices in latencies if choices > 1]
    return {
        "bot": bot,
        "games": games,
        "wins": wins,
        "avg_turns": mean(turns),
        "decisions": len(latencies),
        "real_decisions": len(real),
        "latency_ms": {
            "mean": 1000 * mean(real),
            "median": 1000 * median(real),
            "p95": 1000 * quantiles(real, n=20)[-1],
            "max": 1000 * max(real),
        },
        "samples_ms": [[1000 * lat, choices] for lat, choices in latencies],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bots", default="value_function,alpha_beta,same_turn_alpha_beta")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/benchmarks/profile.json")
    args = parser.parse_args()

    results = []
    for bot in args.bots.split(","):
        result = profile(bot, args.opponent, args.games, args.seed)
        results.append(result)
        lat = result["latency_ms"]
        print(
            f"{bot:22s} {result['real_decisions']:5d} real decisions | "
            f"median {lat['median']:7.2f}ms  p95 {lat['p95']:8.2f}ms  "
            f"max {lat['max']:8.0f}ms | {result['wins']}/{args.games} wins",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"opponent": args.opponent, "results": results}))
    print(f"-> {out}")


if __name__ == "__main__":
    require_fixed_hashseed()
    main()
