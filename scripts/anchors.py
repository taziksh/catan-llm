"""Anchor win rates for the eval configuration.

Plays a reference bot in the LLM's seat (COLORS[0]) against three
value_function bots on the eval seeds, one game per seed. The resulting win
rates anchor the expert/peer/hopeless levels that LLM scores are read against.
"""

import argparse
import json
import time
from pathlib import Path

from catanatron import Game
from catanatron.state_functions import get_actual_victory_points

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinism import require_fixed_hashseed
from catan_llm.schema import ENGINE_VERSION

OPPONENT = "value_function"


def play(anchor, seed):
    players = [BOTS[anchor](COLORS[0])] + [
        BOTS[OPPONENT](color) for color in COLORS[1:]
    ]
    game = Game(players, seed=seed)
    winner = game.play()
    vps = {c: get_actual_victory_points(game.state, c) for c in game.state.colors}
    own = vps[COLORS[0]]
    margin = (own - max(v for c, v in vps.items() if c != COLORS[0])) / 10
    return {
        "seed": seed,
        "win": winner == COLORS[0],
        "vp_margin": margin,
        "turns": game.state.num_turns,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", default="victory_point,value_function,alpha_beta")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--out", default="data/benchmarks")
    args = parser.parse_args()

    results = {}
    for anchor in args.anchors.split(","):
        start = time.time()
        games = [play(anchor, seed) for seed in range(args.games)]
        wins = sum(g["win"] for g in games)
        results[anchor] = {
            "games": args.games,
            "wins": wins,
            "win_rate": wins / args.games,
            "mean_vp_margin": sum(g["vp_margin"] for g in games) / args.games,
            "avg_turns": sum(g["turns"] for g in games) / args.games,
            "per_seed": games,
        }
        print(
            f"{anchor:>16}: {wins}/{args.games} ({wins / args.games:.1%}) "
            f"in {time.time() - start:.0f}s",
            flush=True,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"anchors_vs_{OPPONENT}_n{args.games}.json"
    out_path.write_text(
        json.dumps(
            {
                "engine_version": ENGINE_VERSION,
                "opponent": OPPONENT,
                "seat": 0,
                "seeds": list(range(args.games)),
                "anchors": results,
            },
            indent=1,
        )
    )
    print(f"-> {out_path}")


if __name__ == "__main__":
    require_fixed_hashseed()
    main()
