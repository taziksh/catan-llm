"""Plays bot games and writes one trajectory JSONL file per game."""

import argparse
import os
import sys

from catanatron import Game

from catan_llm.bots import BOTS, COLORS
from catan_llm.extract import TrajectoryAccumulator, deterministic_game_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", default="alpha_beta,random")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/games")
    args = parser.parse_args()

    names = args.players.split(",")
    for n in range(args.games):
        players = [BOTS[name](color) for name, color in zip(names, COLORS)]
        game = Game(players, seed=args.seed + n)
        game.id = deterministic_game_id(game)
        accumulator = TrajectoryAccumulator(args.out)
        winner = game.play(accumulators=[accumulator])
        print(
            f"{game.id}: winner={winner} "
            f"turns={game.state.num_turns} -> {accumulator.path}"
        )


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        # Hash randomization must be off before interpreter start for
        # reproducible games, so re-exec with it set.
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    main()
