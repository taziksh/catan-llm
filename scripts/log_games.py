"""Plays bot games and writes one trajectory JSONL file per game."""

import argparse
from multiprocessing import Pool

from catanatron import Game

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinism import EVAL_SEED_LIMIT, require_fixed_hashseed
from catan_llm.extract import TrajectoryAccumulator, deterministic_game_id


def play_game(names, seed, out):
    players = [BOTS[name](color) for name, color in zip(names, COLORS)]
    game = Game(players, seed=seed)
    game.id = deterministic_game_id(game)
    accumulator = TrajectoryAccumulator(out)
    winner = game.play(accumulators=[accumulator])
    print(
        f"{game.id}: winner={winner} "
        f"turns={game.state.num_turns} -> {accumulator.path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", default="alpha_beta,random")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=EVAL_SEED_LIMIT,
        help="first seed; seeds below %(default)s are reserved for the "
        "catan-v1 eval taskset",
    )
    parser.add_argument("--out", default="data/games")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel game processes; per-game rng makes results "
        "identical to a sequential run",
    )
    args = parser.parse_args()

    names = args.players.split(",")
    jobs = [(names, args.seed + n, args.out) for n in range(args.games)]
    if args.workers == 1:
        for job in jobs:
            play_game(*job)
    else:
        with Pool(args.workers) as pool:
            pool.starmap(play_game, jobs, chunksize=1)


if __name__ == "__main__":
    require_fixed_hashseed()
    main()
