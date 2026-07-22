"""Sanity check for the catanatron install: one full game, AlphaBeta should beat Random."""

import time
from importlib.metadata import version

from catanatron import Color, Game, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer


def main():
    players = [
        RandomPlayer(Color.RED),
        AlphaBetaPlayer(Color.BLUE),
    ]
    game = Game(players, seed=42)

    start = time.time()
    winner = game.play()
    elapsed = time.time() - start

    print(f"catanatron version: {version('catanatron')}")
    print(f"winner: {winner}, turns: {game.state.num_turns}, elapsed: {elapsed:.1f}s")
    assert winner == Color.BLUE, f"expected AlphaBeta (BLUE) to win, got {winner}"
    print("OK: AlphaBeta won")


if __name__ == "__main__":
    main()
