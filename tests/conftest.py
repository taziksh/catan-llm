"""Shared fixtures: one logged game reused across test modules."""

import pytest
from catanatron import Color, Game, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer

from catan_llm.extract import TrajectoryAccumulator, deterministic_game_id
from catan_llm.schema import DecisionRecord, GameRecord

SEED = 7


def _players():
    return [RandomPlayer(Color.RED), AlphaBetaPlayer(Color.BLUE)]


def log_game(out_dir):
    game = Game(_players(), seed=SEED)
    game.id = deterministic_game_id(game)
    accumulator = TrajectoryAccumulator(out_dir)
    game.play(accumulators=[accumulator])
    return accumulator.path


@pytest.fixture(scope="session")
def logged_path(tmp_path_factory):
    return log_game(tmp_path_factory.mktemp("log"))


@pytest.fixture(scope="session")
def trajectory(logged_path):
    lines = logged_path.read_text().splitlines()
    return GameRecord.model_validate_json(lines[0]), [
        DecisionRecord.model_validate_json(line) for line in lines[1:]
    ]
