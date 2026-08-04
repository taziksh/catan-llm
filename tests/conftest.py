"""Shared fixtures: one logged game reused across test modules."""

import json

import pytest
from catanatron import Color, Game, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer

from catan_llm.bots import BOTS, COLORS
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


@pytest.fixture
def record_env_style_game(tmp_path):
    """Returns a factory that logs a bot game with one seat marked llm."""

    def record(seed):
        players = [BOTS["value_function"](color) for color in COLORS]
        game = Game(players, seed=seed)
        accumulator = TrajectoryAccumulator(tmp_path)
        accumulator.before(game)
        llm_color = COLORS[0]
        while game.winning_color() is None and game.state.num_turns < 300:
            current = game.state.current_player()
            if current.color == llm_color:
                action = game.playable_actions[0]
            else:
                action = current.decide(game, game.playable_actions)
            accumulator.step(game, action)
            game.execute(action)
        accumulator.after(game)

        lines = accumulator.path.read_text().splitlines()
        header = json.loads(lines[0])
        header["seats"][llm_color.value] = "llm"
        accumulator.path.write_text(
            "\n".join([json.dumps(header)] + lines[1:]) + "\n"
        )
        return accumulator.path

    return record
