"""Replay fidelity check against a game logged the way the env logs them."""

import json
import sys
from pathlib import Path

from catanatron import Game

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from teacher_agreement import replay_game

from catan_llm.bots import BOTS, COLORS
from catan_llm.extract import TrajectoryAccumulator


def test_replay_completes_on_env_style_game(tmp_path):
    players = [BOTS["value_function"](color) for color in COLORS]
    game = Game(players, seed=42)
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
    accumulator.path.write_text("\n".join([json.dumps(header)] + lines[1:]) + "\n")

    agree, total = replay_game(accumulator.path, BOTS["value_function"])
    assert total > 0
    assert 0 <= agree <= total
