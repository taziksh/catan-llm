"""Player that scores each legal move with the value model and plays the best one."""

import random

import lightgbm as lgb
import numpy as np
from catanatron.features import create_sample_vector
from catanatron.models.player import Player as EnginePlayer

from catan_llm.determinize import determinize
from catan_llm.extract import to_action
from catan_llm.serialize import move_id
from catan_llm.simulation import detached_game_copy

MODEL_PATH = "experiments/tier1_value_exit/value_model/model.txt"


def afterstate_features(world, move_index, future_seed, hero_index, turn):
    """Features of the position reached by playing one move."""
    afterstate = detached_game_copy(world)
    rng = random.Random(future_seed)
    afterstate.random = rng
    afterstate.state.random = rng
    afterstate.execute(afterstate.playable_actions[move_index])
    hero = afterstate.state.colors[hero_index]
    row = create_sample_vector(afterstate, hero)
    row += [
        float(hero_index),
        float(turn),
        float(afterstate.state.current_color() == hero),
    ]
    return row


class ValueModelPlayer(EnginePlayer):
    """Scores each action by the predicted P(win) of the position it leads to,
    averaged over `worlds` samples of the hidden information."""

    def __init__(self, color, worlds, seed=0, model_path=MODEL_PATH):
        super().__init__(color)
        self.worlds = worlds
        self.seed = seed
        self.model = lgb.Booster(model_file=model_path)
        self.decisions = 0

    def decide(self, game, playable_actions):
        self.decisions += 1
        if len(playable_actions) < 2:
            return playable_actions[0]
        hero_index = list(game.state.colors).index(self.color)
        catan_map = game.state.board.map
        moves = {
            move_id(*to_action(action, catan_map)): action
            for action in playable_actions
        }
        rng = random.Random(f"{self.seed}:{self.decisions}")
        rows = []
        for _ in range(self.worlds):
            scenario = rng.getrandbits(32)
            world = determinize(game, hero_index, scenario)
            future = random.Random(f"{scenario}:continuation").getrandbits(63)
            world_map = world.state.board.map
            indices = {
                move_id(*to_action(action, world_map)): index
                for index, action in enumerate(world.playable_actions)
            }
            for mid in moves:
                rows.append(
                    afterstate_features(
                        world, indices[mid], future, hero_index, game.state.num_turns
                    )
                )
        values = self.model.predict(np.asarray(rows, dtype=np.float32))
        means = values.reshape(self.worlds, len(moves)).mean(axis=0)
        best = list(moves)[int(means.argmax())]
        return moves[best]
