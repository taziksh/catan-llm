"""Player that simulates every legal move to completion and plays the best one."""

import random

from catanatron.models.player import Player as EnginePlayer

from catan_llm.determinize import determinize
from catan_llm.extract import to_action
from catan_llm.schema import Player
from catan_llm.serialize import move_id
from catan_llm.simulation import playout_reward, rollout_action

HERO_POLICY = "alpha_beta"
OPPONENT_POLICY = "value_function"


class RolloutPlayer(EnginePlayer):
    """For each playable action, play N playouts with scripted bots."""

    def __init__(self, color, scenarios=8, seed=0):
        super().__init__(color)
        self.scenarios = scenarios
        self.seed = seed
        self.decisions = 0

    def decide(self, game, playable_actions):
        self.decisions += 1
        if len(playable_actions) < 2:
            return playable_actions[0]
        hero = Player(self.color.value)
        colors = list(game.state.colors)
        policies = {
            Player(color.value): HERO_POLICY if color == self.color else OPPONENT_POLICY
            for color in colors
        }
        catan_map = game.state.board.map
        moves = {
            move_id(*to_action(action, catan_map)): action
            for action in playable_actions
        }

        rng = random.Random(f"{self.seed}:{self.decisions}")
        totals = {mid: 0.0 for mid in moves}
        for _ in range(self.scenarios):
            scenario = rng.getrandbits(32)
            world = determinize(game, colors.index(self.color), scenario)
            future = random.Random(f"{scenario}:continuation").getrandbits(63)
            world_map = world.state.board.map
            indices = {
                move_id(*to_action(action, world_map)): index
                for index, action in enumerate(world.playable_actions)
            }
            for mid in moves:
                outcome = rollout_action(world, indices[mid], policies, seed=future)
                totals[mid] += playout_reward(outcome, hero)
        best = max(totals.items(), key=lambda pair: pair[1])[0]
        return moves[best]
