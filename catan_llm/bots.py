"""Registry of scripted catanatron bots."""

from catanatron import Color, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import CONTENDER_WEIGHTS, ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

def _rollout(color, scenarios):
    from catan_llm.rollout_player import RolloutPlayer

    return RolloutPlayer(color, scenarios=scenarios)


def _tagged(name, factory):
    def make(color):
        player = factory(color)
        player.bot_name = name
        return player
    return make


_FACTORIES = {
    "random": RandomPlayer,
    "weighted_random": WeightedRandomPlayer,
    "victory_point": VictoryPointPlayer,
    "value_function": ValueFunctionPlayer,
    "alpha_beta": AlphaBetaPlayer,
    "alpha_beta_d3": lambda color: AlphaBetaPlayer(color, depth=3),
    "alpha_beta_d4": lambda color: AlphaBetaPlayer(color, depth=4),
    "alpha_beta_prunned": lambda color: AlphaBetaPlayer(color, prunning=True),
    "alpha_beta_contender": lambda color: AlphaBetaPlayer(
        color, value_fn_builder_name="C", params=CONTENDER_WEIGHTS
    ),
    "value_function_contender": lambda color: ValueFunctionPlayer(
        color, "C", params=CONTENDER_WEIGHTS
    ),
    "rollout_8": lambda color: _rollout(color, 8),
    "rollout_32": lambda color: _rollout(color, 32),
    "greedy_playouts_100": lambda color: GreedyPlayoutsPlayer(color, num_playouts=100),
    "mcts_100": lambda color: MCTSPlayer(color, num_simulations=100),
    "mcts_500": lambda color: MCTSPlayer(color, num_simulations=500),
    "same_turn_alpha_beta": SameTurnAlphaBetaPlayer,
    "greedy_playouts": GreedyPlayoutsPlayer,
    "mcts": MCTSPlayer,
}
BOTS = {name: _tagged(name, factory) for name, factory in _FACTORIES.items()}
COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
