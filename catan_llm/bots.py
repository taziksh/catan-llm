"""Registry of scripted catanatron bots."""

from catanatron import Color, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

BOTS = {
    "random": RandomPlayer,
    "weighted_random": WeightedRandomPlayer,
    "victory_point": VictoryPointPlayer,
    "value_function": ValueFunctionPlayer,
    "alpha_beta": AlphaBetaPlayer,
    "same_turn_alpha_beta": SameTurnAlphaBetaPlayer,
    "greedy_playouts": GreedyPlayoutsPlayer,
    "mcts": MCTSPlayer,
}
COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
