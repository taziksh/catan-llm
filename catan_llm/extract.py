"""Extracts trajectory records from catanatron games.

TrajectoryAccumulator hooks into Game.play() and writes one JSONL file per
game: a GameRecord followed by its DecisionRecords.
"""

import re
from pathlib import Path
from typing import Any

from catanatron.game import Game, GameAccumulator
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS, CatanMap
from catanatron.state_functions import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_largest_army,
    get_longest_road_color,
    get_longest_road_length,
    get_played_dev_cards,
    get_player_freqdeck,
    get_visible_victory_points,
)

from catan_llm.schema import (
    Action,
    ActionType,
    Bank,
    Board,
    Building,
    DecisionRecord,
    DevCard,
    GameConfig,
    GameRecord,
    Layout,
    Outcome,
    Phase,
    Player,
    PlayerState,
    Port,
    Resource,
    Tile,
)

_NO_PAYLOAD = {
    ActionType.ROLL,
    ActionType.END_TURN,
    ActionType.BUY_DEVELOPMENT_CARD,
    ActionType.PLAY_KNIGHT_CARD,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.CANCEL_TRADE,
}


def _bot_name(player) -> str:
    """Snake-cased class name minus the Player suffix, e.g. 'alpha_beta'."""
    name = type(player).__name__.removesuffix("Player")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def deterministic_game_id(game: Game) -> str:
    """Stable game id for reruns."""
    bots = "-".join(_bot_name(p) for p in game.state.players)
    return f"{bots}_s{game.seed}"


def _to_action(action, catan_map: CatanMap) -> Action:
    """Converts a catanatron Action to a schema [action_type, payload] pair."""
    action_type = ActionType(action.action_type.value)
    value = action.value
    payload: Any
    if action_type in _NO_PAYLOAD:
        payload = None
    elif action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = value[0], value[1]
        tile_id = catan_map.land_tiles[coordinate].id
        payload = [tile_id, victim.value if victim is not None else None]
    elif action_type == ActionType.CONFIRM_TRADE:
        payload = list(value[:10]) + [value[10].value]
    elif isinstance(value, tuple):
        payload = list(value)
    else:
        payload = value
    return (action_type, payload)


def decision_record(game: Game, action, i: int) -> DecisionRecord:
    """Builds a DecisionRecord from a game right before `action` is applied."""
    state = game.state
    catan_map = state.board.map

    # Roads are stored in both directions; keep u < v.
    roads = []
    for (u, v), color in state.board.roads.items():
        if u < v:
            assert state.board.roads[(v, u)] == color
            roads.append((u, v, Player(color.value)))

    board = Board(
        buildings=[
            (node, Player(color.value), Building(kind))
            for node, (color, kind) in sorted(state.board.buildings.items())
        ],
        roads=sorted(roads),
        robber=catan_map.land_tiles[state.board.robber_coordinate].id,
    )

    largest_army_color, _ = get_largest_army(state)
    players = {}
    for color in state.colors:
        players[Player(color.value)] = PlayerState(
            hand={
                Resource(r): n
                for r, n in zip(RESOURCES, get_player_freqdeck(state, color))
            },
            devs_played={
                DevCard(card): get_played_dev_cards(state, color, card)
                for card in DEVELOPMENT_CARDS
            },
            devs_in_hand={
                DevCard(card): get_dev_cards_in_hand(state, color, card)
                for card in DEVELOPMENT_CARDS
            },
            vps_public=get_visible_victory_points(state, color),
            vps_actual=get_actual_victory_points(state, color),
            road_len=get_longest_road_length(state, color),
            has_longest_road=get_longest_road_color(state) == color,
            has_largest_army=largest_army_color == color,
        )

    legal_actions = [_to_action(a, catan_map) for a in game.playable_actions]
    try:
        chosen_action = game.playable_actions.index(action)
    except ValueError:
        # OFFER_TRADE is legal during PLAY_TURN without being enumerated
        # in playable_actions (see catanatron.game.is_valid_action).
        if action.action_type.value != "OFFER_TRADE":
            raise
        legal_actions.append(_to_action(action, catan_map))
        chosen_action = len(legal_actions) - 1

    return DecisionRecord(
        game_id=game.id,
        i=i,
        turn=state.num_turns,
        phase=Phase(state.current_prompt.value),
        actor=Player(state.current_color().value),
        board=board,
        players=players,
        bank=Bank(
            resources={
                Resource(r): n for r, n in zip(RESOURCES, state.resource_freqdeck)
            },
            dev_cards_left=len(state.development_listdeck),
        ),
        legal_actions=legal_actions,
        chosen_action=chosen_action,
    )


def game_record(game: Game, trading: bool = False) -> GameRecord:
    """Builds a GameRecord from a finished game."""
    state = game.state
    catan_map = state.board.map

    tiles = [
        Tile(id=t.id, resource=t.resource, number=t.number)
        for t in sorted(catan_map.land_tiles.values(), key=lambda t: t.id)
    ]
    ports = []
    for port in sorted(catan_map.ports_by_id.values(), key=lambda p: p.id):
        a_ref, b_ref = PORT_DIRECTION_TO_NODEREFS[port.direction]
        nodes = sorted((port.nodes[a_ref], port.nodes[b_ref]))
        ports.append(Port(resource=port.resource, nodes=tuple(nodes)))

    winner = game.winning_color()
    return GameRecord(
        game_id=game.id,
        seed=game.seed,
        config=GameConfig(vps_to_win=game.vps_to_win, trading=trading),
        seats={Player(p.color.value): _bot_name(p) for p in state.players},
        layout=Layout(tiles=tiles, ports=ports),
        outcome=Outcome(
            winner=Player(winner.value) if winner is not None else None,
            final_vps={
                Player(c.value): get_actual_victory_points(state, c)
                for c in state.colors
            },
            turns=state.num_turns,
            truncated=winner is None,
        ),
    )


class TrajectoryAccumulator(GameAccumulator):
    """Collects records during a game and writes <out_dir>/<game_id>.jsonl."""

    def __init__(self, out_dir, trading: bool = False):
        self.out_dir = Path(out_dir)
        self.trading = trading
        self.decisions = []
        self.path = None

    def before(self, game):
        self.decisions = []

    def step(self, game, action):
        self._backfill_result(game.state)
        self.decisions.append(decision_record(game, action, len(self.decisions)))

    def _backfill_result(self, state):
        """Copies the previous action's outcome into its decision record."""
        if not self.decisions:
            return
        record = state.action_records[len(self.decisions) - 1]
        result = record.result
        self.decisions[-1].result = (
            list(result) if isinstance(result, tuple) else result
        )

    def after(self, game):
        self._backfill_result(game.state)
        record = game_record(game, trading=self.trading)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{game.id}.jsonl"
        with open(self.path, "w") as f:
            f.write(record.model_dump_json() + "\n")
            for decision in self.decisions:
                f.write(decision.model_dump_json() + "\n")
