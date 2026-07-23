"""Trajectory schema for logged Catan games.

A trajectory file is JSONL: one GameRecord for the game, followed by one
DecisionRecord per choice point.
"""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel

SCHEMA_VERSION = 1

# vendor/catanatron @ d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816, installed from source.
ENGINE_VERSION = "catanatron-3.3.0+d3f4ad0"


class Player(str, Enum):
    RED = "RED"
    BLUE = "BLUE"
    WHITE = "WHITE"
    ORANGE = "ORANGE"


class Resource(str, Enum):
    WOOD = "WOOD"
    BRICK = "BRICK"
    SHEEP = "SHEEP"
    WHEAT = "WHEAT"
    ORE = "ORE"


class Building(str, Enum):
    SETTLEMENT = "SETTLEMENT"
    CITY = "CITY"


class Phase(str, Enum):
    BUILD_INITIAL_SETTLEMENT = "BUILD_INITIAL_SETTLEMENT"
    BUILD_INITIAL_ROAD = "BUILD_INITIAL_ROAD"
    PLAY_TURN = "PLAY_TURN"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"
    DECIDE_TRADE = "DECIDE_TRADE"
    DECIDE_ACCEPTEES = "DECIDE_ACCEPTEES"


class DevCard(str, Enum):
    KNIGHT = "KNIGHT"
    VICTORY_POINT = "VICTORY_POINT"
    YEAR_OF_PLENTY = "YEAR_OF_PLENTY"
    MONOPOLY = "MONOPOLY"
    ROAD_BUILDING = "ROAD_BUILDING"


class ActionType(str, Enum):
    # payload: null
    ROLL = "ROLL"
    END_TURN = "END_TURN"
    BUY_DEVELOPMENT_CARD = "BUY_DEVELOPMENT_CARD"
    PLAY_KNIGHT_CARD = "PLAY_KNIGHT_CARD"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"
    CANCEL_TRADE = "CANCEL_TRADE"
    # payload: node: int
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"
    BUILD_CITY = "BUILD_CITY"
    # payload: [u: int, v: int]
    BUILD_ROAD = "BUILD_ROAD"
    # payload: [tile_id: int, victim: Player | null]
    MOVE_ROBBER = "MOVE_ROBBER"
    # payload: Resource
    DISCARD_RESOURCE = "DISCARD_RESOURCE"
    PLAY_MONOPOLY = "PLAY_MONOPOLY"
    # payload: [Resource] or [Resource, Resource]
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"
    # payload: 5 slots [give | null, give | null, give | null, give | null, receive]
    MARITIME_TRADE = "MARITIME_TRADE"
    # payload: 10 ints [5 give counts, 5 receive counts], Resource order
    OFFER_TRADE = "OFFER_TRADE"
    ACCEPT_TRADE = "ACCEPT_TRADE"
    REJECT_TRADE = "REJECT_TRADE"
    # payload: 11 slots [5 give counts, 5 receive counts, accepter: Player]
    CONFIRM_TRADE = "CONFIRM_TRADE"


# [action_type, payload] — payload shape depends on action_type, see ActionType comments.
Action = tuple[ActionType, Any]


class Tile(BaseModel):
    id: int
    resource: Optional[Resource]  # null = desert
    number: Optional[int]


class Port(BaseModel):
    resource: Optional[Resource]  # null = 3:1
    nodes: tuple[int, int]


class Layout(BaseModel):
    tiles: list[Tile]
    ports: list[Port]


class GameConfig(BaseModel):
    vps_to_win: int
    trading: bool
    discard_limit: int = 7


class Outcome(BaseModel):
    winner: Optional[Player]  # null = truncated with no winner
    final_vps: dict[Player, int]
    turns: int
    truncated: bool


class GameRecord(BaseModel):
    type: Literal["game"] = "game"
    schema_version: int = SCHEMA_VERSION
    game_id: str
    seed: Optional[int]
    engine_version: str = ENGINE_VERSION
    config: GameConfig
    seats: dict[Player, str]  # e.g. {RED: "alphabeta", BLUE: "value_function"}
    layout: Layout
    outcome: Outcome


class Board(BaseModel):
    buildings: list[tuple[int, Player, Building]]  # [node, color, type]
    roads: list[tuple[int, int, Player]]  # [u, v, color]
    robber: int  # tile_id


class PlayerState(BaseModel):
    hand: dict[Resource, int]
    devs_played: dict[DevCard, int]
    devs_in_hand: dict[DevCard, int]
    vps_public: int
    vps_actual: int
    road_len: int
    has_longest_road: bool
    has_largest_army: bool


class Bank(BaseModel):
    resources: dict[Resource, int]
    dev_cards_left: int


class DecisionRecord(BaseModel):
    type: Literal["decision"] = "decision"
    game_id: str
    i: int  # decision index within the game
    turn: int
    phase: Phase
    actor: Player
    board: Board
    players: dict[Player, PlayerState]
    bank: Bank
    legal_actions: list[Action]
    chosen_action: int  # index into legal_actions
    result: Any = None  # [die: int, die: int] | DevCard | Resource | null
