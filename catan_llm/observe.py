"""Actor's view of a decision, derived from the omniscient log records.

Never logged: observations are recomputed from GameRecord + DecisionRecord.
OpponentView structurally cannot hold hidden information.
"""

from pydantic import BaseModel

from catan_llm.schema import (
    Action,
    Bank,
    Board,
    DecisionRecord,
    DevCard,
    GameRecord,
    Layout,
    Phase,
    Player,
    PlayerState,
)


class OpponentView(BaseModel):
    color: Player
    card_count: int  # hand composition is hidden
    dev_card_count: int  # unplayed dev cards are hidden
    devs_played: dict[DevCard, int]
    vps_public: int
    road_len: int
    has_longest_road: bool
    has_largest_army: bool


class Observation(BaseModel):
    actor: Player
    turn: int
    phase: Phase
    turn_order: list[Player]  # seating order, cyclic
    vps_to_win: int
    trading: bool
    layout: Layout
    board: Board
    you: PlayerState  # own hand and actual VPs are known to the actor
    opponents: list[OpponentView]  # seating order
    bank: Bank
    legal_actions: list[Action]


def opponent_view(color: Player, ps: PlayerState) -> OpponentView:
    return OpponentView(
        color=color,
        card_count=sum(ps.hand.values()),
        dev_card_count=sum(ps.devs_in_hand.values()),
        devs_played=ps.devs_played,
        vps_public=ps.vps_public,
        road_len=ps.road_len,
        has_longest_road=ps.has_longest_road,
        has_largest_army=ps.has_largest_army,
    )


def observe(game: GameRecord, decision: DecisionRecord) -> Observation:
    opponents = [
        opponent_view(color, ps)
        for color, ps in decision.players.items()
        if color != decision.actor
    ]
    return Observation(
        actor=decision.actor,
        turn=decision.turn,
        phase=decision.phase,
        turn_order=list(decision.players),
        vps_to_win=game.config.vps_to_win,
        trading=game.config.trading,
        layout=game.layout,
        board=decision.board,
        you=decision.players[decision.actor],
        opponents=opponents,
        bank=decision.bank,
        legal_actions=decision.legal_actions,
    )
