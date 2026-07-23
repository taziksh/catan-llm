"""Replay verifier: re-executes a logged game and checks every record."""

from conftest import log_game

from catan_llm.schema import ActionType, DevCard, Phase, Resource

# Phases where the legal action types are fully determined by the phase.
PHASE_ACTIONS = {
    Phase.BUILD_INITIAL_SETTLEMENT: {ActionType.BUILD_SETTLEMENT},
    Phase.BUILD_INITIAL_ROAD: {ActionType.BUILD_ROAD},
    Phase.DISCARD: {ActionType.DISCARD_RESOURCE},
    Phase.MOVE_ROBBER: {ActionType.MOVE_ROBBER},
}


def test_replay_matches_log(logged_path, tmp_path_factory):
    replay_path = log_game(tmp_path_factory.mktemp("replay"))
    logged = logged_path.read_text().splitlines()
    replayed = replay_path.read_text().splitlines()
    assert len(logged) == len(replayed)
    for i, (a, b) in enumerate(zip(logged, replayed)):
        assert a == b, f"line {i} differs"


def test_invariants(trajectory):
    game_rec, decisions = trajectory
    for i, rec in enumerate(decisions):
        assert rec.game_id == game_rec.game_id
        assert rec.i == i
        assert 0 <= rec.chosen_action < len(rec.legal_actions)
        for player_state in rec.players.values():
            assert all(n >= 0 for n in player_state.hand.values())

        action_types = {action_type for action_type, _ in rec.legal_actions}
        if rec.phase in PHASE_ACTIONS:
            assert action_types <= PHASE_ACTIONS[rec.phase]
        if ActionType.DISCARD_RESOURCE in action_types:
            assert rec.phase == Phase.DISCARD
        if ActionType.MOVE_ROBBER in action_types:
            assert rec.phase == Phase.MOVE_ROBBER

    turns = [rec.turn for rec in decisions]
    assert turns == sorted(turns)
    assert game_rec.outcome.turns >= turns[-1]


def test_results(trajectory):
    _, decisions = trajectory
    rolls = 0
    for rec in decisions:
        chosen_type, _ = rec.legal_actions[rec.chosen_action]
        if chosen_type == ActionType.ROLL:
            rolls += 1
            die_a, die_b = rec.result
            assert 1 <= die_a <= 6 and 1 <= die_b <= 6
        elif chosen_type == ActionType.BUY_DEVELOPMENT_CARD:
            assert rec.result in {card.value for card in DevCard}
        elif chosen_type == ActionType.MOVE_ROBBER:
            assert rec.result is None or rec.result in {r.value for r in Resource}
        elif chosen_type == ActionType.DISCARD_RESOURCE:
            assert rec.result == rec.legal_actions[rec.chosen_action][1]
        else:
            assert rec.result is None
    assert rolls > 0
