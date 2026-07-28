"""Serializer and parser tests."""

from itertools import takewhile

from hypothesis import assume, given
from hypothesis import strategies as st

from catan_llm.parse import parse_answer, parse_move
from catan_llm.schema import ActionType, Player, Resource
from catan_llm.serialize import decision_to_prompt, move_id

NULLARY = [
    ActionType.ROLL,
    ActionType.END_TURN,
    ActionType.BUY_DEVELOPMENT_CARD,
    ActionType.PLAY_KNIGHT_CARD,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.CANCEL_TRADE,
    ActionType.ACCEPT_TRADE,
    ActionType.REJECT_TRADE,
]
nodes = st.integers(0, 53)
resources = st.sampled_from([r.value for r in Resource])
players = st.sampled_from([p.value for p in Player])
counts = st.lists(st.integers(0, 5), min_size=10, max_size=10)


def options_of(prompt):
    body = prompt.split("YOUR OPTIONS\n", 1)[1]
    return list(takewhile(bool, body.splitlines()))


@st.composite
def maritime(draw):
    given_up = draw(st.integers(2, 4))
    return [draw(resources)] * given_up + [None] * (4 - given_up) + [draw(resources)]


def actions():
    """Generates an [action_type, payload] pair for every action type."""
    return st.one_of(
        st.tuples(st.sampled_from(NULLARY), st.none()),
        st.tuples(
            st.sampled_from([ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY]),
            nodes,
        ),
        st.tuples(
            st.just(ActionType.BUILD_ROAD), st.lists(nodes, min_size=2, max_size=2)
        ),
        st.tuples(
            st.just(ActionType.MOVE_ROBBER),
            st.tuples(st.integers(0, 18), st.none() | players).map(list),
        ),
        st.tuples(
            st.sampled_from([ActionType.DISCARD_RESOURCE, ActionType.PLAY_MONOPOLY]),
            resources,
        ),
        st.tuples(
            st.just(ActionType.PLAY_YEAR_OF_PLENTY),
            st.lists(resources, min_size=1, max_size=2),
        ),
        st.tuples(st.just(ActionType.MARITIME_TRADE), maritime()),
        st.tuples(st.just(ActionType.OFFER_TRADE), counts),
        st.tuples(
            st.just(ActionType.CONFIRM_TRADE),
            st.builds(lambda c, p: c + [p], counts, players),
        ),
    )


def test_prompt_structure(trajectory):
    game_rec, decisions = trajectory
    rec = next(r for r in decisions if len(r.legal_actions) > 1)
    prompt = decision_to_prompt(game_rec, rec)
    for action_type, payload in rec.legal_actions:
        assert f"\n{move_id(action_type, payload)}" in prompt
    assert 'Reply with "answer: <move id>".' in prompt
    assert "No trading between players" in prompt
    assert f"turn order: " in prompt and f"{rec.actor.value} (you)" in prompt


def test_info_set(trajectory):
    game_rec, decisions = trajectory
    for rec in decisions[:30]:
        prompt = decision_to_prompt(game_rec, rec)
        # Itemized hands appear exactly twice: the actor's and the bank's.
        assert prompt.count("wood:") == 2
        for color, ps in rec.players.items():
            if color != rec.actor:
                assert f"{color.value}: {sum(ps.hand.values())} card" in prompt


def test_settlement_annotation(trajectory):
    game_rec, decisions = trajectory
    rec = next(
        r
        for r in decisions
        if r.legal_actions[r.chosen_action][0] == ActionType.BUILD_SETTLEMENT
    )
    prompt = decision_to_prompt(game_rec, rec)
    node = rec.legal_actions[rec.chosen_action][1]
    assert f"settlement:{node} (adjacent: " in prompt


def test_parse_answer():
    assert parse_answer("answer: 3", 10) == 3
    assert parse_answer("Answer: 3", 10) == 3
    assert parse_answer("thinking...\nanswer: 2\nno wait\nanswer: 4", 10) == 4
    assert parse_answer("**answer: 12**", 14) == 12
    assert parse_answer("7", 10) == 7
    assert parse_answer("7.", 10) == 7
    assert parse_answer("answer: 12", 10) is None
    assert parse_answer("I pick option 3, probably", 10) is None
    assert parse_answer("", 10) is None


def test_all_decisions_render(trajectory):
    game_rec, decisions = trajectory
    for rec in decisions:
        prompt = decision_to_prompt(game_rec, rec)
        options = options_of(prompt)
        assert len(options) == len(rec.legal_actions)
        assert len(set(options)) == len(options)


def test_index_round_trip(trajectory):
    _, decisions = trajectory
    for rec in decisions:
        n = len(rec.legal_actions)
        for i in range(n):
            assert parse_answer(f"answer: {i}", n) == i


def test_move_round_trip(trajectory):
    _, decisions = trajectory
    for rec in decisions:
        ids = [move_id(t, p) for t, p in rec.legal_actions]
        assert len(set(ids)) == len(ids)
        for i, mid in enumerate(ids):
            assert parse_move(f"answer: {mid}", rec.legal_actions) == i


@given(action=actions())
def test_move_id_is_a_bare_token(action):
    mid = move_id(*action)
    assert mid and not any(c.isspace() for c in mid)


@given(action=actions())
def test_move_id_identifies_the_action(action):
    assert parse_move(f"answer: {move_id(*action)}", [action]) == 0


@given(pair=st.lists(actions(), min_size=2, max_size=2, unique_by=repr))
def test_distinct_actions_get_distinct_ids(pair):
    assert move_id(*pair[0]) != move_id(*pair[1])


@given(action=actions(), before=st.text(), after=st.sampled_from(["", ".", "**", "`", ",", '"']))
def test_parse_move_survives_surrounding_text(action, before, after):
    assume("answer" not in before.lower())
    reply = f"{before}\nanswer: {move_id(*action)}{after}"
    assert parse_move(reply, [action]) == 0


@given(action=actions())
def test_parse_move_rejects_illegal_moves(action):
    assume(move_id(*action) != "roll")
    assert parse_move(f"answer: {move_id(*action)}", [(ActionType.ROLL, None)]) is None
