"""Serializer and parser tests."""

from catan_llm.parse import parse_answer
from catan_llm.schema import ActionType
from catan_llm.serialize import decision_to_prompt


def test_prompt_structure(trajectory):
    game_rec, decisions = trajectory
    rec = next(r for r in decisions if len(r.legal_actions) > 1)
    prompt = decision_to_prompt(game_rec, rec)
    for i in range(len(rec.legal_actions)):
        assert f"\n{i}. " in prompt
    assert 'Reply with "answer: <option number>".' in prompt
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
    assert f"build settlement at node {node} (adjacent: " in prompt


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
        options = [line for line in prompt.splitlines() if line[:1].isdigit()]
        assert len(options) == len(rec.legal_actions)
        assert len(set(options)) == len(options)


def test_index_round_trip(trajectory):
    _, decisions = trajectory
    for rec in decisions:
        n = len(rec.legal_actions)
        for i in range(n):
            assert parse_answer(f"answer: {i}", n) == i
