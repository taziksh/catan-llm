"""Replay fidelity check against a game logged the way the env logs them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from teacher_agreement import replay_game

from catan_llm.bots import BOTS


def test_replay_completes_on_env_style_game(record_env_style_game):
    results = list(
        replay_game(record_env_style_game(42), BOTS["value_function"])
    )
    total = len(results)
    agree = sum(
        replayed.decision.legal_actions[replayed.decision.chosen_action]
        == teacher_action
        for replayed, teacher_action in results
    )
    assert total > 0
    assert 0 <= agree <= total
    assert all(
        replayed.game_record.game_id == replayed.decision.game_id
        and replayed.game.state.current_color().value
        == replayed.decision.actor.value
        and replayed.game.random is replayed.game.state.random
        for replayed, _ in results
    )
    assert len({id(replayed.game.random) for replayed, _ in results}) == total
