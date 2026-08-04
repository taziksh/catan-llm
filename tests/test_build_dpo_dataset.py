import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_dpo_dataset import make_pair

from catan_llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from catan_llm.serialize import decision_to_prompt, move_id


def _score(game, decision, best_index):
    chosen_id = move_id(*decision.legal_actions[decision.chosen_action])
    best_id = move_id(*decision.legal_actions[best_index])
    return {
        "schema_version": 1,
        "game_id": game.game_id,
        "seed": game.seed,
        "decision": decision.i,
        "move_type": decision.legal_actions[decision.chosen_action][0].value,
        "chosen_id": chosen_id,
        "chosen_value": 1.0,
        "best_id": best_id,
        "best_value": 2.0,
        "strict_preference": True,
        "any_timeout": False,
    }


def test_pair_direction_and_prompt_identity(trajectory):
    game, decisions = trajectory
    decision = next(d for d in decisions if len(d.legal_actions) > 1)
    best_index = next(
        i for i in range(len(decision.legal_actions))
        if i != decision.chosen_action
    )
    score = _score(game, decision, best_index)

    pair = make_pair(game, decision, score)

    assert pair["prompt_version"] == PROMPT_VERSION
    assert pair["prompt"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": decision_to_prompt(game, decision)},
    ]
    assert pair["chosen"] == [
        {"role": "assistant", "content": f"answer: {score['best_id']}"}
    ]
    assert pair["rejected"] == [
        {"role": "assistant", "content": f"answer: {score['chosen_id']}"}
    ]
    assert pair["teacher_value_gap"] == 1.0


@pytest.mark.parametrize(
    ("strict_preference", "any_timeout"),
    [(False, False), (True, True)],
)
def test_non_strict_or_timed_out_rows_are_excluded(
    trajectory, strict_preference, any_timeout
):
    game, decisions = trajectory
    decision = next(d for d in decisions if len(d.legal_actions) > 1)
    best_index = next(
        i for i in range(len(decision.legal_actions))
        if i != decision.chosen_action
    )
    score = _score(game, decision, best_index)
    score["strict_preference"] = strict_preference
    score["any_timeout"] = any_timeout
    if not strict_preference:
        score["best_value"] = score["chosen_value"]

    assert make_pair(game, decision, score) is None


def test_rejects_reversed_or_misaligned_evidence(trajectory):
    game, decisions = trajectory
    decision = next(d for d in decisions if len(d.legal_actions) > 1)
    best_index = next(
        i for i in range(len(decision.legal_actions))
        if i != decision.chosen_action
    )
    score = _score(game, decision, best_index)
    score["chosen_id"] = score["best_id"]

    with pytest.raises(ValueError, match="logged move mismatch"):
        make_pair(game, decision, score)
