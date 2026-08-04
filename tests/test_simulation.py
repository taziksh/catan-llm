from catanatron import Game

from catan_llm.bots import BOTS, COLORS
from catan_llm.extract import decision_record
from catan_llm.schema import Player
from catan_llm.simulation import rollout_action, score_actions


def test_rollout_action_is_deterministic_and_does_not_mutate_input():
    game = Game(
        [BOTS["value_function"](color) for color in COLORS],
        seed=42,
    )
    policies = {Player(color.value): "value_function" for color in COLORS}
    before = decision_record(game, game.playable_actions[0], i=0).model_dump()
    rng_before = game.state.random.getstate()

    first = rollout_action(game, action_index=0, policies=policies)
    second = rollout_action(game, action_index=0, policies=policies)

    assert first == second
    assert decision_record(game, game.playable_actions[0], i=0).model_dump() == before
    assert game.state.random.getstate() == rng_before
    assert set(first.victory_points) == set(policies)
    assert first.winner is None or first.winner in policies


def test_score_actions_uses_paired_scenarios():
    game = Game(
        [BOTS["value_function"](color) for color in COLORS],
        seed=42,
    )
    policies = {Player(color.value): "value_function" for color in COLORS}
    before = decision_record(game, game.playable_actions[0], i=0).model_dump()

    first = score_actions(
        game,
        action_indices=[0, 1],
        policies=policies,
        scenario_seeds=[7, 11],
    )
    second = score_actions(
        game,
        action_indices=[0, 1],
        policies=policies,
        scenario_seeds=[7, 11],
    )

    assert first == second
    assert set(first) == {0, 1}
    assert all(
        [scenario.seed for scenario in score.scenarios] == [7, 11]
        for score in first.values()
    )
    assert all(0.0 <= score.win_rate <= 1.0 for score in first.values())
    assert decision_record(game, game.playable_actions[0], i=0).model_dump() == before
