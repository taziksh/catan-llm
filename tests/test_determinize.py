"""Checks that determinized worlds conserve the card supply, keep the
hero's cards and all public state intact, and reproduce per seed."""

import pytest
from catanatron import Game
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    VICTORY_POINT,
)

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinize import determinize, determinize_pair

SEED = 42


def _new_game():
    return Game(
        [BOTS["value_function"](color) for color in COLORS],
        seed=SEED,
    )


def _resource_total(state, index):
    return sum(
        state.player_state[f"P{index}_{resource}_IN_HAND"]
        for resource in RESOURCES
    )


def _dev_total(state, index):
    return sum(
        state.player_state[f"P{index}_{card}_IN_HAND"]
        for card in DEVELOPMENT_CARDS
    )


def _opponents(game, hero):
    return [
        index for index in range(len(game.state.colors)) if index != hero
    ]


@pytest.fixture(scope="module")
def mid_game():
    """A game paused where at least two opponents hold hidden cards."""
    game = _new_game()
    while game.winning_color() is None:
        game.play_tick()
        hero = game.state.current_player_index
        totals = [
            _resource_total(game.state, index)
            for index in _opponents(game, hero)
        ]
        if sum(totals) >= 6 and sum(total > 0 for total in totals) >= 2:
            return game
    raise AssertionError("game ended before a suitable mid-game state")


def _snapshot(game):
    return (
        dict(game.state.player_state),
        list(game.state.resource_freqdeck),
        list(game.state.development_listdeck),
        game.state.random.getstate(),
        list(game.playable_actions),
        {
            color: dict(buildings)
            for color, buildings in game.state.buildings_by_color.items()
        },
    )


def test_resource_and_dev_conservation(mid_game):
    hero = mid_game.state.current_player_index
    world = determinize(mid_game, hero, seed=7)
    state = world.state
    indices = range(len(state.colors))

    for slot, resource in enumerate(RESOURCES):
        held = sum(
            state.player_state[f"P{index}_{resource}_IN_HAND"]
            for index in indices
        )
        assert state.resource_freqdeck[slot] + held == 19

    for card in DEVELOPMENT_CARDS:
        held = sum(
            state.player_state[f"P{index}_{card}_IN_HAND"]
            + state.player_state[f"P{index}_PLAYED_{card}"]
            for index in indices
        )
        assert (
            state.development_listdeck.count(card) + held
            == starting_devcard_bank().count(card)
        )


def test_hero_hand_is_invariant(mid_game):
    hero = mid_game.state.current_player_index
    world = determinize(mid_game, hero, seed=7)
    for key in [f"{item}_IN_HAND" for item in RESOURCES + DEVELOPMENT_CARDS]:
        assert (
            world.state.player_state[f"P{hero}_{key}"]
            == mid_game.state.player_state[f"P{hero}_{key}"]
        )


def test_public_state_is_invariant(mid_game):
    hero = mid_game.state.current_player_index
    world = determinize(mid_game, hero, seed=7)
    state, original = world.state, mid_game.state

    assert state.resource_freqdeck == original.resource_freqdeck
    assert len(state.development_listdeck) == len(
        original.development_listdeck
    )
    assert state.buildings_by_color == original.buildings_by_color
    assert state.board.robber_coordinate == original.board.robber_coordinate
    assert state.current_player_index == original.current_player_index
    assert state.num_turns == original.num_turns

    for index in _opponents(mid_game, hero):
        assert _resource_total(state, index) == _resource_total(
            original, index
        )
        assert _dev_total(state, index) == _dev_total(original, index)
        for card in DEVELOPMENT_CARDS:
            assert (
                state.player_state[f"P{index}_PLAYED_{card}"]
                == original.player_state[f"P{index}_PLAYED_{card}"]
            )
        assert (
            state.player_state[f"P{index}_VICTORY_POINTS"]
            == original.player_state[f"P{index}_VICTORY_POINTS"]
        )

    for index in range(len(state.colors)):
        assert state.player_state[f"P{index}_ACTUAL_VICTORY_POINTS"] == (
            state.player_state[f"P{index}_VICTORY_POINTS"]
            + state.player_state[f"P{index}_{VICTORY_POINT}_IN_HAND"]
        )


def test_same_seed_reproduces_world(mid_game):
    hero = mid_game.state.current_player_index
    first = determinize(mid_game, hero, seed=11)
    second = determinize(mid_game, hero, seed=11)
    assert first.state.player_state == second.state.player_state
    assert (
        first.state.development_listdeck
        == second.state.development_listdeck
    )
    assert (
        first.state.random.getstate() == second.state.random.getstate()
    )
    assert first.playable_actions == second.playable_actions


def test_different_seeds_vary(mid_game):
    hero = mid_game.state.current_player_index
    worlds = {
        (
            tuple(
                sorted(world.state.player_state.items())
            ),
            tuple(world.state.development_listdeck),
        )
        for world in (
            determinize(mid_game, hero, seed) for seed in range(12)
        )
    }
    assert len(worlds) > 1


def test_input_game_is_not_mutated(mid_game):
    hero = mid_game.state.current_player_index
    before = _snapshot(mid_game)
    determinize(mid_game, hero, seed=3)
    determinize_pair(mid_game, hero, seeds=[5, 6])
    assert _snapshot(mid_game) == before


def test_world_is_playable(mid_game):
    hero = mid_game.state.current_player_index
    world = determinize(mid_game, hero, seed=13)
    assert world.playable_actions == mid_game.playable_actions
    world.execute(world.playable_actions[0])
    for _ in range(50):
        if world.winning_color() is not None:
            break
        world.play_tick()


def test_determinize_pair_matches_single_calls(mid_game):
    hero = mid_game.state.current_player_index
    worlds = determinize_pair(mid_game, hero, seeds=[3, 9])
    singles = [
        determinize(mid_game, hero, seed) for seed in (3, 9)
    ]
    assert [world.state.player_state for world in worlds] == [
        single.state.player_state for single in singles
    ]
    assert [world.state.development_listdeck for world in worlds] == [
        single.state.development_listdeck for single in singles
    ]


def test_determinize_validates_inputs(mid_game):
    hero = mid_game.state.current_player_index
    with pytest.raises(ValueError):
        determinize(mid_game, hero_index=-1, seed=0)
    with pytest.raises(ValueError):
        determinize(mid_game, hero_index=4, seed=0)
    with pytest.raises(ValueError):
        determinize_pair(mid_game, hero, seeds=[])
    with pytest.raises(ValueError):
        determinize_pair(mid_game, hero, seeds=[3, 3])


def test_zero_card_opponents():
    game = _new_game()
    hero = game.state.current_player_index
    world = determinize(game, hero, seed=5)
    assert world.state.player_state == game.state.player_state
    assert len(world.state.development_listdeck) == 25
    assert world.state.resource_freqdeck == [19] * 5


def test_empty_bank(mid_game):
    hero = mid_game.state.current_player_index
    base = determinize(mid_game, hero, seed=17)
    state = base.state
    for slot, resource in enumerate(RESOURCES):
        state.player_state[f"P{hero}_{resource}_IN_HAND"] += (
            state.resource_freqdeck[slot]
        )
        state.resource_freqdeck[slot] = 0

    world = determinize(base, hero, seed=23)
    assert world.state.resource_freqdeck == [0] * 5
    for index in _opponents(base, hero):
        assert _resource_total(world.state, index) == _resource_total(
            state, index
        )
    for slot, resource in enumerate(RESOURCES):
        held = sum(
            world.state.player_state[f"P{index}_{resource}_IN_HAND"]
            for index in range(len(state.colors))
        )
        assert held == 19


def test_empty_dev_deck(mid_game):
    hero = mid_game.state.current_player_index
    base = determinize(mid_game, hero, seed=19)
    state = base.state
    opponents = _opponents(base, hero)
    for position, card in enumerate(state.development_listdeck):
        index = opponents[position % len(opponents)]
        state.player_state[f"P{index}_{card}_IN_HAND"] += 1
        if card == VICTORY_POINT:
            state.player_state[f"P{index}_ACTUAL_VICTORY_POINTS"] += 1
    state.development_listdeck = []

    world = determinize(base, hero, seed=29)
    assert world.state.development_listdeck == []
    for index in opponents:
        assert _dev_total(world.state, index) == _dev_total(state, index)
    for card in DEVELOPMENT_CARDS:
        held = sum(
            world.state.player_state[f"P{index}_{card}_IN_HAND"]
            + world.state.player_state[f"P{index}_PLAYED_{card}"]
            for index in range(len(state.colors))
        )
        assert held == starting_devcard_bank().count(card)
