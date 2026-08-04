"""Information-set determinization of hidden Catan engine state."""

import random
from collections.abc import Sequence

from catanatron import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.decks import (
    starting_devcard_bank,
    starting_resource_bank,
)
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    VICTORY_POINT,
)
from catanatron.state import State

from catan_llm.simulation import detached_game_copy

PLAYABLE_DEV_CARDS = [
    card for card in DEVELOPMENT_CARDS if card != VICTORY_POINT
]
MAX_DEV_DEALS = 1000


def _opponent_indices(state: State, hero_index: int) -> list[int]:
    return [
        index for index in range(len(state.colors)) if index != hero_index
    ]


def _resample_resources(
    state: State, hero_index: int, rng: random.Random
) -> None:
    """Deal opponents uniform hands from the unseen resource multiset."""
    opponents = _opponent_indices(state, hero_index)
    counts = {
        index: sum(
            state.player_state[f"P{index}_{resource}_IN_HAND"]
            for resource in RESOURCES
        )
        for index in opponents
    }
    pool = []
    for slot, resource in enumerate(RESOURCES):
        unseen = (
            starting_resource_bank()[slot]
            - state.resource_freqdeck[slot]
            - state.player_state[f"P{hero_index}_{resource}_IN_HAND"]
        )
        if unseen < 0:
            raise ValueError(
                f"visible {resource} exceeds the standard supply"
            )
        pool.extend([resource] * unseen)
    if len(pool) != sum(counts.values()):
        raise ValueError(
            "unseen resources do not match opponent hand counts"
        )

    rng.shuffle(pool)
    cursor = 0
    for index in opponents:
        hand = pool[cursor:cursor + counts[index]]
        cursor += counts[index]
        for resource in RESOURCES:
            state.player_state[f"P{index}_{resource}_IN_HAND"] = hand.count(
                resource
            )


def _resample_development_cards(
    state: State, hero_index: int, rng: random.Random, vps_to_win: int
) -> None:
    """Deal opponent dev hands and the draw deck from the unseen multiset.

    Deals are rejected while any opponent's resampled hidden victory-point
    cards would already win the game, since such worlds are impossible for
    an in-progress game.
    """
    opponents = _opponent_indices(state, hero_index)
    counts = {
        index: sum(
            state.player_state[f"P{index}_{card}_IN_HAND"]
            for card in DEVELOPMENT_CARDS
        )
        for index in opponents
    }
    public_points = {
        index: (
            state.player_state[f"P{index}_ACTUAL_VICTORY_POINTS"]
            - state.player_state[f"P{index}_{VICTORY_POINT}_IN_HAND"]
        )
        for index in opponents
    }

    starting_counts = {
        card: starting_devcard_bank().count(card)
        for card in DEVELOPMENT_CARDS
    }
    pool = []
    for card in DEVELOPMENT_CARDS:
        seen = state.player_state[f"P{hero_index}_{card}_IN_HAND"] + sum(
            state.player_state[f"P{index}_PLAYED_{card}"]
            for index in range(len(state.colors))
        )
        unseen = starting_counts[card] - seen
        if unseen < 0:
            raise ValueError(
                f"visible {card} exceeds the standard composition"
            )
        pool.extend([card] * unseen)
    deck_size = len(state.development_listdeck)
    if len(pool) != sum(counts.values()) + deck_size:
        raise ValueError(
            "unseen development cards do not match opponent and deck counts"
        )

    for _ in range(MAX_DEV_DEALS):
        rng.shuffle(pool)
        hands = {}
        cursor = 0
        for index in opponents:
            hands[index] = pool[cursor:cursor + counts[index]]
            cursor += counts[index]
        if all(
            public_points[index] + hands[index].count(VICTORY_POINT)
            < vps_to_win
            for index in opponents
        ):
            break
    else:
        raise ValueError(
            "no feasible hidden development deal below the win threshold"
        )

    for index in opponents:
        for card in DEVELOPMENT_CARDS:
            state.player_state[f"P{index}_{card}_IN_HAND"] = hands[
                index
            ].count(card)
        state.player_state[f"P{index}_ACTUAL_VICTORY_POINTS"] = (
            public_points[index]
            + state.player_state[f"P{index}_{VICTORY_POINT}_IN_HAND"]
        )
        for card in PLAYABLE_DEV_CARDS:
            state.player_state[f"P{index}_{card}_OWNED_AT_START"] = (
                state.player_state[f"P{index}_{card}_IN_HAND"] > 0
            )
    state.development_listdeck = pool[cursor:]


def determinize(game: Game, hero_index: int, seed: int) -> Game:
    """Resample the hidden world behind the hero's information set.

    Keeps the hero's exact cards and all public state; redraws opponent
    hands and the draw deck uniformly from the unseen card multisets. The
    detached copy's RNG is reseeded from ``seed``, so the sampled world
    and its future stochastic outcomes are deterministic given the inputs.

    Args:
        game: Live engine state to determinize. Never mutated.
        hero_index: Seat index into ``game.state.colors``.
        seed: Seed for the sampled world and its future RNG.
    """
    if not 0 <= hero_index < len(game.state.colors):
        raise ValueError(f"invalid hero index: {hero_index}")

    world = detached_game_copy(game)
    rng = random.Random(seed)
    world.random = rng
    world.state.random = rng
    _resample_resources(world.state, hero_index, rng)
    _resample_development_cards(
        world.state, hero_index, rng, world.vps_to_win
    )
    world.playable_actions = generate_playable_actions(world.state)
    return world


def determinize_pair(
    game: Game, hero_index: int, seeds: Sequence[int]
) -> list[Game]:
    """Sample one hidden world per seed, in ``seeds`` order.

    Scoring every candidate action against the same returned worlds pairs
    the comparisons: outcome differences then reflect the actions, not
    the sampled hidden information.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return [determinize(game, hero_index, seed) for seed in seeds]
