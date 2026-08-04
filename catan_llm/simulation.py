"""Deterministic continuations from live Catan engine states."""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from catanatron import Game
from catanatron.state_functions import get_actual_victory_points

from catan_llm.bots import BOTS
from catan_llm.schema import Player


@dataclass(frozen=True)
class GameOutcome:
    winner: Player | None
    victory_points: dict[Player, int]
    turns: int

    @property
    def truncated(self) -> bool:
        return self.winner is None


@dataclass(frozen=True)
class ScenarioOutcome:
    seed: int
    outcome: GameOutcome


@dataclass(frozen=True)
class ActionScore:
    action_index: int
    hero: Player
    scenarios: tuple[ScenarioOutcome, ...]

    @property
    def win_rate(self) -> float:
        wins = sum(scenario.outcome.winner == self.hero for scenario in self.scenarios)
        return wins / len(self.scenarios)

    @property
    def mean_vp_margin(self) -> float:
        margins = []
        for scenario in self.scenarios:
            points = scenario.outcome.victory_points
            opponent_best = max(
                value for player, value in points.items() if player != self.hero
            )
            margins.append(points[self.hero] - opponent_best)
        return sum(margins) / len(margins)

    @property
    def truncation_rate(self) -> float:
        truncated = sum(scenario.outcome.truncated for scenario in self.scenarios)
        return truncated / len(self.scenarios)


def game_outcome(game: Game) -> GameOutcome:
    """Read the terminal or truncated outcome of a live game."""
    winner = game.winning_color()
    return GameOutcome(
        winner=Player(winner.value) if winner is not None else None,
        victory_points={
            Player(color.value): get_actual_victory_points(game.state, color)
            for color in game.state.colors
        },
        turns=game.state.num_turns,
    )


def detached_game_copy(game: Game) -> Game:
    """Copy a game without sharing its RNG stream or player list."""
    copied = game.copy()
    copied_rng = random.Random()
    copied_rng.setstate(game.state.random.getstate())
    copied.random = copied_rng
    copied.state.random = copied_rng
    copied.state.players = list(game.state.players)
    copied.playable_actions = list(game.playable_actions)
    return copied


def rollout_action(
    game: Game,
    action_index: int,
    policies: Mapping[Player, str],
    *,
    seed: int | None = None,
) -> GameOutcome:
    """Force one legal action, then finish the copied game with named bots."""
    if not 0 <= action_index < len(game.playable_actions):
        raise ValueError(f"invalid action index: {action_index}")

    colors = {Player(color.value) for color in game.state.colors}
    if set(policies) != colors:
        raise ValueError("policies must specify exactly one bot per player")

    unknown = set(policies.values()) - set(BOTS)
    if unknown:
        raise ValueError(f"unknown bot policies: {', '.join(sorted(unknown))}")

    continuation = detached_game_copy(game)
    if seed is not None:
        continuation_rng = random.Random(seed)
        continuation.random = continuation_rng
        continuation.state.random = continuation_rng
    root_action = continuation.playable_actions[action_index]
    continuation.state.players = [
        BOTS[policies[Player(player.color.value)]](player.color)
        for player in continuation.state.players
    ]
    continuation.execute(root_action)
    continuation.play()
    return game_outcome(continuation)


def score_actions(
    game: Game,
    action_indices: Sequence[int],
    policies: Mapping[Player, str],
    scenario_seeds: Sequence[int],
) -> dict[int, ActionScore]:
    """Score actions using identical future RNG seeds for every action."""
    if not action_indices:
        raise ValueError("action_indices must not be empty")
    if len(set(action_indices)) != len(action_indices):
        raise ValueError("action_indices must be unique")
    if not scenario_seeds:
        raise ValueError("scenario_seeds must not be empty")
    if len(set(scenario_seeds)) != len(scenario_seeds):
        raise ValueError("scenario_seeds must be unique")

    hero = Player(game.state.current_color().value)
    return {
        action_index: ActionScore(
            action_index=action_index,
            hero=hero,
            scenarios=tuple(
                ScenarioOutcome(
                    seed=seed,
                    outcome=rollout_action(
                        game,
                        action_index,
                        policies,
                        seed=seed,
                    ),
                )
                for seed in scenario_seeds
            ),
        )
        for action_index in action_indices
    }
