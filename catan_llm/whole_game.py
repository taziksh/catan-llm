"""Synchronous on-policy Catan trajectories for whole-game RL."""

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

from catanatron import Color, Game
from catanatron.models.player import Player as EnginePlayer

from catan_llm.bots import BOTS, COLORS
from catan_llm.extract import observe_live
from catan_llm.parse import parse_move
from catan_llm.schema import Action, Player
from catan_llm.serialize import observation_to_prompt
from catan_llm.simulation import GameOutcome, game_outcome

DEFAULT_INVALID_RETRIES = 1
DEFAULT_MAX_TURNS = 500
DEFAULT_VP_COEF = 0.1
OPPONENT_POLICY = "value_function"
RETRY_MESSAGE = (
    "Your last reply had no valid answer. Reply with "
    '"answer: <move id>" using one of the ids listed.'
)


class PolicyPlayer(EnginePlayer):
    """Engine placeholder for the seat controlled by the language model."""

    def decide(self, game, playable_actions):
        raise RuntimeError("policy seats are driven by the GRPO rollout")


@dataclass
class SampledCompletion:
    text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    error: str | None = None


@dataclass
class DecisionSample:
    prompt_ids: list[int]
    completion_ids: list[int]
    old_logprobs: list[float] | None = None


@dataclass
class Rollout:
    seed: int
    rollout_index: int
    game: Game
    hero_color: Color
    fallback_rng: random.Random
    samples: list[DecisionSample] = field(default_factory=list)
    invalid_replies: int = 0
    decision_states: int = 0
    retry_count: int = 0
    failed: str | None = None
    outcome: GameOutcome | None = None
    reward: float | None = None
    advantage: float = 0.0

    @property
    def hero(self) -> Player:
        return Player(self.hero_color.value)


@dataclass(frozen=True)
class PendingDecision:
    rollout: Rollout
    user_prompt: str
    legal_actions: list[Action]


def hero_vps(rollout: Rollout) -> int:
    if rollout.outcome is None:
        raise ValueError("rollout has no outcome")
    return rollout.outcome.victory_points[rollout.hero]


def vp_margin(rollout: Rollout) -> int:
    if rollout.outcome is None:
        raise ValueError("rollout has no outcome")
    points = rollout.outcome.victory_points
    return points[rollout.hero] - max(
        value for player, value in points.items() if player != rollout.hero
    )


def terminal_reward(rollout: Rollout, vp_coef: float = DEFAULT_VP_COEF) -> float:
    """Project reward: a dominant win term plus normalized own VP."""
    if rollout.outcome is None:
        raise ValueError("rollout has no outcome")
    won = rollout.outcome.winner == rollout.hero
    return float(won) + vp_coef * min(hero_vps(rollout), 10) / 10


def group_advantages(rewards: list[float]) -> list[float]:
    """Center and normalize rewards within one rollout group."""
    if not rewards:
        raise ValueError("rewards must not be empty")
    baseline = sum(rewards) / len(rewards)
    scale = statistics.pstdev(rewards)
    if scale == 0.0:
        return [0.0] * len(rewards)
    return [(reward - baseline) / scale for reward in rewards]


def _new_rollout(
    seed: int,
    rollout_index: int,
    run_seed: int,
    opponents: Sequence[str] = (OPPONENT_POLICY,) * 3,
) -> Rollout:
    players = [
        PolicyPlayer(COLORS[0]),
        *(BOTS[name](color) for name, color in zip(opponents, COLORS[1:], strict=True)),
    ]
    game = Game(players, seed=seed)
    game.id = f"grpo_s{seed}_r{rollout_index}"
    return Rollout(
        seed=seed,
        rollout_index=rollout_index,
        game=game,
        hero_color=players[0].color,
        fallback_rng=random.Random(f"{run_seed}:{seed}:{rollout_index}:fallback"),
    )


def training_seeds(seed_start: int, offset: int, count: int) -> list[int]:
    """Pick count board seeds, skipping the first offset eligible ones.

    Seeds divisible by 10 are reserved for validation.
    """
    seeds = []
    seed = seed_start
    to_skip = offset
    while len(seeds) < count:
        if seed % 10 != 0:
            if to_skip:
                to_skip -= 1
            else:
                seeds.append(seed)
        seed += 1
    return seeds


def make_rollouts(
    seeds: Sequence[int],
    group_size: int,
    run_seed: int,
    opponents: Sequence[str] = (OPPONENT_POLICY,) * 3,
) -> list[Rollout]:
    return [
        _new_rollout(seed, rollout_index, run_seed, opponents)
        for seed in seeds
        for rollout_index in range(group_size)
    ]


def _is_over(rollout: Rollout, max_turns: int) -> bool:
    game = rollout.game
    return game.winning_color() is not None or game.state.num_turns >= max_turns


def _advance_to_policy(rollout: Rollout, max_turns: int) -> PendingDecision | None:
    """Play bots and forced hero actions until a model decision is needed."""
    game = rollout.game
    while not _is_over(rollout, max_turns):
        current = game.state.current_player()
        if current.color == rollout.hero_color and len(game.playable_actions) > 1:
            observation = observe_live(game)
            state = observation_to_prompt(observation)
            if rollout.retry_count:
                state = f"{RETRY_MESSAGE}\n\n{state}"
            else:
                rollout.decision_states += 1
            return PendingDecision(
                rollout=rollout,
                user_prompt=state,
                legal_actions=list(observation.legal_actions),
            )
        if current.color == rollout.hero_color:
            action = game.playable_actions[0]
        else:
            action = current.decide(game, game.playable_actions)
        game.execute(action)
    rollout.outcome = game_outcome(game)
    return None


def rollout_games(
    rollouts: list[Rollout],
    complete: Callable[[list[str]], list[SampledCompletion]],
    *,
    invalid_retries: int = DEFAULT_INVALID_RETRIES,
    max_turns: int = DEFAULT_MAX_TURNS,
    vp_coef: float = DEFAULT_VP_COEF,
) -> list[Rollout]:
    """Finish a synchronous rollout batch using batched model completions."""
    if invalid_retries < 0:
        raise ValueError("invalid_retries must be non-negative")
    while True:
        pending = [
            decision
            for rollout in rollouts
            if rollout.failed is None and rollout.outcome is None
            if (decision := _advance_to_policy(rollout, max_turns)) is not None
        ]
        if not pending:
            break
        completions = complete([decision.user_prompt for decision in pending])
        if len(completions) != len(pending):
            raise RuntimeError(
                f"sampler returned {len(completions)} completions "
                f"for {len(pending)} prompts"
            )
        for decision, completion in zip(pending, completions, strict=True):
            rollout = decision.rollout
            if completion.error is not None:
                rollout.failed = completion.error
                continue
            index = parse_move(completion.text, decision.legal_actions)
            if index is None:
                rollout.invalid_replies += 1
                rollout.retry_count += 1
                if rollout.retry_count <= invalid_retries:
                    continue
                action = rollout.fallback_rng.choice(rollout.game.playable_actions)
            else:
                rollout.samples.append(
                    DecisionSample(
                        prompt_ids=completion.prompt_ids,
                        completion_ids=completion.completion_ids,
                    )
                )
                action = rollout.game.playable_actions[index]
            rollout.retry_count = 0
            rollout.game.execute(action)

    for rollout in rollouts:
        if rollout.failed is None and rollout.outcome is not None:
            rollout.reward = terminal_reward(rollout, vp_coef)
    return rollouts


def assign_group_advantages(rollouts: list[Rollout]) -> dict:
    """Center completed rewards separately for every seeded-board group."""
    grouped = {}
    for rollout in rollouts:
        grouped.setdefault(rollout.seed, []).append(rollout)

    degenerate = 0
    failed = 0
    for group in grouped.values():
        completed = [rollout for rollout in group if rollout.reward is not None]
        failed += len(group) - len(completed)
        if len(completed) < 2:
            degenerate += 1
            continue
        advantages = group_advantages([float(rollout.reward) for rollout in completed])
        if all(value == 0.0 for value in advantages):
            degenerate += 1
        for rollout, advantage in zip(completed, advantages, strict=True):
            rollout.advantage = advantage
    return {
        "groups": len(grouped),
        "degenerate_groups": degenerate,
        "failed_games": failed,
    }


def trainable_samples(rollouts: list[Rollout]) -> list[tuple[Rollout, DecisionSample]]:
    return [
        (rollout, sample)
        for rollout in rollouts
        if rollout.reward is not None and rollout.advantage != 0.0
        for sample in rollout.samples
    ]
