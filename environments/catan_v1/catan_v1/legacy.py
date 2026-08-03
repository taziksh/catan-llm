"""Prime Hosted Training-compatible (v0) Catan environment.

Prime's hosted evaluator and the legacy environment bridge load Hub packages
through ``load_environment``.  This adapter keeps the existing game and prompt
logic, while presenting one full Catan game as a stateless multi-turn rollout.
"""

import random
from uuid import uuid4

import verifiers as vf
from catanatron import Game
from catanatron.game import TURNS_LIMIT
from catanatron.state_functions import get_actual_victory_points
from datasets import Dataset

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinism import EVAL_SEED_LIMIT, check_fixed_hashseed
from catan_llm.extract import (
    TrajectoryAccumulator,
    deterministic_game_id,
    observe_live,
)
from catan_llm.parse import parse_move
from catan_llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from catan_llm.serialize import observation_to_prompt
from catan_v1.taskset import (
    DEFAULT_SEATS,
    ENV_VERSION,
    LlmPlayer,
    parse_seat_kinds,
    validate_seed_range,
)

DEFAULT_MAX_TURNS = 500


def _content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


class HostedCatanEnv(vf.MultiTurnEnv):
    """One model seat against three scripted bots, one rollout per full game."""

    def __init__(
        self,
        *,
        seats: str,
        invalid_retries: int,
        trajectory_dir: str | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seat_kinds = parse_seat_kinds(seats, required_agents=1)
        self.agent_seat = self.seat_kinds.index("agent")
        self.invalid_retries = invalid_retries
        self.trajectory_dir = trajectory_dir

    def _is_over(self, game: Game) -> bool:
        return game.winning_color() is not None or game.state.num_turns >= TURNS_LIMIT

    def _record_and_execute(self, state: vf.State, action) -> None:
        game = state["_game"]
        accumulator = state.get("_accumulator")
        if accumulator is not None:
            accumulator.step(game, action)
        game.execute(action)

    def _advance_to_agent(self, state: vf.State) -> None:
        """Play forced moves and scripted seats until the model must choose."""
        game = state["_game"]
        seat_of = state["_seat_of"]
        while not self._is_over(game):
            current = game.state.current_player()
            seat = seat_of[current.color]
            if seat == self.agent_seat and len(game.playable_actions) > 1:
                return
            if seat == self.agent_seat:
                action = game.playable_actions[0]
            else:
                action = current.decide(game, game.playable_actions)
            self._record_and_execute(state, action)

    def _set_decision_prompt(
        self, state: vf.State, *, retry_message: str | None = None
    ) -> None:
        obs = observe_live(state["_game"])
        state["_observation"] = obs
        prompt = observation_to_prompt(obs)
        if retry_message:
            prompt = f"{retry_message}\n\n{prompt}"
        else:
            state["_decisions"] += 1
        state["_current_prompt"] = [
            vf.SystemMessage(content=self.system_prompt or SYSTEM_PROMPT),
            vf.UserMessage(content=prompt),
        ]

    def _finalize_game(self, state: vf.State) -> None:
        if state.get("_finalized"):
            return
        state["_finalized"] = True
        game = state["_game"]
        accumulator = state.get("_accumulator")
        if accumulator is not None:
            accumulator.after(game)

        winner = game.winning_color()
        final_vps = {
            color: get_actual_victory_points(game.state, color)
            for color in game.state.colors
        }
        agent_color = state["_players"][self.agent_seat].color
        agent_vps = final_vps[agent_color]
        opponent_best = max(
            vp for color, vp in final_vps.items() if color != agent_color
        )
        state["_result"] = {
            "won": winner == agent_color,
            "vps": agent_vps,
            "rank": 1.0 + sum(vp > agent_vps for vp in final_vps.values()),
            "vp_margin": (agent_vps - opponent_best) / 10,
            "truncated": winner is None,
            "turns": game.state.num_turns,
        }
        info = dict(state.get("info") or {})
        info["catan"] = {
            "seat": self.agent_seat,
            "color": agent_color.value,
            "turn_position": game.state.colors.index(agent_color),
            "game_id": game.id,
            "env_version": ENV_VERSION,
            "prompt_version": PROMPT_VERSION,
            "seed": state["_seed"],
            "turns": game.state.num_turns,
            "winner": winner.value if winner else None,
        }
        state["info"] = info

    async def setup_state(self, state: vf.State) -> vf.State:
        seed = int(state["info"]["seed"])
        players = [
            LlmPlayer(COLORS[i]) if kind == "agent" else BOTS[kind](COLORS[i])
            for i, kind in enumerate(self.seat_kinds)
        ]
        game = Game(players, seed=seed)
        game.id = f"{deterministic_game_id(game)}_{uuid4().hex[:8]}"
        state["_seed"] = seed
        state["_players"] = players
        state["_game"] = game
        state["_seat_of"] = {player.color: i for i, player in enumerate(players)}
        state["_rng"] = random.Random(seed)
        state["_accumulator"] = None
        state["_retry_count"] = 0
        state["_invalid"] = 0
        state["_decisions"] = 0
        state["_finalized"] = False

        if self.trajectory_dir:
            accumulator = TrajectoryAccumulator(self.trajectory_dir)
            state["_accumulator"] = accumulator
            accumulator.before(game)

        self._advance_to_agent(state)
        if self._is_over(game):
            self._finalize_game(state)
            state["final_env_response"] = [
                vf.UserMessage(content="Game over before an agent decision.")
            ]
        else:
            self._set_decision_prompt(state)
            state["prompt"] = state["_current_prompt"]
        return state

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if not state["trajectory"]:
            return state["_current_prompt"]
        previous = [
            *state["trajectory"][-1]["prompt"],
            *state["trajectory"][-1]["completion"],
        ]
        await self.env_response(previous, state)
        if state.get("final_env_response") is not None:
            return state["final_env_response"]
        return state["_current_prompt"]

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        game = state["_game"]
        obs = state["_observation"]
        index = parse_move(_content(messages[-1]), obs.legal_actions)
        if index is None and state["_retry_count"] < self.invalid_retries:
            state["_retry_count"] += 1
            self._set_decision_prompt(
                state,
                retry_message=(
                    "Your last reply had no valid answer. Reply with "
                    '"answer: <move id>" using one of the ids listed.'
                ),
            )
            return state["_current_prompt"]

        if index is None:
            state["_invalid"] += 1
            action = state["_rng"].choice(game.playable_actions)
        else:
            action = game.playable_actions[index]
        state["_retry_count"] = 0
        self._record_and_execute(state, action)
        self._advance_to_agent(state)

        if self._is_over(game):
            self._finalize_game(state)
            result = state["_result"]
            final = [
                vf.UserMessage(
                    content=(
                        f"Game over. Your final VP: {result['vps']}; "
                        f"won: {result['won']}."
                    )
                )
            ]
            state["final_env_response"] = final
            return final

        self._set_decision_prompt(state)
        return state["_current_prompt"]

    @vf.cleanup
    async def finalize_catan(self, state: vf.State) -> None:
        if "_game" in state:
            self._finalize_game(state)


def _result(state: vf.State) -> dict:
    return state.get("_result") or {}


def reward_win(state: vf.State) -> float:
    return float(_result(state).get("won", False))


def reward_vp(state: vf.State) -> float:
    return min(float(_result(state).get("vps", 0.0)), 10.0) / 10.0


def invalid_rate(state: vf.State) -> float:
    return float(state.get("_invalid", 0)) / max(float(state.get("_decisions", 0)), 1.0)


def truncated(state: vf.State) -> float:
    return float(_result(state).get("truncated", True))


def game_length(state: vf.State) -> float:
    return float(_result(state).get("turns", 0.0))


def decisions(state: vf.State) -> float:
    return float(state.get("_decisions", 0.0))


def rank(state: vf.State) -> float:
    return float(_result(state).get("rank", 4.0))


def vp_margin(state: vf.State) -> float:
    return float(_result(state).get("vp_margin", -1.0))


def load_hosted_environment(
    *,
    seats: str = DEFAULT_SEATS,
    invalid_retries: int = 1,
    vp_coef: float = 0.1,
    trajectory_dir: str | None = None,
    system_prompt: str | None = None,
    seed_start: int = 0,
    num_seeds: int = EVAL_SEED_LIMIT,
    max_turns: int = DEFAULT_MAX_TURNS,
    timeout_seconds: float | None = 1800,
) -> vf.Environment:
    validate_seed_range(seed_start, num_seeds)
    parse_seat_kinds(seats, required_agents=1)
    if invalid_retries < 0:
        raise ValueError("invalid_retries must be non-negative")
    if vp_coef < 0:
        raise ValueError("vp_coef must be non-negative")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive or None")
    check_fixed_hashseed()

    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "Catan game loading."}],
                "answer": "",
                "info": {"seed": seed},
            }
            for seed in range(seed_start, seed_start + num_seeds)
        ]
    )
    rubric = vf.Rubric(
        funcs=[reward_win, reward_vp],
        weights=[1.0, vp_coef],
    )
    for metric in (
        invalid_rate,
        truncated,
        game_length,
        decisions,
        rank,
        vp_margin,
    ):
        rubric.add_metric(metric)
    return HostedCatanEnv(
        dataset=dataset,
        system_prompt=system_prompt or SYSTEM_PROMPT,
        rubric=rubric,
        seats=seats,
        invalid_retries=invalid_retries,
        trajectory_dir=trajectory_dir,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
    )
