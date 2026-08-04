"""Verifiers v1 taskset and environment for complete Catan games."""

import random
from collections.abc import Iterator
from contextlib import AsyncExitStack
from importlib.metadata import PackageNotFoundError, version
from uuid import uuid4

import verifiers.v1 as vf
from catanatron import Game
from catanatron.game import TURNS_LIMIT
from catanatron.models.player import Player as EnginePlayer
from catanatron.state_functions import get_actual_victory_points
from pydantic import Field, field_validator

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinism import EVAL_SEED_LIMIT, check_fixed_hashseed
from catan_llm.extract import TrajectoryAccumulator, deterministic_game_id, observe_live
from catan_llm.parse import parse_move
from catan_llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from catan_llm.serialize import observation_to_prompt

AGENT_SEAT = "agent"
DEFAULT_SEATS = "agent,value_function,value_function,value_function"


def _environment_version() -> str:
    try:
        return version("catan-v1")
    except PackageNotFoundError:
        return "development"


ENV_VERSION = _environment_version()


def parse_seat_kinds(
    seats: str, *, required_agents: int | None = None
) -> tuple[str, ...]:
    """Parse and validate a four-seat lineup."""
    seat_kinds = tuple(kind.strip() for kind in seats.split(","))
    if len(seat_kinds) != len(COLORS):
        raise ValueError("seats must contain exactly four comma-separated entries")

    unknown = sorted(set(seat_kinds) - {AGENT_SEAT, *BOTS})
    if unknown:
        raise ValueError(f"unknown seat kind(s): {', '.join(unknown)}")

    agent_count = seat_kinds.count(AGENT_SEAT)
    if required_agents is None and agent_count == 0:
        raise ValueError("seats must contain at least one agent")
    if required_agents is not None and agent_count != required_agents:
        count = "one" if required_agents == 1 else str(required_agents)
        raise ValueError(f"seats must contain exactly {count} agent seat")
    return seat_kinds


def validate_seed_range(seed_start: int, num_seeds: int) -> None:
    """Keep evaluation seeds below the training boundary."""
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    if seed_start < EVAL_SEED_LIMIT < seed_start + num_seeds:
        raise ValueError(
            f"seed range crosses into training seeds at {EVAL_SEED_LIMIT}"
        )


class LlmPlayer(EnginePlayer):
    """Seat driven by the env, the engine never calls decide()."""

    def decide(self, game, playable_actions):
        raise RuntimeError("LLM seats are driven by CatanEnv")


class CatanData(vf.TaskData):
    info: dict[str, int]


class CatanEnvConfig(vf.EnvConfig):
    seats: str = DEFAULT_SEATS
    player0: vf.AgentConfig = vf.AgentConfig(harness={"id": "catan_v1_harness"})
    player1: vf.AgentConfig = vf.AgentConfig(harness={"id": "catan_v1_harness"})
    player2: vf.AgentConfig = vf.AgentConfig(harness={"id": "catan_v1_harness"})
    player3: vf.AgentConfig = vf.AgentConfig(harness={"id": "catan_v1_harness"})
    invalid_retries: int = Field(1, ge=0)
    vp_coef: float = Field(0.1, ge=0.0)
    trajectory_dir: str | None = None
    system_prompt: str | None = None

    @field_validator("seats")
    @classmethod
    def validate_seats(cls, seats: str) -> str:
        return ",".join(parse_seat_kinds(seats))


class CatanEnv(vf.Env[CatanEnvConfig]):
    async def run(self, task, agents):
        check_fixed_hashseed()
        seed = task.data.info["seed"]
        seat_kinds = parse_seat_kinds(self.config.seats)
        engine_players = [
            LlmPlayer(COLORS[i]) if kind == AGENT_SEAT else BOTS[kind](COLORS[i])
            for i, kind in enumerate(seat_kinds)
        ]
        game = Game(engine_players, seed=seed)
        game.id = f"{deterministic_game_id(game)}_{uuid4().hex[:8]}"
        rng = random.Random(seed)
        accumulator = None
        if self.config.trajectory_dir:
            accumulator = TrajectoryAccumulator(self.config.trajectory_dir)
            accumulator.before(game)

        seat_of = {p.color: i for i, p in enumerate(engine_players)}
        invalid = [0] * len(seat_kinds)
        dropped = [0] * len(seat_kinds)
        asked = [0] * len(seat_kinds)

        async def ask(interaction, seat):
            obs = observe_live(game)
            state = observation_to_prompt(obs)
            prompt = state
            for _ in range(self.config.invalid_retries + 1):
                try:
                    segment = await interaction.turn(prompt)
                except RuntimeError:
                    # The run can end mid-game, e.g. on provider failure.
                    dropped[seat] += 1
                    return rng.choice(game.playable_actions)
                if segment.terminated:
                    dropped[seat] += 1
                    return rng.choice(game.playable_actions)
                index = parse_move(segment.last_reply or "", obs.legal_actions)
                if index is not None:
                    return game.playable_actions[index]
                prompt = (
                    "Your last reply had no valid answer. Reply with "
                    f'"answer: <move id>" using one of the ids listed.\n\n{state}'
                )
            invalid[seat] += 1
            return rng.choice(game.playable_actions)

        async with AsyncExitStack() as stack:
            interactions = {}
            for i, kind in enumerate(seat_kinds):
                if kind == AGENT_SEAT:
                    seat_task = vf.Task(
                        vf.TaskData(
                            idx=task.data.idx,
                            prompt=None,
                            system_prompt=self.config.system_prompt or SYSTEM_PROMPT,
                        )
                    )
                    agent = getattr(agents, f"player{i}")
                    interactions[i] = await stack.enter_async_context(
                        agent.interaction(seat_task)
                    )

            while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
                current = game.state.current_player()
                seat = seat_of[current.color]
                if seat in interactions and len(game.playable_actions) > 1:
                    asked[seat] += 1
                    action = await ask(interactions[seat], seat)
                elif seat in interactions:
                    action = game.playable_actions[0]
                else:
                    action = current.decide(game, game.playable_actions)
                if accumulator is not None:
                    accumulator.step(game, action)
                game.execute(action)

        if accumulator is not None:
            accumulator.after(game)
        winner = game.winning_color()
        final_vps = {
            color: get_actual_victory_points(game.state, color)
            for color in game.state.colors
        }
        for i, interaction in interactions.items():
            color = engine_players[i].color
            vps = final_vps[color]
            trace = interaction.trace
            trace.record_reward("reward_win", float(winner == color))
            trace.record_reward(
                "reward_vp", min(vps, 10) / 10, weight=self.config.vp_coef
            )
            trace.record_metric("invalid_rate", invalid[i] / max(asked[i], 1))
            trace.record_metric("dropped_rate", dropped[i] / max(asked[i], 1))
            trace.record_metric("truncated", float(winner is None))
            trace.record_metric("game_length", float(game.state.num_turns))
            trace.record_metric("decisions", float(asked[i]))
            trace.record_metric("rank", 1.0 + sum(v > vps for v in final_vps.values()))
            trace.record_metric(
                "vp_margin",
                (vps - max(v for c, v in final_vps.items() if c != color)) / 10,
            )
            trace.info["catan"] = {
                "seat": i,
                "color": color.value,
                "turn_position": game.state.colors.index(color),
                "game_id": game.id,
                "env_version": ENV_VERSION,
                "prompt_version": PROMPT_VERSION,
                "seed": seed,
                "turns": game.state.num_turns,
                "winner": winner.value if winner else None,
            }


class CatanTasksetConfig(vf.TasksetConfig):
    seed_start: int = Field(0, ge=0)

    @field_validator("seed_start")
    @classmethod
    def validate_seed_start(cls, seed_start: int) -> int:
        validate_seed_range(seed_start, 1)
        return seed_start


class CatanTaskset(vf.Taskset[vf.Task[CatanData], CatanTasksetConfig]):
    def load(self) -> Iterator[vf.Task]:
        start = self.config.seed_start
        end = EVAL_SEED_LIMIT if start < EVAL_SEED_LIMIT else start + EVAL_SEED_LIMIT
        for i in range(start, end):
            yield vf.Task(
                CatanData(idx=i, name=f"game#{i}", prompt=None, info={"seed": i})
            )
