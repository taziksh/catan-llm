"""catan-v1: seeded Settlers of Catan with LLM seats vs scripted bots or self-play.

One rollout is one full game. Scripted seats decide host-side, LLM seats get a
self-contained state prompt per decision and answer with an option index.
"""

import random
from contextlib import AsyncExitStack
from typing import Iterator

import verifiers.v1 as vf
from catanatron import Game
from catanatron.game import TURNS_LIMIT
from catanatron.models.player import Player as EnginePlayer
from catanatron.state_functions import get_actual_victory_points
from pydantic import Field

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinism import EVAL_SEED_LIMIT, check_fixed_hashseed
from catan_llm.extract import TrajectoryAccumulator, deterministic_game_id, observe_live
from catan_llm.parse import parse_answer
from catan_llm.serialize import observation_to_prompt

SYSTEM_PROMPT = (
    "You are playing Settlers of Catan. Each turn you receive the full game "
    "state and a numbered list of your legal options. You may reason first; "
    'only your final "answer: <option number>" line counts.'
)


class LlmPlayer(EnginePlayer):
    """Seat driven by the env, the engine never calls decide()."""

    def decide(self, game, playable_actions):
        raise RuntimeError("LLM seats are driven by CatanEnv")


class CatanData(vf.TaskData):
    info: dict  # {"seed": int}


class CatanEnvConfig(vf.EnvConfig):
    seats: str = "agent,value_function,value_function,value_function"  # per seat: "agent" | BOTS key
    player0: vf.AgentConfig = vf.AgentConfig(harness={"id": "catan_v1_harness"})
    player1: vf.AgentConfig = vf.AgentConfig(harness={"id": "catan_v1_harness"})
    player2: vf.AgentConfig = vf.AgentConfig(harness={"id": "catan_v1_harness"})
    player3: vf.AgentConfig = vf.AgentConfig(harness={"id": "catan_v1_harness"})
    invalid_retries: int = Field(1, ge=0)
    vp_coef: float = 0.1  # weight of reward_vp = min(vps, 10) / 10
    trajectory_dir: str | None = None


class CatanEnv(vf.Env[CatanEnvConfig]):
    async def run(self, task, agents):
        check_fixed_hashseed()
        seed = task.data.info["seed"]
        seat_kinds = self.config.seats.split(",")
        engine_players = [
            LlmPlayer(COLORS[i]) if kind == "agent" else BOTS[kind](COLORS[i])
            for i, kind in enumerate(seat_kinds)
        ]
        game = Game(engine_players, seed=seed)
        game.id = deterministic_game_id(game)
        rng = random.Random(seed)
        accumulator = None
        if self.config.trajectory_dir:
            accumulator = TrajectoryAccumulator(self.config.trajectory_dir)
            accumulator.before(game)

        seat_of = {p.color: i for i, p in enumerate(engine_players)}
        invalid = [0] * len(seat_kinds)
        asked = [0] * len(seat_kinds)

        async def ask(interaction, seat):
            n = len(game.playable_actions)
            state = observation_to_prompt(observe_live(game))
            prompt = state
            for _ in range(self.config.invalid_retries + 1):
                try:
                    segment = await interaction.turn(prompt)
                except RuntimeError:
                    # The run can end mid-game, e.g. on provider failure.
                    break
                if segment.terminated:
                    break
                index = parse_answer(segment.last_reply or "", n)
                if index is not None:
                    return game.playable_actions[index]
                # The stateless harness drops history, so a retry restates the
                # full state alongside the complaint.
                prompt = (
                    "Your last reply had no valid answer. Reply with "
                    f'"answer: <option number>" between 0 and {n - 1}.\n\n{state}'
                )
            invalid[seat] += 1
            return rng.choice(game.playable_actions)

        async with AsyncExitStack() as stack:
            interactions = {}
            for i, kind in enumerate(seat_kinds):
                if kind == "agent":
                    seat_task = vf.Task(
                        vf.TaskData(
                            idx=task.data.idx, prompt=None, system_prompt=SYSTEM_PROMPT
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
                if accumulator:
                    accumulator.step(game, action)
                game.execute(action)

        if accumulator:
            accumulator.after(game)
        winner = game.winning_color()
        final_vps = {c: get_actual_victory_points(game.state, c) for c in game.state.colors}
        for i, interaction in interactions.items():
            color = engine_players[i].color
            vps = final_vps[color]
            trace = interaction.trace
            trace.record_reward("reward_win", float(winner == color))
            trace.record_reward("reward_vp", min(vps, 10) / 10, weight=self.config.vp_coef)
            trace.record_metric("invalid_rate", invalid[i] / max(asked[i], 1))
            trace.record_metric("truncated", float(winner is None))
            trace.record_metric("game_length", float(game.state.num_turns))
            trace.record_metric("decisions", float(asked[i]))
            trace.record_metric("rank", 1.0 + sum(v > vps for v in final_vps.values()))
            trace.record_metric(
                "vp_margin", (vps - max(v for c, v in final_vps.items() if c != color)) / 10
            )
            trace.info["catan"] = {
                "seat": i,
                "color": color.value,
                "turn_position": game.state.colors.index(color),
                "game_id": game.id,
                "seed": seed,
                "turns": game.state.num_turns,
                "winner": winner.value if winner else None,
            }


class CatanTaskset(vf.Taskset[vf.Task[CatanData], vf.TasksetConfig]):
    def load(self) -> Iterator[vf.Task]:
        # Bounded so eval seeds can never cross into the training range.
        for i in range(EVAL_SEED_LIMIT):
            yield vf.Task(
                CatanData(idx=i, name=f"game#{i}", prompt=None, info={"seed": i})
            )
