"""Reconstruct live engine states from logged Catan trajectories."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from catanatron import Game
from catanatron.models.enums import ActionRecord
from catanatron.models.player import Player as EnginePlayer

from catan_llm.bots import COLORS
from catan_llm.extract import to_action
from catan_llm.schema import DecisionRecord, GameRecord, Player
from catan_llm.simulation import detached_game_copy


class ReplayPlayer(EnginePlayer):
    """Placeholder for a seat driven by logged actions."""

    def decide(self, game, playable_actions):
        raise RuntimeError("replay seats are driven by the trajectory")


@dataclass(frozen=True)
class ReplayStep:
    """The live game before a logged decision, or after the last one."""

    game_record: GameRecord
    decision: DecisionRecord | None
    game: Game


@dataclass(frozen=True)
class ReplayedDecision:
    game_record: GameRecord
    decision: DecisionRecord
    game: Game


def _read_game_record(path: Path) -> GameRecord:
    with path.open() as lines:
        first = lines.readline()
    if not first.strip():
        raise ValueError(f"{path}: empty trajectory")
    return GameRecord.model_validate_json(first)


def _load_trajectory(path: Path) -> tuple[GameRecord, list[DecisionRecord]]:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path}: empty trajectory")
    return (
        GameRecord.model_validate_json(lines[0]),
        [DecisionRecord.model_validate_json(line) for line in lines[1:]],
    )


def replay_steps(path: Path) -> Iterator[ReplayStep]:
    """Yield the live game before each logged decision and after the last."""
    game_record, decisions = _load_trajectory(path)
    if game_record.seed is None:
        raise ValueError(f"{path}: trajectory has no seed")

    players = [ReplayPlayer(color) for color in COLORS]
    game = Game(
        players,
        seed=game_record.seed,
        discard_limit=game_record.config.discard_limit,
        vps_to_win=game_record.config.vps_to_win,
    )
    game.id = game_record.game_id

    for decision in decisions:
        engine_actor = Player(game.state.current_color().value)
        if decision.actor != engine_actor:
            raise RuntimeError(
                f"{path.name}: actor diverged at decision {decision.i}"
            )

        playable = game.playable_actions
        replayed_actions = [
            to_action(action, game.state.board.map) for action in playable
        ]
        logged_actions = decision.legal_actions
        if (
            replayed_actions != logged_actions
            and replayed_actions != logged_actions[:-1]
        ):
            raise RuntimeError(
                f"{path.name}: legal actions diverged at decision {decision.i}"
            )

        chosen_index = decision.chosen_action
        if not 0 <= chosen_index < len(playable):
            raise RuntimeError(
                f"{path.name}: decision {decision.i} chose "
                "an unenumerated action"
            )

        yield ReplayStep(game_record=game_record, decision=decision, game=game)

        chosen = playable[chosen_index]
        result = decision.result
        if result is None:
            game.execute(chosen)
        else:
            game.execute(
                chosen,
                action_record=ActionRecord(
                    chosen,
                    tuple(result) if isinstance(result, list) else result,
                ),
            )

    yield ReplayStep(game_record=game_record, decision=None, game=game)


def replay_model_decisions(path: Path) -> Iterator[ReplayedDecision]:
    """Yield a detached engine state at each non-forced model decision."""
    seats = _read_game_record(path).seats
    model_colors = [color for color, kind in seats.items() if kind == "llm"]
    if len(model_colors) != 1:
        raise ValueError(
            f"{path}: expected exactly one llm seat, found {len(model_colors)}"
        )
    model_color = model_colors[0]

    for step in replay_steps(path):
        if step.decision is None:
            continue
        if (
            step.decision.actor == model_color
            and len(step.game.playable_actions) > 1
        ):
            yield ReplayedDecision(
                game_record=step.game_record,
                decision=step.decision,
                game=detached_game_copy(step.game),
            )
