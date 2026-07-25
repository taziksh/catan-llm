"""Replays a run's eval games and scores each LLM decision against the alpha_beta teacher."""

import argparse
import json
from pathlib import Path

from catanatron import Game
from catanatron.models.player import Player as EnginePlayer

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinism import check_fixed_hashseed
from catan_llm.extract import to_action


class ReplayPlayer(EnginePlayer):
    """Placeholder for the LLM seat during replay."""

    def decide(self, game, playable_actions):
        raise RuntimeError("replay seats are driven by the log")


def _normalize(action) -> list:
    return json.loads(json.dumps(list(action)))


def replay_game(path: Path, teacher_cls) -> tuple[int, int]:
    """Returns (agreements, comparisons) for one logged game."""
    lines = [json.loads(line) for line in open(path)]
    header, decisions = lines[0], lines[1:]
    seats = header["seats"]
    llm_color = next(color for color, kind in seats.items() if kind == "llm")
    players = [
        ReplayPlayer(color) if seats[color.value] == "llm" else BOTS[seats[color.value]](color)
        for color in COLORS
    ]
    game = Game(players, seed=header["seed"])
    rng = game.state.random
    by_color = {player.color.value: player for player in players}
    teacher = teacher_cls(next(c for c in COLORS if c.value == llm_color))
    agree, total = 0, 0
    for decision in decisions:
        playable = game.playable_actions
        catan_map = game.state.board.map
        logged = decision["legal_actions"]
        replayed = [_normalize(to_action(a, catan_map)) for a in playable]
        # decision_record may append one OFFER_TRADE the engine does not enumerate.
        if replayed != logged and replayed != logged[:-1]:
            raise RuntimeError(f"{path.name}: replay diverged at decision {decision['i']}")
        index = decision["chosen_action"]
        if index >= len(playable):
            raise RuntimeError(f"{path.name}: decision {decision['i']} chose an unenumerated action")
        chosen = playable[index]
        if decision["actor"] == llm_color:
            if len(playable) > 1:
                # game.copy() shares state.random, so shield it from the teacher's simulations.
                snapshot = rng.getstate()
                teacher_pick = teacher.decide(game.copy(), playable)
                rng.setstate(snapshot)
                agree += int(teacher_pick == chosen)
                total += 1
        else:
            # Bot decide() simulations consume the shared rng, exactly as in the original game.
            bot_pick = by_color[decision["actor"]].decide(game, playable)
            if bot_pick != chosen:
                raise RuntimeError(f"{path.name}: bot deviated at decision {decision['i']}")
        game.execute(chosen)
    return agree, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="outputs run dir containing traces.jsonl")
    parser.add_argument("--trajectories", required=True, help="dir with the run's game jsonl files")
    parser.add_argument("--teacher", default="alpha_beta")
    args = parser.parse_args()
    check_fixed_hashseed()

    teacher_cls = BOTS[args.teacher]
    game_ids = []
    for line in open(Path(args.run) / "traces.jsonl"):
        episode = json.loads(line)
        game_ids.append(episode["traces"][0]["info"]["catan"]["game_id"])

    agree, total = 0, 0
    for game_id in game_ids:
        a, t = replay_game(Path(args.trajectories) / f"{game_id}.jsonl", teacher_cls)
        agree += a
        total += t
        print(f"{game_id}: {a}/{t}")
    rate = agree / total
    se = (rate * (1 - rate) / total) ** 0.5
    print(f"TOTAL: {agree}/{total} = {rate:.1%} (SE {se:.1%})")


if __name__ == "__main__":
    main()
