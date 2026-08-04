"""Replays a run's eval games and scores each LLM decision against the alpha_beta teacher."""

import argparse
import json
from pathlib import Path

from catan_llm.bots import BOTS
from catan_llm.determinism import check_fixed_hashseed
from catan_llm.extract import to_action
from catan_llm.replay import replay_model_decisions


def replay_game(path: Path, teacher_cls, probe=None):
    """Yield each replayed decision and its teacher/probe result."""
    teacher = None
    for replayed in replay_model_decisions(path):
        game = replayed.game
        playable = game.playable_actions
        catan_map = game.state.board.map
        if teacher is None:
            teacher = teacher_cls(game.state.current_color())
        if probe is None:
            result = to_action(teacher.decide(game, playable), catan_map)
        else:
            result = probe(teacher, game, playable)
        yield replayed, result


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
        path = Path(args.trajectories) / f"{game_id}.jsonl"
        a = t = 0
        for replayed, teacher_action in replay_game(path, teacher_cls):
            decision = replayed.decision
            chosen = decision.legal_actions[decision.chosen_action]
            a += int(teacher_action == chosen)
            t += 1
        agree += a
        total += t
        print(f"{game_id}: {a}/{t}")
    rate = agree / total
    se = (rate * (1 - rate) / total) ** 0.5
    print(f"TOTAL: {agree}/{total} = {rate:.1%} (SE {se:.1%})")


if __name__ == "__main__":
    main()
