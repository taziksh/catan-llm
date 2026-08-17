"""Measures how stable the value model's move picks are at different world counts."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_reward_cache import continuation_seed, hero_color, playable_move_ids, scenario_seeds

from catan_llm.determinism import require_fixed_hashseed
from catan_llm.determinize import determinize
from catan_llm.extract import to_action
from catan_llm.replay import replay_model_decisions
from catan_llm.schema import Player
from catan_llm.serialize import move_id
from catan_llm.value_model_player import afterstate_features

TOTAL_WORLDS = 64
WORLD_COUNTS = [1, 2, 4, 8, 16, 32]


def game_value_matrices(path, model_path, seed, wanted):
    """(game_id, decision) -> (TOTAL_WORLDS, n_moves) value matrix per wanted decision."""
    model = lgb.Booster(model_file=model_path)
    out = {}
    for replayed in replay_model_decisions(Path(path)):
        record = replayed.game_record
        key = (record.game_id, replayed.decision.i)
        if key not in wanted:
            continue
        game = replayed.game
        colors = [Player(color.value) for color in game.state.colors]
        hero_index = colors.index(hero_color(record))
        move_ids = playable_move_ids(replayed)
        if len(move_ids) < 2:
            continue
        rows = []
        for scenario in scenario_seeds(seed, record.game_id, replayed.decision.i, TOTAL_WORLDS):
            world = determinize(game, hero_index, scenario)
            future = continuation_seed(scenario)
            catan_map = world.state.board.map
            indices = {
                move_id(*to_action(action, catan_map)): index
                for index, action in enumerate(world.playable_actions)
            }
            for move in move_ids:
                rows.append(
                    afterstate_features(
                        world, indices[move], future, hero_index, replayed.decision.turn
                    )
                )
        values = model.predict(np.asarray(rows, dtype=np.float32))
        out[key] = values.reshape(TOTAL_WORLDS, len(move_ids))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", default="data/dagger_traces/r3pairs")
    parser.add_argument("--bank", default="data/audits/flip_bank_a.jsonl")
    parser.add_argument("--model", default="experiments/tier1_value_exit/value_model/model.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    require_fixed_hashseed()

    wanted = set()
    for line in open(args.bank):
        row = json.loads(line)
        wanted.add((row["game_id"], row["decision"]))
    games = {game_id for game_id, _ in wanted}
    files = []
    for path in sorted(Path(args.trajectories).glob("*.jsonl")):
        with open(path) as f:
            if json.loads(f.readline())["game_id"] in games:
                files.append(path)
    print(f"{len(wanted)} decisions across {len(files)} games")

    matrices = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(game_value_matrices, str(path), args.model, args.seed, wanted)
            for path in files
        ]
        for future in futures:
            matrices.update(future.result())
    print(f"scored {len(matrices)} decisions at {TOTAL_WORLDS} worlds")

    print(f"{'worlds':>7} {'same pick':>10} {'P(win) given up':>16}")
    for count in WORLD_COUNTS:
        agree, loss = [], []
        for values in matrices.values():
            v_full = values.mean(axis=0)
            best = v_full.argmax()
            groups = values.reshape(-1, count, values.shape[1]).mean(axis=1)
            picks = groups.argmax(axis=1)
            agree.append(np.mean([a == b for a, b in combinations(picks, 2)]))
            loss.append(np.mean(v_full[best] - v_full[picks]))
        print(f"{count:>7} {np.mean(agree):>10.1%} {np.mean(loss):>16.4f}")


if __name__ == "__main__":
    main()
