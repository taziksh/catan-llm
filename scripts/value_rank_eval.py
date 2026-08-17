"""Ranks logged decisions by value-model afterstates and scores stability against the reward cache."""

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from build_reward_cache import continuation_seed, hero_color, playable_move_ids, scenario_seeds

from catanatron.features import create_sample_vector
from catan_llm.determinism import require_fixed_hashseed
from catan_llm.determinize import determinize
from catan_llm.extract import to_action
from catan_llm.replay import replay_model_decisions
from catan_llm.schema import Player
from catan_llm.serialize import move_id
from catan_llm.simulation import detached_game_copy

MARGIN_BINS = [0.005, 0.01, 0.02, 0.05, 0.1]


def afterstate_features(world, move_index, future_seed, hero_index, turn):
    afterstate = detached_game_copy(world)
    rng = random.Random(future_seed)
    afterstate.random = rng
    afterstate.state.random = rng
    afterstate.execute(afterstate.playable_actions[move_index])
    hero = afterstate.state.colors[hero_index]
    row = create_sample_vector(afterstate, hero)
    row += [
        float(hero_index),
        float(turn),
        float(afterstate.state.current_color() == hero),
    ]
    return row


def rank_game(path, model_path, seed, worlds):
    model = lgb.Booster(model_file=model_path)
    results = []
    for replayed in replay_model_decisions(Path(path)):
        game = replayed.game
        record = replayed.game_record
        hero = hero_color(record)
        colors = [Player(color.value) for color in game.state.colors]
        hero_index = colors.index(hero)
        move_ids = playable_move_ids(replayed)
        if len(move_ids) < 2:
            continue
        rows = []
        for scenario in scenario_seeds(seed, record.game_id, replayed.decision.i, worlds):
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
        values = values.reshape(worlds, len(move_ids))
        half = worlds // 2
        results.append(
            {
                "game_id": record.game_id,
                "decision": replayed.decision.i,
                "turn": replayed.decision.turn,
                "phase": replayed.decision.phase,
                "hero_vp": replayed.decision.players[hero].vps_actual,
                "moves": move_ids,
                "v_mean": values.mean(axis=0).tolist(),
                "top": move_ids[int(values.mean(axis=0).argmax())],
                "top_a": move_ids[int(values[:half].mean(axis=0).argmax())],
                "top_b": move_ids[int(values[half:].mean(axis=0).argmax())],
            }
        )
    return results


def report(results, cache):
    scored = [r for r in results if (r["game_id"], r["decision"]) in cache]
    agree_half = np.mean([r["top_a"] == r["top_b"] for r in scored])
    print(f"decisions: {len(scored)} (of {len(results)} ranked)")
    print(f"split-half self-agreement: {agree_half:.1%}")

    best = {}
    for key, r in cache.items():
        ordered = sorted(r.values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        best[key] = (max(r, key=r.get), margin)
    hits = [r["top"] == best[(r["game_id"], r["decision"])][0] for r in scored]
    print(f"engine-best agreement: {np.mean(hits):.1%}")
    for lo, hi in zip([0] + MARGIN_BINS, MARGIN_BINS + [float("inf")]):
        rows = [
            r["top"] == best[(r["game_id"], r["decision"])][0]
            for r in scored
            if lo <= best[(r["game_id"], r["decision"])][1] < hi
        ]
        if rows:
            print(f"  mc margin [{lo}, {hi}): {np.mean(rows):.1%} of {len(rows)}")

    correlations = []
    for r in scored:
        values = cache[(r["game_id"], r["decision"])]
        paired = [(v, values[m]) for m, v in zip(r["moves"], r["v_mean"]) if m in values]
        if len(paired) > 2:
            rho = spearmanr([p[0] for p in paired], [p[1] for p in paired]).statistic
            if not np.isnan(rho):
                correlations.append(rho)
    print(f"mean Spearman vs playout rewards: {np.mean(correlations):.3f} over {len(correlations)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", default="data/dagger_traces/r3pairs")
    parser.add_argument("--blunders", default="experiments/decision_regret/blunders.jsonl")
    parser.add_argument("--model", default="experiments/tier1_value_exit/value_model/model.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--out", default="experiments/tier1_value_exit/rank_eval.jsonl")
    args = parser.parse_args()
    require_fixed_hashseed()

    cache = defaultdict(dict)
    for line in open(args.blunders):
        row = json.loads(line)
        cache[(row["game_id"], row["decision"])] = row["move_values"]

    files = sorted(Path(args.trajectories).glob("*.jsonl"))[: args.max_games]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(rank_game, str(path), args.model, args.seed, args.worlds)
            for path in files
        ]
        for i, future in enumerate(as_completed(futures)):
            results.extend(future.result())
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(files)} games ranked")

    results.sort(key=lambda r: (r["game_id"], r["decision"]))
    with open(args.out, "w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")
    report(results, cache)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
