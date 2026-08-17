"""Samples logged bot games into a feature matrix labeled with terminal outcomes."""

import argparse
import json
import random
import re
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from catanatron.features import create_sample_vector, get_feature_ordering

from catan_llm.determinism import EVAL_SEED_LIMIT, require_fixed_hashseed
from catan_llm.replay import replay_steps
from catan_llm.schema import Player
from catan_llm.simulation import VP_CAP

R3PAIRS_SEEDS = range(10_150, 10_450)  # same seeds = same boards as the regret dataset
EXTRA_FEATURES = ["HERO_SEAT", "TURN", "IS_HERO_TURN"]

SEED_RE = re.compile(r"_s(\d+)\.jsonl$")


def file_seed(path: Path) -> int | None:
    match = SEED_RE.search(path.name)
    return int(match.group(1)) if match else None


def game_rows(path: Path, file_idx: int, stride: int):
    """Extract feature, target, and meta arrays for one logged game."""
    features, won, final_vp, margin = [], [], [], []
    turn, hero_seat, decision_i = [], [], []
    game_record = None
    for step in replay_steps(path):
        if step.decision is None:
            continue
        game_record = step.game_record
        if step.decision.i % stride != 0:
            continue
        colors = step.game.state.colors
        seat = random.Random(
            f"{game_record.seed}:{step.decision.i}:viewpoint"
        ).randrange(len(colors))
        hero = colors[seat]
        row = create_sample_vector(step.game, hero)
        row += [
            float(seat),
            float(step.decision.turn),
            float(step.decision.actor == Player(hero.value)),
        ]
        vps = game_record.outcome.final_vps
        hero_vp = vps[Player(hero.value)]
        best_other = max(v for p, v in vps.items() if p != Player(hero.value))
        features.append(row)
        won.append(game_record.outcome.winner == Player(hero.value))
        final_vp.append(min(hero_vp, VP_CAP))
        margin.append(hero_vp - best_other)
        turn.append(step.decision.turn)
        hero_seat.append(seat)
        decision_i.append(step.decision.i)
    return {
        "X": np.asarray(features, dtype=np.float32),
        "won": np.asarray(won, dtype=bool),
        "final_vp": np.asarray(final_vp, dtype=np.int8),
        "margin": np.asarray(margin, dtype=np.int8),
        "turn": np.asarray(turn, dtype=np.int16),
        "hero_seat": np.asarray(hero_seat, dtype=np.int8),
        "decision_i": np.asarray(decision_i, dtype=np.int32),
        "seed": np.full(len(features), game_record.seed, dtype=np.int32),
        "file_idx": np.full(len(features), file_idx, dtype=np.int32),
        "truncated": game_record.outcome.truncated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="data/games")
    parser.add_argument("--out", default="experiments/tier1_value_exit/corpus")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    require_fixed_hashseed()

    files = sorted(Path(args.games).glob("*.jsonl"))
    legacy = [f for f in files if (file_seed(f) or 0) < EVAL_SEED_LIMIT]
    overlap = [f for f in files if file_seed(f) in R3PAIRS_SEEDS]
    kept = [f for f in files if f not in set(legacy) | set(overlap)]
    print(
        f"scanned {len(files)} files: kept {len(kept)}, excluded "
        f"{len(legacy)} legacy (<{EVAL_SEED_LIMIT}), "
        f"{len(overlap)} r3pairs-overlap [10150-10449]"
    )

    jobs = [(path, idx, args.stride) for idx, path in enumerate(kept)]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.starmap(game_rows, jobs, chunksize=8)
    else:
        results = [game_rows(*job) for job in jobs]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = get_feature_ordering(4) + EXTRA_FEATURES
    truncated = sum(r["truncated"] for r in results)
    stats = {}
    for split in ("train", "val"):
        part = [
            r
            for r in results
            if (r["seed"][0] % args.val_every == 0) == (split == "val")
        ]
        arrays = {
            key: np.concatenate([r[key] for r in part])
            for key in part[0]
            if key != "truncated"
        }
        np.savez(out_dir / f"{split}.npz", **arrays)
        stats[split] = {
            "games": len(part),
            "rows": len(arrays["won"]),
            "win_rate": round(float(arrays["won"].mean()), 4),
        }
        print(
            f"{split}: {stats[split]['games']} games -> "
            f"{stats[split]['rows']} rows | "
            f"win rate {stats[split]['win_rate']:.3f}"
        )
    print(f"truncated games: {truncated} (kept, won=False)")

    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "games_dir": args.games,
        "stride": args.stride,
        "val_every": args.val_every,
        "excluded_legacy": len(legacy),
        "excluded_r3pairs_overlap": len(overlap),
        "git_commit": git,
        "feature_names": feature_names,
        "splits": stats,
        "files": [f.name for f in kept],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir}/{{train,val}}.npz + manifest.json ({len(feature_names)} cols)")


if __name__ == "__main__":
    main()
