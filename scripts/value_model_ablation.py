"""Scores the fitted value model on games from other bot lineups."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from build_value_corpus import game_rows
from plots import NONTHINKING, SFT, THINKING, paper_axes, save

import matplotlib.pyplot as plt

from catan_llm.determinism import require_fixed_hashseed

LINEUP_COLORS = {
    "value_function": THINKING,
    "value_function_contender": SFT,
    "victory_point": NONTHINKING,
    "weighted_random": "#8e6bb8",
    "random": "#c0392b",
}


def lineup_arrays(files, stride):
    results = [game_rows(path, idx, stride) for idx, path in enumerate(files)]
    arrays = {
        key: np.concatenate([r[key] for r in results])
        for key in results[0]
        if key != "truncated"
    }
    arrays["truncated_games"] = sum(r["truncated"] for r in results)
    return arrays


def turn_aucs(won, pred, turn):
    centers, aucs = [], []
    for lo in range(0, 45, 5):
        mask = (turn >= lo) & (turn < (lo + 5 if lo < 40 else 1000))
        y = won[mask]
        if mask.sum() < 300 or y.all() or not y.any():
            continue
        centers.append(lo + 2.5)
        aucs.append(roc_auc_score(y, pred[mask]))
    return centers, aucs


def auc_summary(scores, ref_auc, out):
    scores = sorted(scores, key=lambda s: -s[1])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(scores) + 1.4))
    for i, (lineup, auc) in enumerate(scores):
        ax.plot(auc, i, "o", color=LINEUP_COLORS[lineup], markersize=7.5, zorder=2)
        ax.annotate(f"{auc:.3f}", (auc, i), textcoords="offset points",
                    xytext=(10, -3), fontsize=9, color="#333")
    ax.axvline(ref_auc, color="#bbb", ls="--", lw=1, zorder=0)
    ax.annotate("training lineup, held out", (ref_auc, 1.02),
                xycoords=("data", "axes fraction"), fontsize=9, color="#999",
                ha="center", va="bottom")
    ax.set_yticks(range(len(scores)), [s[0] for s in scores], fontsize=10)
    ax.set_ylim(len(scores) - 0.5, -0.5)
    ax.set_xlim(0.75, ref_auc + 0.03)
    ax.set_xlabel("AUC, 200 games per lineup")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eee")
    ax.grid(axis="y", visible=False)
    ax.tick_params(left=False)
    save(out, "auc_summary.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="experiments/tier1_value_exit/bot_ablation/games")
    parser.add_argument("--corpus", default="experiments/tier1_value_exit/corpus")
    parser.add_argument("--model", default="experiments/tier1_value_exit/value_model/model.txt")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--out", default="experiments/tier1_value_exit/bot_ablation")
    args = parser.parse_args()
    require_fixed_hashseed()

    model = lgb.Booster(model_file=args.model)
    lineups = defaultdict(list)
    for path in sorted(Path(args.games).glob("*.jsonl")):
        lineups[path.name.split("-")[0]].append(path)

    fig_auc, ax_auc = plt.subplots(figsize=(6.4, 4.0))
    val = np.load(Path(args.corpus) / "val.npz")
    val_pred = model.predict(val["X"])
    centers, aucs = turn_aucs(val["won"], val_pred, val["turn"])
    ax_auc.plot(centers, aucs, color="#999", ls="--", marker="o", markersize=4,
                lw=1.6, label="training lineup, held out")
    ref_auc = roc_auc_score(val["won"], val_pred)

    rows, scores = [], []
    for lineup, files in sorted(lineups.items()):
        arrays = lineup_arrays(files, args.stride)
        pred = model.predict(arrays["X"])
        won, turn = arrays["won"], arrays["turn"]
        late = turn >= 30
        rows.append((lineup, arrays, pred))
        scores.append((lineup, roc_auc_score(won, pred)))
        print(
            f"{lineup}: {len(files)} games ({arrays['truncated_games']} truncated), "
            f"{len(won)} states | win rate {won.mean():.3f} | "
            f"AUC {roc_auc_score(won, pred):.3f} | "
            f"AUC turn 30+ {roc_auc_score(won[late], pred[late]):.3f} | "
            f"Brier {brier_score_loss(won, pred):.3f}"
        )
        centers, aucs = turn_aucs(won, pred, turn)
        ax_auc.plot(centers, aucs, color=LINEUP_COLORS[lineup], marker="o",
                    markersize=4, lw=1.6, label=lineup)
    ax_auc.axhline(0.5, color="#ddd", lw=1)
    ax_auc.set_ylim(0.45, 1.0)
    paper_axes(ax_auc, xlabel="turn", ylabel="AUC")
    ax_auc.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(args.out, "auc_by_lineup.png")

    fig, axes = plt.subplots(1, len(rows), figsize=(2.4 * len(rows), 2.7), sharey=True)
    for ax, (lineup, arrays, pred) in zip(axes, rows):
        edges = np.quantile(pred, np.linspace(0, 1, 11))
        mids, obs = [], []
        for a, b in zip(edges, edges[1:]):
            sel = (pred >= a) & (pred < b if b < edges[-1] else pred <= b)
            if sel.sum() < 50:
                continue
            mids.append(pred[sel].mean())
            obs.append(arrays["won"][sel].mean())
        ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1)
        ax.plot(mids, obs, color=LINEUP_COLORS[lineup], marker="o", markersize=4, lw=1.6)
        ax.set_title(lineup, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        paper_axes(ax, xlabel="predicted P(win)")
    axes[0].set_ylabel("observed win rate")
    save(args.out, "calibration_by_lineup.png")
    auc_summary(scores, ref_auc, args.out)
    print(f"-> {args.out}: auc_by_lineup.png, calibration_by_lineup.png, auc_summary.png")


if __name__ == "__main__":
    main()
