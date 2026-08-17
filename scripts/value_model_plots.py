"""Renders value-model calibration and trajectory charts."""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from plots import BOT, NONTHINKING, SFT, THINKING, paper_axes, save

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from catanatron.features import create_sample_vector
from catan_llm.determinism import require_fixed_hashseed
from catan_llm.replay import replay_steps
from catan_llm.schema import Player

PLAYER_COLORS = {"RED": "#c0392b", "BLUE": THINKING, "WHITE": BOT, "ORANGE": NONTHINKING}
TURN_BANDS = [(0, 15, "turns 0-14"), (15, 30, "turns 15-29"), (30, 1000, "turns 30+")]


def calibration(val, pred, out):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharey=True)
    for ax, (lo, hi, label) in zip(axes, TURN_BANDS):
        mask = (val["turn"] >= lo) & (val["turn"] < hi)
        p, y = pred[mask], val["won"][mask]
        edges = np.quantile(p, np.linspace(0, 1, 11))
        mids, obs = [], []
        for a, b in zip(edges, edges[1:]):
            sel = (p >= a) & (p < b if b < edges[-1] else p <= b)
            if sel.sum() < 50:
                continue
            mids.append(p[sel].mean())
            obs.append(y[sel].mean())
        ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1)
        ax.plot(mids, obs, color=THINKING, marker="o", markersize=5, lw=1.8)
        ax.set_title(f"{label} ({mask.sum()} states)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        paper_axes(ax, xlabel="predicted P(win)")
    axes[0].set_ylabel("observed win rate")
    save(out, "calibration.png")


def auc_by_turn(val, pred, out):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    centers, aucs = [], []
    for lo in range(0, 45, 5):
        mask = (val["turn"] >= lo) & (val["turn"] < (lo + 5 if lo < 40 else 1000))
        y = val["won"][mask]
        if mask.sum() < 500 or y.all() or not y.any():
            continue
        centers.append(lo + 2.5)
        aucs.append(roc_auc_score(y, pred[mask]))
    ax.plot(centers, aucs, color=THINKING, marker="o", markersize=5, lw=1.8)
    ax.axhline(0.5, color="#bbb", ls="--", lw=1)
    ax.set_ylim(0.45, 1.0)
    paper_axes(ax, xlabel="turn", ylabel="held-out AUC")
    save(out, "auc_by_turn.png")


def stability(bank_a, bank_b, rank_eval, out):
    """Same-move agreement between independent scorings: MC rollouts vs V ranking."""
    picks = defaultdict(list)  # (game_id, decision) -> 4 argmax picks, 8 rollouts each
    for path in (bank_a, bank_b):
        rewards = defaultdict(lambda: defaultdict(dict))
        for line in open(path):
            row = json.loads(line)
            rewards[row["game_id"], row["decision"]][row["move_id"]][
                row["scenario_seed"]
            ] = row["reward"]
        for key, per_move in rewards.items():
            for half in (slice(0, 8), slice(8, 16)):
                means = {}
                for move, scenarios in per_move.items():
                    ordered = [scenarios[seed] for seed in sorted(scenarios)]
                    means[move] = sum(ordered[half]) / 8
                picks[key].append(max(means, key=means.get))

    mc_pairs = [x == y for p in picks.values() for x, y in combinations(p, 2)]
    mc = sum(mc_pairs) / len(mc_pairs)
    matched = [
        row
        for row in map(json.loads, open(rank_eval))
        if (row["game_id"], row["decision"]) in picks
    ]
    vf = sum(row["top_a"] == row["top_b"] for row in matched) / len(matched)
    print(
        f"stability on {len(picks)} audited decisions: "
        f"MC {mc:.1%} (4 askings, 8 rollouts each) | V {vf:.1%}"
    )

    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    labels = ["monte carlo\n(8 rollouts)", "value function"]
    bars = ax.barh(labels, [mc * 100, vf * 100],
                   color=[BOT, THINKING], height=0.55, zorder=3)
    for bar, value in zip(bars, (mc, vf)):
        ax.text(value * 100 + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{value:.0%}", va="center", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    paper_axes(ax, xlabel="picked the same move both times")
    save(out, "stability_bars.png")


def game_features(path):
    rows, turns, seats = [], [], None
    for step in replay_steps(path):
        if step.decision is None:
            continue
        colors = step.game.state.colors
        if seats is None:
            seats = [
                f"{color.value} {step.game_record.seats[Player(color.value)]}"
                for color in colors
            ]
        for seat, color in enumerate(colors):
            row = create_sample_vector(step.game, color)
            row += [
                float(seat),
                float(step.decision.turn),
                float(step.decision.actor == Player(color.value)),
            ]
            rows.append(row)
        turns.append(step.decision.turn)
    winner = step.game_record.outcome.winner
    winner_seat = next(
        i for i, color in enumerate(step.game.state.colors)
        if Player(color.value) == winner
    )
    return np.asarray(rows, dtype=np.float32), turns, seats, winner_seat


def trajectories(model, games, out):
    fig, axes = plt.subplots(1, len(games), figsize=(3.1 * len(games), 3.2), sharey=True)
    for ax, path in zip(axes, games):
        X, turns, seats, winner_seat = game_features(path)
        pred = model.predict(X).reshape(len(turns), 4)
        for seat, label in enumerate(seats):
            won = seat == winner_seat
            ax.plot(turns, pred[:, seat], color=PLAYER_COLORS[label.split()[0]],
                    lw=2.4 if won else 1.2, label=label + " (winner)" if won else label)
        ax.set_ylim(0, 1)
        ax.set_title(f"seed {path.name.rsplit('_s', 1)[1].removesuffix('.jsonl')}", fontsize=10)
        paper_axes(ax, xlabel="turn")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("P(win)")
    save(out, "trajectories.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="experiments/tier1_value_exit/corpus")
    parser.add_argument("--model", default="experiments/tier1_value_exit/value_model/model.txt")
    parser.add_argument("--games", default="data/games")
    parser.add_argument("--out", default="experiments/tier1_value_exit/value_model")
    parser.add_argument("--audits", default="data/audits")
    parser.add_argument("--rank-eval", default="experiments/tier1_value_exit/rank_eval.jsonl")
    args = parser.parse_args()
    require_fixed_hashseed()

    model = lgb.Booster(model_file=args.model)
    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text())
    val = np.load(corpus / "val.npz")
    pred = model.predict(val["X"])
    calibration(val, pred, args.out)
    auc_by_turn(val, pred, args.out)
    audits = Path(args.audits)
    stability(audits / "flip_bank_a.jsonl", audits / "flip_bank_b.jsonl",
              args.rank_eval, args.out)

    val_files = [
        Path(args.games) / name
        for name in manifest["files"]
        if int(name.rsplit("_s", 1)[1].removesuffix(".jsonl")) % 10 == 0
    ]
    picks = [f for f in val_files if sum(1 for _ in open(f)) > 250][:3]
    trajectories(model, picks, args.out)
    print(f"-> {args.out}: calibration.png, auc_by_turn.png, stability_bars.png, trajectories.png")


if __name__ == "__main__":
    main()
