"""Renders benchmark and trajectory charts to data/plots/."""

import argparse
import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font="Helvetica Neue", font_scale=1.05)


def wilson_interval(wins, n, z=1.96):
    center = (wins + z * z / 2) / (n + z * z)
    half = z * sqrt(wins * (n - wins) / n + z * z / 4) / (n + z * z)
    return center - half, center + half


def winrate_matrix(bench, out):
    bots = sorted(bench["totals"], key=lambda b: -bench["totals"][b]["wins"])
    df = pd.DataFrame(index=bots, columns=bots, dtype=float)
    n = bench["games_per_pair"]
    for pair in bench["pairs"]:
        a, b = pair["bots"]
        df.loc[a, b] = 100 * pair["wins"][a] / n
        df.loc[b, a] = 100 * pair["wins"][b] / n
    plt.figure(figsize=(8, 6.5))
    ax = sns.heatmap(
        df, annot=True, fmt=".0f", cmap="vlag", center=50, vmin=0, vmax=100,
        cbar_kws={"label": "win rate (%)"}, linewidths=2, linecolor="white",
    )
    ax.set_title(f"Win rate (%), row vs column, n={n}/pair")
    plt.tight_layout()
    plt.savefig(out / "winrate_matrix.png", dpi=150)
    plt.close()


def winrate_ci(bench, out):
    n_games = bench["games_per_pair"] * (len(bench["bots"]) - 1)
    rows = []
    for bot, t in bench["totals"].items():
        lo, hi = wilson_interval(t["wins"], t["games"])
        rows.append({"bot": bot, "rate": 100 * t["wins"] / t["games"],
                     "lo": 100 * lo, "hi": 100 * hi})
    df = pd.DataFrame(rows).sort_values("rate", ascending=False)
    plt.figure(figsize=(7, 5.5))
    plt.errorbar(
        df["bot"], df["rate"],
        yerr=[df["rate"] - df["lo"], df["hi"] - df["rate"]],
        fmt="o", capsize=4,
    )
    plt.ylim(0, 100)
    plt.ylabel("overall win rate (%), 95% CI")
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Round-robin win rate, {n_games} games/bot")
    plt.tight_layout()
    plt.savefig(out / "winrate_ci.png", dpi=150)
    plt.close()


def branching_factor(games_dir, out):
    rows = []
    for path in sorted(Path(games_dir).glob("*.jsonl")):
        for line in path.read_text().splitlines()[1:]:
            rec = json.loads(line)
            rows.append({"phase": rec["phase"], "choices": len(rec["legal_actions"])})
    df = pd.DataFrame(rows)
    grid = sns.displot(
        df, x="choices", col="phase", col_wrap=3, height=2.8, aspect=1.2,
        bins=range(0, 57, 2), facet_kws={"sharey": False},
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("legal actions", "decisions")
    grid.figure.suptitle(f"Branching factor per phase ({len(df)} decisions)", y=1.03)
    grid.savefig(out / "branching_factor.png", dpi=150)
    plt.close()


def turns_per_pair(bench, out):
    df = pd.DataFrame(
        {"pair": f"{a} v {b}", "turns": pair["avg_turns"]}
        for pair in bench["pairs"]
        for a, b in [pair["bots"]]
    ).sort_values("turns")
    plt.figure(figsize=(8, 5.5))
    ax = sns.barplot(df, x="turns", y="pair", color="C0")
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=3)
    plt.xlabel(f"avg turns per game, n={bench['games_per_pair']}")
    plt.ylabel(None)
    plt.title("Game length by matchup")
    plt.tight_layout()
    plt.savefig(out / "turns_per_pair.png", dpi=150)
    plt.close()


def decision_latency(profile_path, out):
    data = json.loads(Path(profile_path).read_text())
    df = pd.DataFrame(
        {"bot": r["bot"], "latency": ms}
        for r in data["results"]
        for ms, choices in r["samples_ms"]
        if choices > 1
    )
    plt.figure(figsize=(7, 4.5))
    ax = sns.boxplot(
        df, x="bot", y="latency", color="C0", whis=(5, 95), showfliers=False
    )
    ax.set_yscale("log")
    ax.set_ylabel("decision latency (ms, log)")
    ax.set_xlabel(None)
    ax.set_title(f"Decision latency vs {data['opponent']} (decisions with >1 legal action)")
    plt.tight_layout()
    plt.savefig(out / "decision_latency.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/roundrobin_s0_n100.json")
    parser.add_argument("--profile", default="data/benchmarks/profile.json")
    parser.add_argument("--games", default="data/games")
    parser.add_argument("--out", default="data/plots")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = json.loads(Path(args.benchmark).read_text())
    winrate_matrix(bench, out)
    winrate_ci(bench, out)
    branching_factor(args.games, out)
    turns_per_pair(bench, out)
    if Path(args.profile).exists():
        decision_latency(args.profile, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
