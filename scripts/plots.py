"""Renders benchmark and trajectory charts to data/plots/."""

import argparse
import json
import re
import tomllib
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter, LogLocator

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
    plt.savefig(out / "winrate_matrix.png", dpi=300)
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
    plt.savefig(out / "winrate_ci.png", dpi=300)
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
    grid.savefig(out / "branching_factor.png", dpi=300)
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
    plt.savefig(out / "turns_per_pair.png", dpi=300)
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
    plt.savefig(out / "decision_latency.png", dpi=300)
    plt.close()


def costs_scatter(costs_path, out):
    rows = []
    for line in Path(costs_path).read_text().splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) != 5 or not cells[0].isdigit():
            continue
        score = re.search(r"\d+(?:\.\d+)?", cells[2])
        cost = re.search(r"\d+(?:\.\d+)?", cells[4])
        if score and cost:
            rows.append({"model": cells[1], "score": float(score.group()),
                         "cost": float(cost.group())})
    df = pd.DataFrame(rows).sort_values(["cost", "score"], ascending=[True, False])
    on_frontier = df["score"] > df["score"].cummax().shift(fill_value=-1.0)

    plt.figure(figsize=(10, 6.5))
    frontier = df[on_frontier]
    plt.step(frontier["cost"], frontier["score"], where="post",
             color="#eb6834", lw=1, ls="--", zorder=1)
    plt.scatter(df.loc[~on_frontier, "cost"], df.loc[~on_frontier, "score"],
                color="#2a78d6", s=55, zorder=2)
    plt.scatter(frontier["cost"], frontier["score"],
                color="#eb6834", s=55, zorder=3, label="cost-efficient frontier")
    # (dx, dy, ha) for the crowded mid-band; others alternate above/below.
    nudge = {
        "MiniMax M3": (-6, 3, "right"),
        "MiMo-V2.5-Pro": (0, -13, "center"),
        "GPT-5.6 Luna": (-6, -2, "right"),
        "Muse Spark 1.1": (0, -13, "center"),
        "GLM-5.2": (2, 7, "left"),
        "Qwen3.7-Max": (-6, 3, "right"),
        "Gemini 3.6 Flash": (0, -13, "center"),
        "Gemini 3.5 Flash": (6, -2, "left"),
        "DeepSeek V4 Pro": (0, -13, "center"),
        "Qwen3.5-397B-A17B": (2, 7, "left"),
    }
    for i, r in enumerate(df.itertuples()):
        dx, dy, ha = nudge.get(r.model, (5, 4, "left") if i % 2 else (5, -10, "left"))
        plt.annotate(r.model, (r.cost, r.score), fontsize=7.5, ha=ha,
                     textcoords="offset points", xytext=(dx, dy))
    plt.xscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    plt.xlabel("estimated cost per game (log)")
    plt.ylabel("AA Intelligence Index")
    plt.title("Model intelligence vs cost per game (500K in / 50K out)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "costs_scatter.png", dpi=300)
    plt.close()


def anchors_winrate(anchors_path, out):
    data = json.loads(Path(anchors_path).read_text())
    rows = []
    for name, r in data["anchors"].items():
        lo, hi = wilson_interval(r["wins"], r["games"])
        rows.append(
            {"anchor": name, "rate": 100 * r["win_rate"], "lo": 100 * lo, "hi": 100 * hi}
        )
    df = pd.DataFrame(rows).sort_values("rate")
    n = next(iter(data["anchors"].values()))["games"]

    plt.figure(figsize=(8, 3.2))
    plt.barh(
        df["anchor"], df["rate"],
        xerr=[df["rate"] - df["lo"], df["hi"] - df["rate"]],
        color="#2a78d6", height=0.55, capsize=4,
    )
    ax = plt.gca()
    ax.axvline(25, color="#888888", ls="--", lw=1)
    ax.text(
        25, 1.02, "chance (25%)",
        transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
        ha="center", color="#666666", fontsize=8.5,
    )
    for _, r in df.reset_index().iterrows():
        plt.annotate(f"{r['rate']:.0f}%", (r["hi"], _), fontsize=9,
                     textcoords="offset points", xytext=(6, -3))
    plt.xlim(0, max(df["hi"]) + 8)
    plt.xlabel("win rate (%), 95% CI")
    plt.title(
        f"Scripted-bot anchors vs 3x {data['opponent']} (seat 0), n={n}/anchor",
        pad=22,
    )
    plt.tight_layout()
    plt.savefig(out / "anchors_winrate.png", dpi=300)
    plt.close()


INVALID_GATE = 0.05


def _mode_label(trace):
    usage = [c.get("usage") or {} for c in trace.get("calls", [])]
    completion = sum(
        u.get("completion_tokens") or u.get("output_tokens") or 0 for u in usage
    )
    visible = sum(
        len(n["message"].get("content") or "")
        for n in trace.get("nodes", [])
        if n.get("sampled")
    ) // 4
    return "thinking" if completion > max(visible, 200) * 3 else "non-thinking"


# Endpoint aliases that serve the same underlying model.
MODEL_ALIASES = {"deepseek-chat": "deepseek-v4-flash"}

ROUND_SEEDS = range(5)


def _load_run(path, round_dir):
    config_path = path.parent / "config.toml"
    if not config_path.exists():
        return None
    config = tomllib.loads(config_path.read_text())
    if round_dir:
        trajectory_dir = config.get("env", {}).get("trajectory_dir", "")
        if not trajectory_dir.endswith(f"/{round_dir}"):
            return None
    episodes = []
    first_party = "openrouter" not in config.get("client", {}).get("base_url", "")
    for line in path.read_text().splitlines():
        episode = json.loads(line)
        if not episode.get("ok") or not episode.get("traces"):
            continue
        trace = episode["traces"][0]
        metrics = trace.get("metrics", {})
        if "vp_margin" not in metrics or metrics.get("invalid_rate", 1.0) > INVALID_GATE:
            continue
        model = trace["agent"]["model"].split("/")[-1].removesuffix(":nitro").lower()
        model = MODEL_ALIASES.get(model, model)
        episodes.append(
            {
                "label": f"{model} ({_mode_label(trace)})",
                "seed": trace["info"]["catan"]["seed"],
                "margin": metrics["vp_margin"] * 10,
                "first_party": first_party,
                "mtime": path.stat().st_mtime,
                "id": path.parent.name[:8],
            }
        )
    return episodes


def models_vs_anchors(outputs_dir, anchors_path, out, round_dir=None):
    # Per model+mode+seed: first-party beats OpenRouter, then latest wins.
    chosen = {}
    for path in Path(outputs_dir).glob("*/*/traces.jsonl"):
        for episode in _load_run(path, round_dir) or []:
            key = (episode["label"], episode["seed"])
            best = chosen.get(key)
            if best and (best["first_party"], best["mtime"]) >= (
                episode["first_party"], episode["mtime"]
            ):
                continue
            if best:
                print(
                    f"models_vs_anchors: {key[0]} seed {key[1]}: "
                    f"run {episode['id']} supersedes {best['id']}"
                )
            chosen[key] = episode
    if not chosen:
        return
    margins = {}
    for (label, _), episode in sorted(chosen.items()):
        margins.setdefault(label, []).append(episode["margin"])
    for label, values in margins.items():
        if len(values) < len(ROUND_SEEDS):
            print(
                f"models_vs_anchors: {label}: only {len(values)} of "
                f"{len(ROUND_SEEDS)} games valid"
            )

    # Reference lines use the same seeds the models played, not all 300.
    anchors = json.loads(Path(anchors_path).read_text())["anchors"]

    def anchor_margin(name):
        rows = [g["vp_margin"] for g in anchors[name]["per_seed"] if g["seed"] in ROUND_SEEDS]
        return 10 * sum(rows) / len(rows)

    lo = anchor_margin("victory_point")
    hi = anchor_margin("alpha_beta")

    def rel(margin):
        return 100 * (margin - lo) / (hi - lo)

    labels = sorted(margins, key=lambda k: -sum(margins[k]) / len(margins[k]))
    scores = [rel(sum(margins[k]) / len(margins[k])) for k in labels]
    counts = [len(margins[k]) for k in labels]

    greens = sns.color_palette("crest", n_colors=max(len(labels), 3))[::-1]
    plt.figure(figsize=(10, 0.62 * len(labels) + 2.2))
    ax = plt.gca()
    ax.barh(range(len(labels)), scores, color=greens[: len(labels)], height=0.62)
    ax.set_yticks(range(len(labels)), labels, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    peer = rel(anchor_margin("value_function"))
    for x, name, style in [
        (100, "alpha_beta", "-"),
        (peer, "value_function", "--"),
        (0, "victory_point", ":"),
    ]:
        ax.axvline(x, color="#999999", ls=style, lw=1)
        ax.text(
            x, 1.03, name,
            transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", color="#666666", fontsize=8.5,
        )
    for i, (score, n) in enumerate(zip(scores, counts)):
        ax.annotate(
            f"{score:.0f}%  ({n} game{'s' if n > 1 else ''})",
            (score, i), fontsize=10.5, fontweight="bold", va="center",
            textcoords="offset points", xytext=(7, 0),
        )
    ax.set_xlim(min(min(scores) - 6, -5), 115)
    ax.set_xlabel("mean VP margin, % of expert anchor (0% = victory_point, 100% = alpha_beta)")
    ax.set_title("LLMs vs 3x value_function bots", pad=30, fontweight="bold")
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(out / "models_vs_anchors.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/roundrobin_s0_n100.json")
    parser.add_argument("--profile", default="data/benchmarks/profile.json")
    parser.add_argument("--games", default="data/games")
    parser.add_argument("--costs", default="costs.md")
    parser.add_argument(
        "--anchors", default="data/benchmarks/anchors_vs_value_function_n300.json"
    )
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--round", default="n5")
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
    if Path(args.costs).exists():
        costs_scatter(args.costs, out)
    if Path(args.anchors).exists():
        anchors_winrate(args.anchors, out)
        if Path(args.outputs).exists():
            models_vs_anchors(args.outputs, args.anchors, out, args.round)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
