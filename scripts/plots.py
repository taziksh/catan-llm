"""Renders benchmark and trajectory charts to data/plots/."""

import argparse
import json
import re
import tomllib
from math import erfc, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter, LogLocator

import scienceplots  # noqa: F401  (registers the styles)

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "xtick.top": False, "ytick.right": False,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "xtick.direction": "out", "ytick.direction": "out",
})

THINKING = "#2d5f8a"
NONTHINKING = "#eb6834"
BOT = "#b0b0b0"
SFT = "#4d8f4f"


def save(out, name):
    plt.tight_layout()
    plt.savefig(Path(out) / name, dpi=300)
    plt.savefig((Path(out) / name).with_suffix(".pdf"))
    plt.close()


def value_label(ax, x, y, text, color="#333"):
    """Places a bar-end label on whichever side keeps it inside the axes."""
    lo, hi = ax.get_xlim()
    inward = x > lo + 0.85 * (hi - lo)
    ax.annotate(
        text, (x, y), fontsize=9.5, fontweight="bold", color=color,
        va="top" if inward else "center",
        ha="right" if inward else "left",
        textcoords="offset points", xytext=(-8, -7) if inward else (8, 0),
    )


def episode_metrics(run_dir):
    """Yields (model, seed, rank, vp_margin, win) per agent trace in a run."""
    for line in open(Path(run_dir) / "traces.jsonl"):
        episode = json.loads(line)
        for trace in episode["traces"]:
            yield (
                trace["agent"]["model"],
                trace["info"]["catan"]["seed"],
                trace["metrics"]["rank"],
                trace["metrics"]["vp_margin"] * 10,
                trace["rewards"]["reward_win"] > 0,
            )


def dot_panels(entities, panels, out, name, title):
    """Dot-plot small multiples: one panel per metric, entities on shared rows."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels) + 1.4, 0.62 * len(entities) + 1.9), sharey=True)
    for ax, (label, values, fmt) in zip(axes, panels):
        for i, entity in enumerate(entities):
            vals = values[entity]
            mean = vals.mean()
            se = vals.std(ddof=1) / len(vals) ** 0.5 if len(vals) > 1 else 0
            ax.errorbar(mean, i, xerr=se or None, color=THINKING, marker="o",
                        markersize=8, capsize=3, lw=1.4, ecolor="#777")
            ax.annotate(fmt.format(mean), (mean, i), fontsize=9.5, fontweight="bold",
                        color="#333", ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 9))
        ax.set_title(label, fontsize=10.5, pad=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#eee")
        ax.grid(axis="y", visible=False)
        ax.tick_params(left=False)
        ax.margins(x=0.18, y=0.3)
    axes[0].set_yticks(range(len(entities)), entities, fontsize=10.5, fontweight="bold")
    axes[0].invert_yaxis()
    fig.suptitle(title, fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(Path(out) / name, dpi=300)
    plt.close()


def paper_axes(ax, xlabel=None, ylabel=None):
    """Applies the paper-figure conventions: open spines, y-grid only."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eee")
    ax.grid(axis="x", visible=False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def se_line(ax, x, means, ses, color, label):
    """Draws a mean line with a shaded ±1 SE band."""
    ax.plot(x, means, color=color, marker="o", markersize=5, lw=1.8, label=label)
    ax.fill_between(x, [m - s for m, s in zip(means, ses)],
                    [m + s for m, s in zip(means, ses)], color=color, alpha=0.15, lw=0)


def point_note(ax, x, y, text, color):
    """Small annotation for a single point that carries the story (e.g. wins)."""
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=color)


def eval_run(outputs_dir, name, episodes=25):
    """Returns the newest run dir for a served model with the expected episode count."""
    runs = sorted(Path(outputs_dir).glob(f"catan_v1--{name}--catan_v1_harness/*"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        traces = run / "traces.jsonl"
        if traces.exists() and sum(1 for _ in open(traces)) == episodes:
            return run
    raise FileNotFoundError(f"no {episodes}-episode run for {name} in {outputs_dir}")


def sft_checkpoint_compare(outputs_dir, out):
    """35B-vs-9B SFT checkpoint curve, computed from the n=25 eval runs."""
    checkpoints = {
        "Qwen3.6-35B-A3B": (THINKING, {
            1000: "sft-35b-1k-nt", 3333: "sft-35b-10k-step26-nt",
            6667: "sft-35b-10k-step52-nt", 10000: "sft-35b-10k-nt",
        }),
        "Qwen3.5-9B": (SFT, {
            1000: "sft-9b-1k-nt", 3333: "sft-9b-10k-step26-nt",
            6667: "sft-9b-10k-step52-nt", 10000: "sft-9b-10k-nt",
        }),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    samples = list(checkpoints["Qwen3.5-9B"][1])
    for label, (color, stages) in checkpoints.items():
        means, ses = [], []
        for name in stages.values():
            rows = list(episode_metrics(eval_run(outputs_dir, name)))
            margins = [margin for *_, margin, _ in rows]
            mean = sum(margins) / len(margins)
            means.append(mean)
            ses.append((sum((m - mean) ** 2 for m in margins) / (len(margins) - 1)) ** 0.5
                       / len(margins) ** 0.5)
        se_line(ax, samples, means, ses, color, label)
    ax.set_xscale("log")
    ax.set_xticks(samples, ["1k", "3.3k", "6.7k", "10k"])
    ax.minorticks_off()
    ax.margins(y=0.15)
    paper_axes(ax, "training samples", "VP margin vs best opponent")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    save(out, "sft_checkpoint_curve_compare.png")


def _matching_trace(traces_dir, seed, margin):
    """Finds the game record in a mixed trace dir matching a run's episode outcome."""
    best = None
    for path in Path(traces_dir).glob(f"*_s{seed}_*.jsonl"):
        game = json.loads(path.read_text().splitlines()[0])
        llm = next(c for c, n in game["seats"].items() if n == "llm")
        vps = game["outcome"]["final_vps"]
        got = vps[llm] - max(v for c, v in vps.items() if c != llm)
        if got == margin and (best is None or path.stat().st_mtime > best[0].stat().st_mtime):
            best = (path, llm)
    return best


def winner_vs_teacher(outputs_dir, traces_root, out):
    """Winner-filtered vs teacher-trained vs base at n=25: margin, trades, cities."""
    models = [
        ("winner 1k", "sft-35b-winner1k-nt", "winner1k-nt"),
        ("teacher 1k", "sft-35b-1k-nt", "tinker-1k-nt"),
        ("base", "qwen36-35b-base-nt", "n10_v2"),
    ]
    panels = ["VP margin", "trades per game", "cities per game"]
    stats = {}
    for label, name, traces_dir in models:
        per = {panel: [] for panel in panels}
        for _, seed, _, margin, _ in episode_metrics(eval_run(outputs_dir, name)):
            per["VP margin"].append(margin)
            match = _matching_trace(Path(traces_root) / traces_dir, seed, margin)
            if match is None:
                print(f"winner_vs_teacher: {name} seed {seed}: no matching trace")
                continue
            path, llm = match
            trades = cities = 0
            for line in path.read_text().splitlines()[1:]:
                d = json.loads(line)
                if d.get("type") == "decision" and d.get("actor") == llm:
                    kind = d["legal_actions"][d["chosen_action"]][0]
                    trades += kind == "MARITIME_TRADE"
                    cities += kind == "BUILD_CITY"
            per["trades per game"].append(trades)
            per["cities per game"].append(cities)
        stats[label] = per

    fig, axes = plt.subplots(1, len(panels), figsize=(8.4, 2.4), sharey=True)
    for ax, panel in zip(axes, panels):
        for i, (label, per) in enumerate(stats.items()):
            vals = per[panel]
            mean = sum(vals) / len(vals)
            se = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 / len(vals) ** 0.5
            ax.errorbar(mean, i, xerr=se, color=THINKING, marker="o",
                        markersize=6, lw=1.4, capsize=0, ecolor=THINKING)
        ax.set_title(panel, fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#eee")
        ax.grid(axis="y", visible=False)
        ax.tick_params(left=False)
        ax.margins(x=0.25, y=0.35)
    axes[0].set_yticks(range(len(stats)), list(stats), fontsize=10)
    axes[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig(Path(out) / "winner_vs_teacher_n25.png", dpi=300)
    plt.close()


def sft_scaling(rows, references, out):
    """Margin vs training-data size for the SFT models, with bot reference lines."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    stages = [stage for stage, _ in rows[next(iter(rows))]]
    for name, points in rows.items():
        color = SFT if "9B" in name else THINKING
        means = [values.mean() for _, values in points]
        ses = [values.std(ddof=1) / len(values) ** 0.5 for _, values in points]
        ax.errorbar(range(len(points)), means, yerr=ses, color=color, marker="o",
                    markersize=6, lw=2, capsize=4, label=name)
        for i, mean in enumerate(means):
            above = all(mean >= other[1][i][1].mean() for other in rows.items())
            ax.annotate(f"{mean:+.1f}", (i, mean), textcoords="offset points",
                        xytext=(10, 7 if above else -16),
                        fontsize=9.5, fontweight="bold", color=color)
    for y, name in references:
        ax.axhline(y, color="#aaa", lw=1, ls="--")
        ax.annotate(name, (len(stages) - 0.98, y), fontsize=8.5, color="#888", va="bottom")
    labels = [f"{stage}\n{len(values)} games" for stage, values in rows[next(iter(rows))]]
    ax.set_xticks(range(len(stages)), labels, fontsize=10.5)
    ax.set_ylabel("mean VP margin vs value_function bots")
    ax.set_title("SFT scaling", fontweight="bold", pad=12)
    ax.set_xlim(-0.2, len(stages) - 0.4)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    save(out, "sft_scaling.png")


def leaderboard(rows, lo, hi, out, name, bots=()):
    """Horizontal leaderboard on the expert-anchor scale with standard errors."""
    def rel(margin):
        return 100 * (margin - lo) / (hi - lo)

    entries = []
    for label, margins, color in rows:
        mean = margins.mean()
        se = margins.std(ddof=1) / len(margins) ** 0.5
        entries.append((label, rel(mean), 100 * se / (hi - lo), len(margins), color))
    for label, value in (("alpha_beta (bot)", hi), ("victory_point (bot)", lo), *bots):
        entries.append((label, rel(value), None, None, BOT))
    entries.sort(key=lambda e: -e[1])

    plt.figure(figsize=(10, 0.52 * len(entries) + 2))
    ax = plt.gca()
    for i, (label, score, se, n, color) in enumerate(entries):
        ax.barh(i, score, color=color, height=0.62, xerr=se,
                error_kw={"ecolor": "#555", "capsize": 3, "lw": 1.1})
    ax.set_yticks(range(len(entries)), [e[0] for e in entries], fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(-8, 132)
    for i, (label, score, se, n, color) in enumerate(entries):
        games = "" if n is None else f"  ({n} games)"
        value_label(ax, score + (se or 0), i, f"{score:.0f}%{games}")
    ax.set_xlabel("mean VP margin, % of expert anchor (0% = victory_point, 100% = alpha_beta)")
    ax.set_title("Catan leaderboard", pad=14, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(False)
    save(out, name)


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
    # (dx, dy, ha) for the crowded mid-band and frontier-line neighbors;
    # others alternate above/below.
    nudge = {
        "MiniMax M3": (-6, 6, "right"),
        "MiMo-V2.5-Pro": (-8, -4, "right"),
        "GPT-5.6 Luna": (-6, -2, "right"),
        "GPT-5.6 Terra": (4, -12, "left"),
        "Kimi K3": (2, 7, "left"),
        "DeepSeek V4 Flash": (0, -14, "center"),
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
    plt.xlabel("estimated cost per game (log), 500K in / 50K out")
    plt.ylabel("AA Intelligence Index")
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
            {"anchor": name, "rate": 100 * r["win_rate"], "lo": 100 * lo, "hi": 100 * hi,
             "wins": r["wins"], "games": r["games"]}
        )
    df = pd.DataFrame(rows).sort_values("rate")

    plt.figure(figsize=(7.2, 2.4))
    ax = plt.gca()
    df = df.sort_values("rate", ascending=False).reset_index(drop=True)
    for i, r in df.iterrows():
        ax.plot([r["lo"], r["hi"]], [i, i], color="#d4d4d4", lw=2.2,
                solid_capstyle="round", zorder=1)
        ax.plot(r["rate"], i, "o", color=THINKING, markersize=7.5, zorder=2)
        ax.annotate(f"{r['wins']}/{r['games']}", (r["hi"], i), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color="#999")
    ax.axvline(25, color="#bbb", ls="--", lw=1, zorder=0)
    ax.annotate("opponent parity (25%)", (25, 1.02), xycoords=("data", "axes fraction"),
                fontsize=9, color="#999", ha="center", va="bottom")
    names = {"alpha_beta": "alpha-beta", "value_function": "value function",
             "victory_point": "victory point"}
    ax.set_yticks(range(len(df)), [names.get(a, a) for a in df["anchor"]], fontsize=10)
    ax.set_ylim(len(df) - 0.5, -0.5)
    ax.set_xlim(-1.5, max(df["hi"]) + 8)
    ax.set_xlabel("win rate (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eee")
    ax.grid(axis="y", visible=False)
    ax.tick_params(left=False)
    save(out, "anchors_winrate.png")


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
                "label": model + (" (thinking)" if _mode_label(trace) == "thinking" else ""),
                "seed": trace["info"]["catan"]["seed"],
                "margin": metrics["vp_margin"] * 10,
                "win": trace["rewards"]["reward_win"] > 0,
                "first_party": first_party,
                "mtime": path.stat().st_mtime,
                "id": path.parent.name[:8],
            }
        )
    return episodes


def _round_episodes(outputs_dir, round_dir):
    """Selects the best episode per (model, seed): first-party, then newest run."""
    chosen = {}
    for path in Path(outputs_dir).glob("*/*/traces.jsonl"):
        for episode in _load_run(path, round_dir) or []:
            key = (episode["label"], episode["seed"])
            best = chosen.get(key)
            if best and (best["first_party"], best["mtime"]) >= (
                episode["first_party"], episode["mtime"]
            ):
                continue
            chosen[key] = episode
    return chosen


def thinking_tokens(outputs_dir, out, round_dirs=("n5", "n10_v2")):
    """Mean thinking-mode output tokens per decision, pooled across eval rounds."""
    completion, prompts = {}, []
    for path in Path(outputs_dir).glob("*/*/traces.jsonl"):
        config_path = path.parent / "config.toml"
        if not config_path.exists():
            continue
        config = tomllib.loads(config_path.read_text())
        trajectory_dir = config.get("env", {}).get("trajectory_dir", "")
        if trajectory_dir.rsplit("/", 1)[-1] not in round_dirs:
            continue
        for line in path.read_text().splitlines():
            episode = json.loads(line)
            if not episode.get("traces"):
                continue
            trace = episode["traces"][0]
            if _mode_label(trace) != "thinking":
                continue
            model = trace["agent"]["model"].split("/")[-1].removesuffix(":nitro").lower()
            model = MODEL_ALIASES.get(model, model)
            completion.setdefault(model, [])
            for call in trace.get("calls", []):
                usage = call.get("usage") or {}
                tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                if tokens:
                    completion[model].append(tokens)
                    prompts.append(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    rows = []
    for model, tokens in completion.items():
        if len(tokens) < 50:  # timed-out or fragmentary runs; the prose covers these
            print(f"thinking_tokens: {model}: only {len(tokens)} usable calls, skipped")
            continue
        mean = sum(tokens) / len(tokens)
        se = (sum((t - mean) ** 2 for t in tokens) / (len(tokens) - 1)) ** 0.5 / len(tokens) ** 0.5
        rows.append((model, mean, se))
    rows.sort(key=lambda r: -r[1])
    limit = 16384 - sum(p for p in prompts if p) / max(1, len([p for p in prompts if p]))

    plt.figure(figsize=(7.2, 0.34 * len(rows) + 1.4))
    ax = plt.gca()
    for i, (model, mean, se) in enumerate(rows):
        ax.errorbar(mean, i, xerr=se, color=THINKING, marker="o",
                    markersize=6, lw=1.4, capsize=0, ecolor=THINKING)
    ax.axvline(limit, color="#aaa", ls="--", lw=1)
    ax.annotate("output limit", (limit, 1.02), xycoords=("data", "axes fraction"),
                fontsize=8.5, color="#888", ha="center", va="bottom")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, limit * 1.12)
    ax.set_xlabel("mean thinking-mode output tokens per decision")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eee")
    ax.grid(axis="y", visible=False)
    ax.tick_params(left=False)
    plt.tight_layout()
    plt.savefig(Path(out) / "thinking_tokens.png", dpi=300)
    plt.close()


def prompt_v1_v2(outputs_dir, out):
    """VP margin under prompt v1 vs v2 per model; v2 pooled from its two rounds."""
    v1 = _round_episodes(outputs_dir, "n5")
    v2 = {**_round_episodes(outputs_dir, "v2_probe"), **_round_episodes(outputs_dir, "n5_v2")}
    margins = {}
    for version, chosen in (("v1", v1), ("v2", v2)):
        for (label, _), episode in sorted(chosen.items()):
            margins.setdefault(label, {}).setdefault(version, []).append(episode["margin"])
    rows = [(label, per) for label, per in margins.items() if "v1" in per and "v2" in per]
    rows.sort(key=lambda r: r[0])

    plt.figure(figsize=(7.2, 0.42 * len(rows) + 1.4))
    ax = plt.gca()
    for i, (label, per) in enumerate(rows):
        for version, offset, point_color in (("v1", -0.16, "#b0b0b0"), ("v2", 0.16, THINKING)):
            vals = per[version]
            mean = sum(vals) / len(vals)
            se = ((sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                  / len(vals) ** 0.5 if len(vals) > 1 else 0)
            ax.errorbar(mean, i + offset, xerr=se or None, color=point_color, marker="o",
                        markersize=6, lw=1.4, capsize=0, ecolor=point_color)
    ax.axvline(0, color="#aaa", lw=1)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=10)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlabel("mean VP margin vs value_function bots, v1 n=5, v2 n=1 unless noted")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=0, label=l)
               for c, l in (("#b0b0b0", "prompt v1"), (THINKING, "prompt v2"))]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.4, loc="upper right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eee")
    ax.grid(axis="y", visible=False)
    ax.tick_params(left=False)
    plt.tight_layout()
    plt.savefig(Path(out) / "prompt_v1_v2.png", dpi=300)
    plt.close()


def models_winrate(outputs_dir, out, round_dir=None, anchors_path=None,
                   sft_round="tinker-10k-nt", fallback_rounds=("n5",)):
    """Combined win-rate leaderboard: LLM round, SFT checkpoints, bot anchors."""
    bot_labels = {"alpha_beta": "alpha-beta (bot)", "value_function": "value function (bot)",
                  "victory_point": "victory point (bot)"}
    rows = []

    def add(label, wins, games, color):
        lo, hi = wilson_interval(wins, games)
        rows.append((label, 100 * wins / games, 100 * lo, 100 * hi, wins, games, color))

    for source, is_sft in ((round_dir, False), (sft_round, True),
                           *((rd, False) for rd in fallback_rounds)):
        wins = {}
        for (label, _), episode in sorted(_round_episodes(outputs_dir, source).items()):
            wins.setdefault(label, []).append(episode["win"])
        for label, results in wins.items():
            if label in {r[0] for r in rows}:
                continue
            if is_sft:
                color = SFT
            else:
                color = THINKING if "(thinking)" in label else NONTHINKING
            add(label, sum(results), len(results), color)
    if anchors_path:
        anchors = json.loads(Path(anchors_path).read_text())["anchors"]
        for name, r in anchors.items():
            add(bot_labels.get(name, name), r["wins"], r["games"], BOT)
    rows.sort(key=lambda r: -r[1])

    plt.figure(figsize=(7.2, 0.42 * len(rows) + 1.4))
    ax = plt.gca()
    for i, (label, rate, lo, hi, wins, games, color) in enumerate(rows):
        ax.errorbar(rate, i, xerr=[[rate - lo], [hi - rate]], color=color, marker="o",
                    markersize=7.5, lw=0, elinewidth=1.1, ecolor="#555", capsize=3, zorder=2)
        ax.annotate(f"{wins}/{games}", (hi, i), textcoords="offset points", xytext=(8, 0),
                    va="center", fontsize=9, color="#999")
    ax.axvline(25, color="#bbb", ls="--", lw=1, zorder=0)
    ax.annotate("opponent parity (25%)", (25, 1.02), xycoords=("data", "axes fraction"),
                fontsize=9, color="#999", ha="center", va="bottom")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-1.5, max(r[3] for r in rows) + 10)
    ax.set_xlabel("win rate (%), 95% Wilson CI")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=0, label=l)
               for c, l in ((THINKING, "thinking"), (NONTHINKING, "non-thinking"),
                            (SFT, "SFT finetune"), (BOT, "scripted bot"))]
    ax.legend(handles=handles, frameon=False, fontsize=9, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.5, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#f0f0f0")
    ax.grid(axis="y", visible=False)
    ax.tick_params(left=False)
    save(out, "models_winrate.png")


def models_vs_anchors(outputs_dir, anchors_path, out, round_dir=None):
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

    rows = []
    for label, values in margins.items():
        mean = sum(values) / len(values)
        se = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 / len(values) ** 0.5
        rows.append((label, mean, se, len(values)))
    rows.sort(key=lambda r: -r[1])

    plt.figure(figsize=(7.2, 0.42 * len(rows) + 1.6))
    ax = plt.gca()
    default_n = max(r[3] for r in rows)
    for i, (label, mean, se, n) in enumerate(rows):
        color = THINKING if "(thinking)" in label else NONTHINKING
        half = 100 * 1.96 * se / (hi - lo)
        ax.errorbar(rel(mean), i, xerr=half, color=color, marker="o", markersize=7.5,
                    lw=0, elinewidth=1.1, ecolor="#555", capsize=3, zorder=2)
        if n != default_n:
            ax.annotate(f"n={n}", (rel(mean) + half, i), textcoords="offset points",
                        xytext=(8, 0), va="center", fontsize=9, color="#999")
    for value, name in ((hi, "alpha-beta"), (anchor_margin("value_function"), "value function"),
                        (lo, "victory point")):
        ax.axvline(rel(value), color="#bbb", lw=1, ls="--", zorder=0)
        ax.annotate(name, (rel(value), 1.02), xycoords=("data", "axes fraction"),
                    fontsize=9, color="#999", ha="center", va="bottom")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(left=False)
    ax.set_xlabel("mean VP margin, % of expert anchor, 95% CI")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=0, label=l)
               for c, l in ((THINKING, "thinking"), (NONTHINKING, "non-thinking"))]
    ax.set_xlim(right=max(rel(mean) + 100 * 1.96 * se / (hi - lo)
                          for _, mean, se, _ in rows) + 10)
    ax.legend(handles=handles, frameon=False, fontsize=9, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.5, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#f0f0f0")
    ax.grid(axis="y", visible=False)
    save(out, "models_vs_anchors.png")


def _pod_run_margins(outputs_dir, model, greedy, episodes):
    """Collects per-game VP margins and wins from pod-served runs matching the cell."""
    rows = []
    for run in Path(outputs_dir).glob(f"catan_v1--{model}--catan_v1_harness/*"):
        config = run / "config.toml"
        traces = run / "traces.jsonl"
        if not config.exists() or not traces.exists():
            continue
        cfg = tomllib.loads(config.read_text())
        if "runpod" not in cfg.get("client", {}).get("base_url", ""):
            continue
        if cfg.get("num_rollouts", 1) != 1:
            continue
        temp = cfg.get("sampling", {}).get("temperature")
        if greedy != (temp == 0):
            continue
        if sum(1 for _ in open(traces)) != episodes:
            continue
        margins, wins = [], 0
        for line in open(traces):
            for trace in json.loads(line)["traces"]:
                margins.append(trace["metrics"]["vp_margin"] * 10)
                wins += trace["rewards"]["reward_win"] > 0
        rows.append((run.stat().st_mtime, margins, wins))
    if not rows:
        raise FileNotFoundError(f"no pod run for {model} greedy={greedy} n={episodes}")
    return max(rows)[1:]


def dagger_stack_compare(outputs_dir, out):
    """dagger2 vs 10k BC on the pod stack, greedy and sampled decoding."""
    cells = {
        ("10k BC", "greedy"): ("sft-9b-10k-nt", True, 25),
        ("10k BC", "temp 1.0"): ("sft-9b-10k-nt", False, 25),
        ("dagger2", "greedy"): ("sft-9b-dagger2-nt", True, 100),
        ("dagger2", "temp 1.0"): ("sft-9b-dagger2-nt", False, 25),
    }
    colors = {"10k BC": THINKING, "dagger2": SFT}
    offsets = {"10k BC": -0.13, "dagger2": 0.13}
    conditions = ["greedy", "temp 1.0"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    stats = {}
    for (model, condition), (name, greedy, episodes) in cells.items():
        margins, wins = _pod_run_margins(outputs_dir, name, greedy, episodes)
        mean = sum(margins) / len(margins)
        se = (sum((m - mean) ** 2 for m in margins) / (len(margins) - 1)) ** 0.5 / len(margins) ** 0.5
        stats[model, condition] = (mean, se)
        x = conditions.index(condition) + offsets[model]
        ax.errorbar(x, mean, yerr=se, color=colors[model], marker="o", markersize=8,
                    capsize=3, lw=1.6, label=model if condition == "greedy" else None)
        note = f"n={len(margins)}"
        if wins:
            note += f", {wins} win" + ("s" if wins > 1 else "")
        point_note(ax, x, mean + se, note, colors[model])
    (m_new, se_new), (m_old, se_old) = stats["dagger2", "greedy"], stats["10k BC", "greedy"]
    t = (m_new - m_old) / (se_new**2 + se_old**2) ** 0.5
    p = erfc(t / sqrt(2))
    ax.text(0.97, 0.97, f"greedy gap +{m_new - m_old:.1f} VP\np = {p:.1e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#333")
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_xlim(-0.5, 1.5)
    ax.margins(y=0.2)
    paper_axes(ax, ylabel="VP margin vs best opponent")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    save(out, "dagger2_stack_compare.png")


def dagger_agreement_by_type(out):
    """Teacher agreement per move type, dagger2 vs 10k BC, same stack and decoding."""
    import collections
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from catan_llm.bots import BOTS
    from teacher_agreement import replay_game

    def per_type(traj_dir):
        rates = collections.defaultdict(lambda: [0, 0])
        for path in sorted(Path(traj_dir).glob("*.jsonl")):
            for replayed, teacher_action in replay_game(path, BOTS["alpha_beta"]):
                decision = replayed.decision
                chosen = decision.legal_actions[decision.chosen_action]
                rates[chosen[0].value][0] += int(teacher_action == chosen)
                rates[chosen[0].value][1] += 1
        return rates

    old = per_type("data/eval_traces/10k-crosscheck")
    new = per_type("data/eval_traces/dagger2-verdict")
    types = [t for t in old if old[t][1] >= 20 and new.get(t, [0, 0])[1] >= 20]
    rows = sorted(types, key=lambda t: new[t][0] / new[t][1] - old[t][0] / old[t][1])
    fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(rows) + 1.6))
    for i, t in enumerate(rows):
        a = 100 * old[t][0] / old[t][1]
        b = 100 * new[t][0] / new[t][1]
        ax.plot([a, b], [i, i], color="#bbb", lw=1.4, zorder=1)
        ax.plot(a, i, "o", color=THINKING, markersize=8, zorder=2)
        ax.plot(b, i, "o", color=SFT, markersize=8, zorder=2)
        ax.annotate(f"{b - a:+.0f}", (max(a, b), i), textcoords="offset points",
                    xytext=(10, -3), fontsize=9, color="#333")
    ax.set_yticks(range(len(rows)), [t.lower() for t in rows], fontsize=10)
    ax.margins(x=0.12, y=0.08)
    paper_axes(ax, xlabel="agreement with alpha_beta (%)")
    ax.plot([], [], "o", color=THINKING, label="10k BC")
    ax.plot([], [], "o", color=SFT, label="dagger2")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    save(out, "dagger2_agreement_by_type.png")


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
    parser.add_argument(
        "--dagger",
        action="store_true",
        help="render only the dagger comparison charts (the agreement replay is slow)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dagger:
        dagger_stack_compare(args.outputs, out)
        dagger_agreement_by_type(out)
        print(f"-> {out}")
        return
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
            models_winrate(
                args.outputs, out, args.round, args.anchors,
                fallback_rounds=("n5", "fable_probe"),
            )
    if Path(args.outputs).exists():
        sft_checkpoint_compare(args.outputs, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
