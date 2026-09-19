"""Report figures, drawn from results/summary.json and results/raw/*.json.

Each figure is written as a 300-dpi PNG (embedded in the .docx report) and as a
vector PDF. The palette is Okabe-Ito, which is safe for colour-blind readers;
each figure keeps the aspect ratio of the report figure it replaces.
"""
import json
import os

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
         "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 9, "figure.dpi": 100, "savefig.dpi": 300,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
SEEDS = [42, 123, 456, 789, 999]


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches=None)
    plt.close(fig)
    print("  saved", name)


def fig_latency(summary, fig_dir):
    """Fig 4.1: mean latency per strategy over 5 seeds, 95% CI error bars, log axis."""
    per = summary["table_4_3_five_seeds_1000s"]["per_strategy"]
    order = ["Greedy Heuristic", "Q-learning", "Always-Cloud", "Always-Edge", "Random", "Always-Local"]
    colors = {"Q-learning": OKABE["vermillion"], "Greedy Heuristic": OKABE["blue"]}
    means = [per[n]["latency"]["mean"] for n in order]
    errs = [per[n]["latency"]["ci95_half"] for n in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(order))[::-1]
    ax.barh(y, means, xerr=errs, color=[colors.get(n, OKABE["grey"]) for n in order],
            edgecolor="black", linewidth=0.5, height=0.6, capsize=3, error_kw={"lw": 0.8})
    for yi, m in zip(y, means):
        ax.text(m * 1.06, yi, f"{m:.4f} s", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 3.0)
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Average task latency (s, log scale)")
    ax.set_title("Average task latency by strategy (mean of 5 seeds, 95% CI)")
    ax.grid(axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir, "fig4_1_latency_by_strategy")


def fig_learning_curve(summary, raw_dir, fig_dir, window=50):
    """Fig 4.2: 50-episode moving average of episode reward, mean +/- 1 SD over 5 seeds."""
    curves = []
    for s in SEEDS:
        r = np.asarray(json.load(open(os.path.join(raw_dir, f"seed_{s}.json")))["episode_rewards"])
        curves.append(np.convolve(r, np.ones(window) / window, mode="valid"))
    c = np.vstack(curves)
    x = np.arange(window, window + c.shape[1])
    m, sd = c.mean(axis=0), c.std(axis=0, ddof=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for row in c:
        ax.plot(x, row, color=OKABE["grey"], lw=0.5, alpha=0.6)
    ax.fill_between(x, m - sd, m + sd, color=OKABE["blue"], alpha=0.25, label="mean ± 1 SD (5 seeds)")
    ax.plot(x, m, color=OKABE["blue"], lw=2, label=f"mean of {window}-episode moving averages")
    ax.plot([], [], color=OKABE["grey"], lw=0.8, label="individual seeds")
    conv = summary["table_4_3_five_seeds_1000s"]["convergence_episode"]
    ax.axvline(np.median(conv), color=OKABE["vermillion"], ls="--", lw=1.2,
               label=f"median convergence episode ({int(np.median(conv)):,})")
    eps = np.maximum(0.05, 0.998 ** np.arange(1, x[-1] + 1))
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, x[-1] + 1), eps, color=OKABE["green"], ls=":", lw=1.2)
    ax2.set_ylabel("exploration rate ε (dotted)", color=OKABE["green"])
    ax2.tick_params(axis="y", colors=OKABE["green"])
    ax2.set_ylim(0, 1.05)
    ax2.spines["right"].set_visible(True)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward (sum over ~600 tasks)")
    ax.set_title("Q-Learning training curve (5 seeds)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", bbox_to_anchor=(0.97, 0.45))
    fig.tight_layout()
    _save(fig, fig_dir, "fig4_2_learning_curve")


def fig_policy(raw_dir, fig_dir, seed=42, min_visits=30):
    """Fig 4.3: greedy action per state from the frozen seed-42 Q-table."""
    qt = json.load(open(os.path.join(raw_dir, f"seed_{seed}.json")))["qtable"]
    table = {tuple(s): (np.array(q), np.array(v)) for s, q, v in zip(qt["states"], qt["q"], qt["visits"])}
    names = ["Local", "Edge", "Cloud"]
    colors = [OKABE["orange"], OKABE["sky"], OKABE["green"]]
    untrained = "#e6e6e6"
    cmap = ListedColormap([untrained] + colors)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for k, net in enumerate(range(3)):
        grid = np.zeros((3, 4), dtype=int)
        labels = [["--"] * 4 for _ in range(3)]
        for qb in range(4):
            for sb in range(3):
                if (qb, sb, net) not in table:
                    continue
                q, v = table[(qb, sb, net)]
                tried, flag = v >= min_visits, ""
                if not tried.any():
                    tried, flag = v > 0, "?"
                if not tried.any():
                    continue
                a = int(np.argmax(np.where(tried, q, -np.inf)))
                grid[sb, qb] = a + 1
                labels[sb][qb] = names[a][0] + flag
        ax = axes[k]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["0", "1–2", "3–5", "≥6"])
        ax.set_yticks(range(3))
        ax.set_yticklabels(["small", "medium", "large"])
        ax.set_xlabel("Tasks at edge server (queue + in service)")
        if k == 0:
            ax.set_ylabel("Task size bin")
        ax.set_title(f"Network quality: {['poor (<0.4)', 'ok (0.4–0.75)', 'good (≥0.75)'][net]}")
        for qb in range(4):
            for sb in range(3):
                ax.text(qb, sb, labels[sb][qb], ha="center", va="center", fontsize=10,
                        fontweight="bold", color="black" if grid[sb, qb] else "#666666")
        for spine in ax.spines.values():
            spine.set_visible(True)
    handles = [Patch(facecolor=c, edgecolor="black", label=n) for c, n in zip(colors, names)]
    handles.append(Patch(facecolor=untrained, edgecolor="black", label="never visited (--)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Learned greedy policy per state (seed 42; '?' = fewer than 30 visits per action)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, fig_dir, "fig4_3_policy_heatmap")


def fig_pareto(summary, fig_dir):
    """Fig 4.4: latency-energy operating points for 7 weightings (seed 42)."""
    rows = summary["pareto"]["rows"]
    lat = np.array([r["avg_latency"] for r in rows])
    eng = np.array([r["avg_energy"] for r in rows])
    wl = np.array([r["w_latency"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.3, 5))
    sc = ax.scatter(lat, eng * 1e4, c=wl, cmap="cividis", vmin=0, vmax=1, s=70,
                    edgecolors="black", linewidths=0.6, zorder=3)
    groups = {}
    for x, y, w in zip(lat, eng * 1e4, wl):          # agents that learned the same policy coincide
        groups.setdefault(round(float(x), 9), []).append((x, y, w))
    for k, pts in enumerate(sorted(groups.values(), key=lambda g: g[0][0])):
        x, y, _ = pts[0]
        ws = ", ".join(f"{w:.2f}" for _, _, w in sorted(pts, key=lambda t: -t[2]))
        above = k % 2 == 0
        ax.annotate(("$w_{lat}$ = " if len(pts) > 1 else "") + ws, (x, y), textcoords="offset points",
                    xytext=(0, 14 if above else -20), ha="left" if k == 0 else "center", fontsize=8)
    e0 = eng.mean() * 1e4
    ax.set_ylim(e0 * 0.9, e0 * 1.1)
    ax.axhline(e0, color=OKABE["grey"], lw=0.8, ls="--", zorder=1)
    spread = eng.max() - eng.min()
    ax.text(0.02, 0.06, f"energy identical at all 7 weightings: {e0:.4f} × 10⁻⁴ J\n"
            + ("(max − min = 0 J: identical to machine precision)" if spread == 0 else f"(max − min = {spread:.1e} J)"),
            transform=ax.transAxes, fontsize=9)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("latency weight $w_{lat}$ (point labels)")
    ax.set_xlabel("Average latency (s)")
    ax.set_ylabel("Average device energy (× 10⁻⁴ J)")
    ax.set_title("Latency–energy operating points of Q-learning agents\ntrained under 7 cost weightings (seed 42)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir, "fig4_4_pareto")


def make_all(summary, raw_dir, fig_dir):
    fig_latency(summary, fig_dir)
    fig_learning_curve(summary, raw_dir, fig_dir)
    fig_policy(raw_dir, fig_dir)
    if "pareto" in summary:
        fig_pareto(summary, fig_dir)
