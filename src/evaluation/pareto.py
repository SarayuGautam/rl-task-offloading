# =============================================================================
# evaluation/pareto.py
#
# Pareto latency-energy trade-off curve - the contribution promised in the
# literature review (§2.7: "varying these weights and reporting the resulting
# Pareto trade-off curve").
#
# IDEA:
#   The reward is  -(w_latency * latency + w_energy * energy),  w_lat + w_eng = 1.
#   A single fixed weight encodes ONE preference. By sweeping the weight from
#   "energy is all that matters" to "latency is all that matters", training a
#   fresh agent for each, and plotting the (avg_latency, avg_energy) it achieves,
#   we trace the FRONTIER of achievable trade-offs.
#
#   Reading the curve: you cannot move to a point below-and-left of the frontier.
#   To get lower latency you must accept higher energy, and vice-versa. An
#   operator picks the weight matching their application (latency-critical car
#   vs. energy-critical sensor).
# =============================================================================

import csv
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.environment.simulation import Simulation
from src.agent.q_learning_agent import QLearningAgent
from src.evaluation.metrics import avg_latency, avg_energy
from src.config import EVAL_NETWORK_QUALITY

TRAIN_EPISODES = 1_500
EVAL_DURATION  = 2_000
# Energy weight beta swept 0 -> 1; latency weight is (1 - beta).
BETAS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]

_CSV_PATH = "experiments/results/pareto_data.csv"


def _save_csv(rows: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["w_latency", "w_energy", "avg_latency", "avg_energy"])
        w.writeheader(); w.writerows(rows)


def _load_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def train_and_eval(w_latency: float, w_energy: float, seed: int = 42) -> dict:
    """Train a Q-Learning agent under one weighting, then evaluate it greedily."""
    agent = QLearningAgent(seed=seed)
    for ep in range(TRAIN_EPISODES):
        sim = Simulation(agent=agent, seed=seed + ep,
                         w_latency=w_latency, w_energy=w_energy)
        sim.run(duration=100)
        agent.end_episode()

    agent.epsilon = 0.0  # greedy evaluation
    tasks = Simulation(agent=agent, seed=seed + 100_000,
                       network_quality=EVAL_NETWORK_QUALITY,
                       w_latency=w_latency, w_energy=w_energy).run(duration=EVAL_DURATION)
    return {"w_latency": w_latency, "w_energy": w_energy,
            "avg_latency": avg_latency(tasks), "avg_energy": avg_energy(tasks)}


def run_pareto(save_path: str = "experiments/results/charts/pareto_curve.png",
               verbose: bool = True, use_cache: bool = True) -> list:
    if use_cache:
        cached = _load_csv(_CSV_PATH)
        cached_betas = {round(1.0 - r["w_latency"], 2) for r in cached}
        if all(round(b, 2) in cached_betas for b in BETAS):
            if verbose:
                print(f"Loaded cached pareto data from {_CSV_PATH}\n")
            rows = sorted(cached, key=lambda r: r["w_latency"], reverse=True)
            _plot(rows, save_path)
            return rows

    rows = []
    if verbose:
        print(f"Sweeping {len(BETAS)} weightings (each trains {TRAIN_EPISODES} episodes)...\n")
    for beta in BETAS:
        w_lat, w_eng = 1.0 - beta, beta
        r = train_and_eval(w_lat, w_eng)
        rows.append(r)
        if verbose:
            print(f"  w_latency={w_lat:.2f} w_energy={w_eng:.2f}  ->  "
                  f"latency={r['avg_latency']:.5f}s  energy={r['avg_energy']:.6f}J")

    _save_csv(rows, _CSV_PATH)
    _plot(rows, save_path)
    return rows


def _plot(rows: list, save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    lat  = [r["avg_latency"]  for r in rows]
    eng  = [r["avg_energy"]   for r in rows]
    wlat = [r["w_latency"]    for r in rows]

    cmap  = cm.RdYlBu_r
    norm  = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colors = [cmap(norm(w)) for w in wlat]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Frontier line in neutral grey, then colored scatter on top
    ax.plot(lat, eng, '-', color='#cccccc', linewidth=1.5, zorder=2)
    sc = ax.scatter(lat, eng, c=wlat, cmap=cmap, norm=norm,
                    s=80, zorder=4, edgecolors='white', linewidths=0.8)

    # Annotate only the two clearly separated end-points
    endpoints = {0.00: (-12, 22, 'right'), 1.00: (-12, 22, 'right')}
    for r in rows:
        wl = round(r["w_latency"], 2)
        if wl in endpoints:
            ox, oy, ha = endpoints[wl]
            ax.annotate(
                f"$w_{{lat}}$={wl:.2f}",
                xy=(r["avg_latency"], r["avg_energy"]),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=10,
                ha=ha,
                arrowprops=dict(arrowstyle='->', color='#555555', lw=0.9),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#aaaaaa', alpha=0.95),
            )

    # Colorbar encodes w_lat for all points
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("$w_{lat}$ (latency weight)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("Average latency (s)", fontsize=12)
    ax.set_ylabel("Average energy (J)", fontsize=12)
    ax.set_title(
        "Pareto trade-off: latency vs energy\n"
        "(each point = Q-Learning agent trained under one weighting)",
        fontsize=12,
    )
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    run_pareto()
