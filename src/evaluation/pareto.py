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

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.environment.simulation import Simulation
from src.agent.q_learning_agent import QLearningAgent
from src.evaluation.metrics import avg_latency, avg_energy

TRAIN_EPISODES = 1_500
EVAL_DURATION  = 2_000
# Energy weight beta swept 0 -> 1; latency weight is (1 - beta).
BETAS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]


def train_and_eval(w_latency: float, w_energy: float, seed: int = 42) -> dict:
    """Train a Q-Learning agent under one weighting, then evaluate it greedily."""
    agent = QLearningAgent(seed=seed)
    for ep in range(TRAIN_EPISODES):
        sim = Simulation(agent=agent, seed=seed + ep,
                         w_latency=w_latency, w_energy=w_energy)
        sim.run(duration=100)
        agent.end_episode()

    agent.epsilon = 0.0  # greedy evaluation
    tasks = Simulation(agent=agent, seed=seed, network_quality=0.9,
                       w_latency=w_latency, w_energy=w_energy).run(duration=EVAL_DURATION)
    return {"w_latency": w_latency, "w_energy": w_energy,
            "avg_latency": avg_latency(tasks), "avg_energy": avg_energy(tasks)}


def run_pareto(save_path: str = "experiments/results/charts/pareto_curve.png",
               verbose: bool = True) -> list:
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

    _plot(rows, save_path)
    return rows


def _plot(rows: list, save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lat = [r["avg_latency"] for r in rows]
    eng = [r["avg_energy"]  for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lat, eng, '-o', color='#e07b54', linewidth=2, markersize=7, zorder=3)
    for r in rows:
        ax.annotate(f"w_lat={r['w_latency']:.2f}",
                    (r["avg_latency"], r["avg_energy"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xlabel("Average latency (s)")
    ax.set_ylabel("Average energy (J)")
    ax.set_title("Pareto trade-off: latency vs energy\n(each point = a Q-Learning agent trained under one weighting)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    run_pareto()
