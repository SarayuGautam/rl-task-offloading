"""Diagnostics for the revised Section 4.4 (deterministic — same config as the
saved chart run, so these numbers describe the exact same trained agent)."""
import numpy as np
from src.environment.simulation import Simulation
from src.agent.q_learning_agent import QLearningAgent
from src.config import RANDOM_SEED, train_sim_seed

EPISODES = 2_000

agent = QLearningAgent(seed=RANDOM_SEED)
for ep in range(EPISODES):
    Simulation(agent=agent, seed=train_sim_seed(RANDOM_SEED, ep)).run(duration=100)
    agent.end_episode()

# --- convergence episode -----------------------------------------------
# Same idea as the original report: 50-episode moving average, "converged"
# once it stays within 5% of the mean of the final 200 episodes for good.
r = np.array(agent.episode_rewards)
window = 50
ma = np.convolve(r, np.ones(window) / window, mode="valid")   # index i -> episode i+window
final_level = ma[-200:].mean()
tol = 0.05 * abs(final_level)
conv_idx = next(i for i in range(len(ma)) if all(abs(ma[i:] - final_level) <= tol))
conv_episode = conv_idx + window
print(f"Final 200-ep moving-avg level: {final_level:.2f}")
print(f"Converged (within 5%, never leaves band again) by episode: {conv_episode}")

# --- Q-value ranges by network-quality bin, well-trained states only ----
MIN_VISITS = 30
print("\nWell-trained states (>=30 visits per action), Q-value by action:")
by_action = {0: [], 1: [], 2: []}
n_well_trained = 0
for state, q in agent.q_table.items():
    visits = agent.visit_counts[state]
    if (visits >= MIN_VISITS).all():
        n_well_trained += 1
        for a in range(3):
            by_action[a].append(q[a])

names = {0: "Local", 1: "Edge", 2: "Cloud"}
for a in range(3):
    vals = by_action[a]
    if vals:
        print(f"  {names[a]:6s}: n={len(vals):3d}  min={min(vals):.4f}  max={max(vals):.4f}")
print(f"  ({n_well_trained} states have all three actions with >=30 visits)")

# --- coverage recap -------------------------------------------------------
cov = agent.coverage(expected_states=36)
print(f"\nCoverage: {cov}")

# --- action distribution by net-quality bin (from the learned greedy policy)
# NOTE: must use the SAME visit-threshold rule as qtable_heatmap() (prefer
# actions with >=30 visits; only fall back to "any visit" if none reach 30),
# not agent._greedy()'s plain "any visit" rule -- otherwise this printout can
# disagree with what Figure 4.3 actually draws for the sparsely-visited states.
MIN_VISITS_DISPLAY = 30
print("\nGreedy action by (queue_bin, size_bin, net_bin) -- matches Figure 4.3's own rule:")
net_names = {0: "poor", 1: "ok", 2: "good"}


def display_greedy(state):
    q = agent.q_table[state]
    vis = agent.visit_counts[state]
    tried = vis >= MIN_VISITS_DISPLAY
    low_confidence = False
    if not tried.any():
        tried = vis > 0
        low_confidence = True
    a = int(np.argmax(np.where(tried, q, -np.inf)))
    return names[a][0] + ("?" if low_confidence else "")


for net_bin in (0, 1, 2):
    row = []
    for q in range(4):
        for s in range(3):
            st = (q, s, net_bin)
            if st in agent.q_table and (agent.visit_counts[st] > 0).any():
                row.append(f"q{q}s{s}={display_greedy(st)}")
    print(f"  net={net_names[net_bin]:5s}: " + " ".join(row))
