"""Unit tests for the tabular Q-learning agent (run: python -m pytest -q)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agent.q_learning_agent import QLearningAgent  # noqa: E402

# Tiny deterministic continuing MDP: 2 states x 3 actions, all rewards negative
# (as in the offloading problem). NEXT[s][a] = s', REW[s][a] = r.
NEXT = {0: [0, 1, 0], 1: [0, 1, 0]}
REW = {0: [-1.0, -2.0, -1.5], 1: [-0.5, -0.2, -3.0]}
GAMMA = 0.9


def q_star(gamma=GAMMA, iters=5_000):
    """Optimal action values by value iteration on the Bellman optimality equation
    Q*(s,a) = r(s,a) + gamma * max_a' Q*(s',a')."""
    q = np.zeros((2, 3))
    for _ in range(iters):
        q = np.array([[REW[s][a] + gamma * q[NEXT[s][a]].max() for a in range(3)] for s in range(2)])
    return q


def test_converges_to_q_star_off_policy():
    """Q-learning with a uniformly random behaviour policy (off-policy) and a
    constant step size converges to Q* on a deterministic MDP."""
    agent = QLearningAgent(alpha=0.15, gamma=GAMMA, seed=0)
    rng = np.random.default_rng(0)
    s = 0
    for _ in range(30_000):
        a = int(rng.integers(0, 3))
        s2 = NEXT[s][a]
        agent.learn((s,), a, REW[s][a], (s2,))
        s = s2
    learned = np.array([agent.q_table[(s,)] for s in range(2)])
    assert np.allclose(learned, q_star(), atol=1e-3), (learned, q_star())
    # greedy policy equals the optimal policy
    assert [agent._greedy((s,)) for s in range(2)] == list(q_star().argmax(axis=1))


def test_single_update_matches_formula():
    """Q <- Q + alpha * (r + gamma * max_a' Q(s',a') - Q), max over tried actions of s'."""
    agent = QLearningAgent(alpha=0.5, gamma=0.9, seed=0)
    agent.learn(("B",), 1, -2.0, ("C",))        # B: Q[1] = 0.5 * -2 = -1.0, tried
    agent.learn(("A",), 0, -1.0, ("B",))        # target = -1 + 0.9 * (-1.0) = -1.9
    assert np.isclose(agent.q_table[("A",)][0], 0.5 * -1.9)
    # successor never visited -> bootstrap value 0 (its initialisation)
    agent.learn(("D",), 2, -1.0, ("Z",))
    assert np.isclose(agent.q_table[("D",)][2], -0.5)


def test_greedy_ignores_untried_actions():
    """Zero-initialised (optimistic) untried actions must not win the greedy arg-max."""
    agent = QLearningAgent(alpha=0.5, gamma=0.9, seed=0)
    agent.learn(("S",), 2, -1.0, ("S",))
    assert agent.q_table[("S",)][0] == 0.0 and agent.q_table[("S",)][2] < 0
    assert agent._greedy(("S",)) == 2
    assert not agent.is_state_trained(("never",))


def test_freeze_stops_learning_and_exploration():
    agent = QLearningAgent(alpha=0.5, gamma=0.9, seed=0)
    agent.learn(("S",), 1, -1.0, ("S",))
    before_q = agent.q_table[("S",)].copy()
    before_n = agent.visit_counts[("S",)].copy()
    agent.freeze()
    assert agent.epsilon == 0.0
    for _ in range(100):
        agent.learn(("S",), 1, -5.0, ("S",))
    assert np.array_equal(agent.q_table[("S",)], before_q)
    assert np.array_equal(agent.visit_counts[("S",)], before_n)
    assert all(agent.act(("S",)) == 1 for _ in range(50))


def test_epsilon_schedule():
    agent = QLearningAgent(seed=0)
    for _ in range(2_000):
        agent.end_episode()
    # 0.998 ** n reaches the 0.05 floor after ceil(ln 0.05 / ln 0.998) = 1,497 episodes
    assert agent.epsilon == 0.05
    agent2 = QLearningAgent(seed=0)
    for _ in range(1_496):
        agent2.end_episode()
    assert agent2.epsilon > 0.05
