# =============================================================================
# agent/q_learning_agent.py
#
# Tabular Q-Learning agent — Module 3 of the curriculum.
#
# HOW IT WORKS:
#   - Maintains a Q-table: Q[state][action] = expected future reward
#   - Uses epsilon-greedy to balance exploration vs exploitation
#   - Updates Q-values using the Bellman equation after each task
#
# BELLMAN UPDATE:
#   Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
#
#   Where:
#     α (alpha)  = learning rate  — how fast to update
#     γ (gamma)  = discount factor — how much future rewards matter
#     r          = immediate reward
#     s'         = next state
# =============================================================================

import numpy as np
from collections import defaultdict
from typing import Tuple

from src.agent.base_agent import BaseAgent
from src.config import (
    ACTION_LOCAL, ACTION_EDGE, ACTION_CLOUD,
    LEARNING_RATE, DISCOUNT_FACTOR,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
)

N_ACTIONS = 3  # Local, Edge, Cloud


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.

    State is a tuple of discrete bin indices — the Q-table is a dict
    mapping state tuples to numpy arrays of Q-values per action.
    Using defaultdict means new states are initialized to 0 automatically.

    Args:
        alpha   : learning rate (default from config)
        gamma   : discount factor (default from config)
        epsilon : starting exploration rate (decays over time)
        seed    : for reproducible random exploration
    """

    def __init__(
        self,
        alpha: float = LEARNING_RATE,
        gamma: float = DISCOUNT_FACTOR,
        epsilon: float = EPSILON_START,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.rng = np.random.default_rng(seed)

        # Q-table: state_tuple → array of shape (N_ACTIONS,)
        # defaultdict initializes unseen states to zeros
        self.q_table: dict = defaultdict(lambda: np.zeros(N_ACTIONS))

        # Tracking
        self.total_steps = 0
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, state: Tuple) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon  → explore: pick random action
        With probability 1-epsilon → exploit: pick best known action
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))   # explore
        return int(np.argmax(self.q_table[state]))         # exploit

    def learn(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """
        Bellman update:
          Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error  = td_target - current_q

        self.q_table[state][action] += self.alpha * td_error

        self.total_steps += 1
        self._current_episode_reward += reward

    # ------------------------------------------------------------------
    # Episode management (call from training loop)
    # ------------------------------------------------------------------

    def end_episode(self):
        """Record episode reward, decay epsilon, reset counter."""
        self.episode_rewards.append(self._current_episode_reward)
        self._current_episode_reward = 0.0
        # Decay once per episode — ensures enough exploration before settling
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save Q-table to disk as a numpy .npz file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert defaultdict to regular dict for saving
        states = list(self.q_table.keys())
        values = np.array([self.q_table[s] for s in states])
        np.savez(path, states=states, values=values, epsilon=[self.epsilon])
        print(f"Q-table saved → {path}  ({len(states)} states learned)")

    def load(self, path: str):
        """Load Q-table from disk."""
        data = np.load(path, allow_pickle=True)
        for state, value in zip(data['states'], data['values']):
            self.q_table[tuple(state)] = value
        self.epsilon = float(data['epsilon'][0])
        print(f"Q-table loaded ← {path}  ({len(self.q_table)} states)")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def action_counts(self) -> dict:
        """Count how often each action is the greedy choice across all states."""
        counts = {0: 0, 1: 0, 2: 0}
        for q_vals in self.q_table.values():
            counts[int(np.argmax(q_vals))] += 1
        return counts
