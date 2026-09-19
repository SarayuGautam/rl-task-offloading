# =============================================================================
# agent/q_learning_agent.py
#
# Tabular Q-learning agent (Watkins, 1989; Watkins & Dayan, 1992).
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
#     α (alpha)  = learning rate  - how fast to update
#     γ (gamma)  = discount factor - how much future rewards matter
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
    Tabular Q-learning agent with epsilon-greedy exploration.

    Training vs evaluation: the simulation calls learn() after every completed
    task. Call freeze() before any evaluation run so the reported policy is the
    one learned in training: it sets epsilon = 0 and turns learn() into a no-op.

    State is a tuple of discrete bin indices - the Q-table is a dict
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
        # defaultdict initializes unseen states to zeros.
        #
        # IMPORTANT: every reward in this problem is strictly negative
        # (reward = -(w_lat*latency + w_eng*energy) < 0), so a zero
        # initialisation is OPTIMISTIC. During training that is desirable - it
        # drives systematic exploration of untried actions. At GREEDY
        # evaluation time it is a bug: an action that has never been tried in
        # a state still holds Q = 0, which beats every action that has been
        # tried (all negative), so argmax silently returns the untried action.
        # visit_counts lets act() mask those out - see _greedy().
        self.q_table: dict = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.visit_counts: dict = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int64))

        # learn() only updates the table while training is True (see freeze()).
        self.training = True

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
        return self._greedy(state)                        # exploit

    def _greedy(self, state: Tuple) -> int:
        """
        Greedy action, restricted to actions actually tried in this state.

        Because all rewards are negative, an untried action sitting at its
        zero initialisation would otherwise always win the argmax. We therefore
        take the argmax over tried actions only. If NO action has been tried in
        this state (the state was never visited during training) there is no
        learned preference to report, and we fall back to ACTION_EDGE - the
        middle tier - rather than letting argmax([0,0,0]) silently return
        ACTION_LOCAL and masquerade as a learned decision.
        """
        q      = self.q_table[state]
        tried  = self.visit_counts[state] > 0
        if not tried.any():
            return ACTION_EDGE
        masked = np.where(tried, q, -np.inf)
        return int(np.argmax(masked))

    def freeze(self):
        """
        Switch to pure greedy evaluation: epsilon = 0 and learning disabled.

        NOTE (fix, defense-final): evaluation runs used to set only
        `epsilon = 0.0`. The simulation still called learn() after every task,
        so the Q-table kept adapting to the evaluation workload during the
        "greedy" evaluation. freeze() evaluates the policy as trained.
        """
        self.epsilon = 0.0
        self.training = False

    def is_state_trained(self, state: Tuple) -> bool:
        """True if at least one action has been tried in this state."""
        return bool((self.visit_counts[state] > 0).any())

    def learn(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """
        Bellman update:
          Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

        The bootstrap term max_a' Q(s',a') is likewise taken over tried actions
        only, so an unexplored successor state contributes 0 (its optimistic
        initialisation) rather than a spurious maximum over untried actions.
        """
        if not self.training:
            return   # frozen: evaluation must not change the learned policy

        current_q = self.q_table[state][action]

        next_tried = self.visit_counts[next_state] > 0
        best_next_q = float(np.max(self.q_table[next_state][next_tried])) if next_tried.any() else 0.0

        td_target = reward + self.gamma * best_next_q
        td_error  = td_target - current_q

        self.q_table[state][action] += self.alpha * td_error
        self.visit_counts[state][action] += 1

        self.total_steps += 1
        self._current_episode_reward += reward

    # ------------------------------------------------------------------
    # Episode management (call from training loop)
    # ------------------------------------------------------------------

    def end_episode(self):
        """Record episode reward, decay epsilon, reset counter."""
        self.episode_rewards.append(self._current_episode_reward)
        self._current_episode_reward = 0.0
        # Decay once per episode - ensures enough exploration before settling
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
        visits = np.array([self.visit_counts[s] for s in states])
        np.savez(path, states=states, values=values, visits=visits,
                 epsilon=[self.epsilon])
        print(f"Q-table saved → {path}  ({len(states)} states learned)")

    def load(self, path: str):
        """Load Q-table from disk."""
        data = np.load(path, allow_pickle=True)
        has_visits = 'visits' in data.files
        for i, (state, value) in enumerate(zip(data['states'], data['values'])):
            key = tuple(state)
            self.q_table[key] = value
            if has_visits:
                self.visit_counts[key] = data['visits'][i]
            else:
                # Legacy file without visit counts: treat any non-zero Q as tried.
                self.visit_counts[key] = (np.asarray(value) != 0).astype(np.int64)
        self.epsilon = float(data['epsilon'][0])
        print(f"Q-table loaded ← {path}  ({len(self.q_table)} states)")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def action_counts(self) -> dict:
        """
        Count how often each action is the greedy choice across all states.

        Uses the same masked greedy rule as act(), so untried actions cannot
        inflate a count.
        """
        counts = {0: 0, 1: 0, 2: 0}
        for state in self.q_table.keys():
            counts[self._greedy(state)] += 1
        return counts

    def coverage(self, expected_states: int = None) -> dict:
        """
        Training-coverage diagnostic.

        Reports how much of the state-action space was actually visited. This
        is what distinguishes a genuinely learned policy from an artefact of
        the Q-table's initialisation, and it must be reported alongside any
        claim about what the policy "learned" to do in a given state.
        """
        n_states = len(self.q_table)
        visited_sa = int(sum(int((v > 0).sum()) for v in self.visit_counts.values()))
        fully = int(sum(1 for v in self.visit_counts.values() if (v > 0).all()))
        out = {
            "states_visited":        n_states,
            "state_actions_visited": visited_sa,
            "state_actions_total":   n_states * N_ACTIONS,
            "fully_explored_states": fully,
            "total_steps":           self.total_steps,
        }
        if expected_states:
            out["expected_states"] = expected_states
            out["state_coverage_pct"] = round(100.0 * n_states / expected_states, 1)
        return out
