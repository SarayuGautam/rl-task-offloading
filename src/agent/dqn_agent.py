# =============================================================================
# agent/dqn_agent.py
#
# Deep Q-Network (DQN) agent - scaffold for 15-credit thesis extension.
# Module 5D: "Preparing for Thesis Extension"
#
# STATUS: Stub only. The interface matches QLearningAgent exactly so the
# simulation.py does NOT need to change when you upgrade.
#
# TO IMPLEMENT (thesis phase):
#   1. Replace Q-table with a neural network (PyTorch)
#   2. Add experience replay buffer
#   3. Add target network (for stable training)
#   4. Implement batch training step
# =============================================================================

import numpy as np
from typing import Tuple
from src.agent.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent - THESIS PLACEHOLDER.

    Same act() / learn() interface as QLearningAgent.
    Simulation.py works without any changes.
    """

    def __init__(self, state_size: int = 3, n_actions: int = 3, seed: int = 42):
        self.state_size = state_size
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        # TODO (thesis): build neural network here
        # self.policy_net = self._build_network()
        # self.target_net = self._build_network()
        # self.replay_buffer = ReplayBuffer(capacity=10_000)
        # self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)

        print("DQNAgent: stub only - implement for thesis")

    def act(self, state: Tuple) -> int:
        # TODO: replace with neural network forward pass
        return int(self.rng.integers(0, self.n_actions))

    def learn(self, state, action, reward, next_state):
        # TODO: store in replay buffer and call _train_step()
        pass

    # ------------------------------------------------------------------
    # Thesis TODO methods
    # ------------------------------------------------------------------

    def _build_network(self):
        """Build a 3-layer MLP: state_size → 64 → 64 → n_actions."""
        # import torch.nn as nn
        # return nn.Sequential(
        #     nn.Linear(self.state_size, 64), nn.ReLU(),
        #     nn.Linear(64, 64), nn.ReLU(),
        #     nn.Linear(64, self.n_actions),
        # )
        raise NotImplementedError("Implement for thesis")

    def _train_step(self):
        """Sample a batch from replay buffer and update policy net."""
        raise NotImplementedError("Implement for thesis")
