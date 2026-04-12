# =============================================================================
# agent/baselines.py
#
# Four naive baseline strategies.
# Used to prove Q-Learning actually learned something useful.
# Module 6A in the curriculum.
# =============================================================================

import numpy as np
from src.agent.base_agent import BaseAgent
from src.config import ACTION_LOCAL, ACTION_EDGE, ACTION_CLOUD


class AlwaysLocalAgent(BaseAgent):
    """Process every task on the device. Ignores servers entirely."""
    def act(self, state): return ACTION_LOCAL
    def learn(self, *args): pass


class AlwaysEdgeAgent(BaseAgent):
    """Offload every task to edge. Ignores queue congestion."""
    def act(self, state): return ACTION_EDGE
    def learn(self, *args): pass


class AlwaysCloudAgent(BaseAgent):
    """Offload every task to cloud. Always high delay."""
    def act(self, state): return ACTION_CLOUD
    def learn(self, *args): pass


class RandomAgent(BaseAgent):
    """Pick local/edge/cloud randomly each time. No intelligence."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        return int(self.rng.integers(0, 3))

    def learn(self, *args):
        pass
