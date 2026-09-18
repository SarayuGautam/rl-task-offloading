# =============================================================================
# agent/baselines.py
#
# Five baseline strategies: four naive fixed policies plus one informed
# heuristic. Used to establish how much a learned policy actually adds.
# Module 6A in the curriculum.
#
# NOTE (added Sep 2026): the four original baselines are all state-blind, so
# beating them shows only that adapting to state helps at all - not that
# LEARNING is what delivers the gain. GreedyHeuristicAgent closes that gap: it
# adapts to exactly the same state the Q-agent sees, but computes its decision
# from the known cost model instead of learning one. It is the correct
# reference point for the claim that a learned policy justifies its cost.
# =============================================================================

import numpy as np
from src.agent.base_agent import BaseAgent
from src.config import (
    ACTION_LOCAL, ACTION_EDGE, ACTION_CLOUD,
    TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX,
    EDGE_CPU_SPEED, W_LATENCY, W_ENERGY,
)
from src.environment.network_model import local_cost, edge_cost, cloud_cost

# Mean edge service time, used to turn a queue count into a wait estimate.
MEAN_SERVICE_TIME = ((TASK_COMPLEXITY_MIN + TASK_COMPLEXITY_MAX) / 2.0) / EDGE_CPU_SPEED


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


class GreedyHeuristicAgent(BaseAgent):
    """
    Myopic cost-minimising heuristic. State-aware but untrained.

    For each arriving task it evaluates the SAME cost model the reward
    function uses, once per tier, and takes the cheapest. The edge queue wait
    is estimated as (tasks in system) x (mean service time) - a rule of thumb,
    not an oracle: it does not know the arriving task's true service time, nor
    the remaining service time of the task currently being served.

    This is deliberately the strongest fixed competitor. It costs nothing to
    run, has no training phase and no memory, and any claim that Q-Learning
    earns its complexity must be made against this, not against Always-Cloud.
    """

    def __init__(self, w_latency: float = W_LATENCY, w_energy: float = W_ENERGY):
        self.w_latency = w_latency
        self.w_energy = w_energy
        self._ctx = None

    def set_context(self, task, edge, quality):
        """Called by Simulation before act(); see simulation._handle_task."""
        self._ctx = (task, edge, quality)

    def act(self, state):
        if self._ctx is None:
            return ACTION_EDGE
        task, edge, quality = self._ctx
        est_wait = edge.queue_length * MEAN_SERVICE_TIME
        options = [
            local_cost(task.size_bits, task.complexity),
            edge_cost(task.size_bits, task.complexity, quality, est_wait),
            cloud_cost(task.size_bits, task.complexity, quality),
        ]
        costs = [self.w_latency * lat + self.w_energy * eng for lat, eng in options]
        return int(np.argmin(costs))

    def learn(self, *args):
        pass
