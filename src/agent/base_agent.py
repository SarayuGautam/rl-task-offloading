# =============================================================================
# agent/base_agent.py
#
# Abstract base class for ALL agents (Q-Learning, DQN, baselines).
# Every agent MUST implement act() and learn().
#
# WHY this matters for your thesis:
#   The simulation only calls agent.act() and agent.learn().
#   You can swap QLearningAgent → DQNAgent without touching simulation.py.
#   This is the "extensible architecture" the proposal promises.
# =============================================================================

from abc import ABC, abstractmethod
from typing import Tuple, Any


class BaseAgent(ABC):
    """
    Interface contract for all task offloading agents.

    State is a tuple of discrete bin indices, e.g. (queue_bin, size_bin, net_bin).
    Action is an integer: 0=Local, 1=Edge, 2=Cloud.
    """

    @abstractmethod
    def act(self, state: Tuple) -> int:
        """
        Choose an action given the current state.

        Args:
            state : tuple of discrete state indices

        Returns:
            action : int  (ACTION_LOCAL=0, ACTION_EDGE=1, ACTION_CLOUD=2)
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """
        Update internal knowledge based on one (s, a, r, s') transition.

        For Q-Learning: update Q-table.
        For DQN: store in replay buffer and train neural network.
        For baselines: do nothing (override with `pass`).
        """
        raise NotImplementedError
