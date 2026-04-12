# =============================================================================
# environment/task_generator.py
#
# Generates computational tasks that arrive randomly over time.
# Tasks follow a Poisson process (most realistic model for network traffic).
#
# Module 1 connection: task size (bits) and complexity (CPU cycles) are the
# core inputs to our delay and energy formulas.
# =============================================================================

import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Generator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import (
    TASK_ARRIVAL_RATE,
    TASK_SIZE_MIN, TASK_SIZE_MAX,
    TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX,
)


@dataclass
class Task:
    """
    Represents a single computational task.

    Attributes:
        task_id      : unique identifier
        arrival_time : when the task appeared (simulation seconds)
        size_bits    : data to be transmitted (bits) — affects transmission delay
        complexity   : CPU cycles needed (millions) — affects computation time
    """
    task_id: int
    arrival_time: float
    size_bits: float
    complexity: float              # million CPU cycles
    finish_time: float = 0.0
    action_taken: int = -1         # 0=local, 1=edge, 2=cloud
    latency: float = 0.0           # seconds
    energy: float = 0.0            # Joules

    @property
    def response_time(self) -> float:
        return self.finish_time - self.arrival_time


class TaskGenerator:
    """
    Generates tasks using a Poisson arrival process.

    In a Poisson process, inter-arrival times follow an exponential
    distribution with mean = 1 / arrival_rate.  This is the standard
    model for independent random events (network packets, user requests).
    """

    def __init__(self, env: simpy.Environment, rng: np.random.Generator):
        self.env = env
        self.rng = rng
        self._task_counter = 0

    def _make_task(self) -> Task:
        """Create a new task with random size and complexity."""
        self._task_counter += 1
        size = self.rng.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX)
        complexity = self.rng.uniform(TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX)
        return Task(
            task_id=self._task_counter,
            arrival_time=self.env.now,
            size_bits=size,
            complexity=complexity,
        )

    def run(self) -> Generator:
        """
        SimPy process: yields tasks one by one.
        Inter-arrival time is exponentially distributed.
        """
        while True:
            inter_arrival = self.rng.exponential(1.0 / TASK_ARRIVAL_RATE)
            yield self.env.timeout(inter_arrival)
            task = self._make_task()
            yield self.env.process(self._dispatch(task))

    def _dispatch(self, task: Task) -> Generator:
        """
        Placeholder dispatch — the simulation.py will replace this
        with actual routing logic once the agent is connected.
        """
        # Just yield 0 time; real routing happens in simulation.py
        yield self.env.timeout(0)
