# =============================================================================
# environment/simulation.py
#
# Main SimPy simulation — ties together task generator, servers, agent.
# The agent plugs in via the `agent` parameter.  Swap any agent freely.
#
# Usage:
#   sim = Simulation(agent=QLearningAgent())
#   tasks = sim.run(duration=500)
# =============================================================================

import simpy
import numpy as np
import csv
import os
from typing import Optional, List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.config import (
    RANDOM_SEED, TASK_ARRIVAL_RATE,
    TASK_SIZE_MIN, TASK_SIZE_MAX,
    TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX,
    ACTION_LOCAL, ACTION_EDGE, ACTION_CLOUD, ACTION_NAMES,
    W_LATENCY, W_ENERGY,
    QUEUE_BINS, TASK_SIZE_BINS, NETWORK_QUALITY_BINS,
)
from src.environment.task_generator import Task
from src.environment.edge_server import EdgeServer
from src.environment.cloud_server import CloudServer
from src.environment.network_model import local_cost, edge_cost, cloud_cost


class Simulation:
    """
    Discrete-event simulation of a three-tier MEC system.

    Each episode the network quality is randomly drawn from [0.5, 1.0]
    so the agent experiences all network quality bins during training.

    Args:
        agent          : any BaseAgent subclass
        seed           : random seed for reproducibility
        log_path       : optional CSV output path
        network_quality: fixed quality override (None = random per episode)
    """

    def __init__(
        self,
        agent=None,
        seed: int = RANDOM_SEED,
        log_path: Optional[str] = None,
        network_quality: Optional[float] = None,
    ):
        self.agent = agent
        self.seed = seed
        self.log_path = log_path
        self._fixed_quality = network_quality

        self.completed_tasks: List[Task] = []
        self._episode_reward = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, duration: float = 500.0) -> List[Task]:
        """Run simulation for `duration` seconds. Returns completed tasks."""
        self.completed_tasks = []
        self._episode_reward = 0.0

        rng = np.random.default_rng(self.seed)

        # Network quality: fixed override OR random in [0.5, 1.0]
        if self._fixed_quality is not None:
            net_quality = self._fixed_quality
        else:
            net_quality = float(rng.uniform(0.5, 1.0))

        env   = simpy.Environment()
        edge  = EdgeServer(env)
        cloud = CloudServer(env)

        env.process(self._arrival_loop(env, edge, cloud, rng, net_quality))
        env.run(until=duration)

        if self.log_path:
            self._write_csv()

        return self.completed_tasks

    @property
    def total_reward(self) -> float:
        return self._episode_reward

    # ------------------------------------------------------------------
    # SimPy processes
    # ------------------------------------------------------------------

    def _arrival_loop(self, env, edge, cloud, rng, net_quality):
        """Poisson arrivals — spawn one handler process per task."""
        task_id = 0
        while True:
            inter_arrival = rng.exponential(1.0 / TASK_ARRIVAL_RATE)
            yield env.timeout(inter_arrival)
            task_id += 1
            task = Task(
                task_id=task_id,
                arrival_time=env.now,
                size_bits=float(rng.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX)),
                complexity=float(rng.uniform(TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX)),
            )
            env.process(self._handle_task(env, task, edge, cloud, rng, net_quality))

    def _handle_task(self, env, task: Task, edge: EdgeServer, cloud: CloudServer,
                     rng, net_quality: float):
        """Process one task: observe → act → execute → reward → learn."""
        state = self._observe(task, edge, net_quality)

        # Agent picks action
        action = self.agent.act(state) if self.agent else ACTION_EDGE

        task.action_taken = action

        # Execute and compute cost
        if action == ACTION_LOCAL:
            latency, energy = local_cost(task.size_bits, task.complexity)
            yield env.timeout(latency)

        elif action == ACTION_EDGE:
            # SimPy handles real queue wait; formula gives full latency
            queue_wait = yield env.process(edge.process(task))
            latency, energy = edge_cost(task.size_bits, task.complexity, queue_wait)

        else:  # ACTION_CLOUD
            yield env.process(cloud.process(task))
            latency, energy = cloud_cost(task.size_bits, task.complexity)

        task.latency     = latency
        task.energy      = energy
        task.finish_time = env.now

        reward = -(W_LATENCY * latency + W_ENERGY * energy)
        self._episode_reward += reward

        if self.agent and hasattr(self.agent, 'learn'):
            next_state = self._observe(task, edge, net_quality)
            self.agent.learn(state, action, reward, next_state)

        self.completed_tasks.append(task)
        yield env.timeout(0)

    # ------------------------------------------------------------------
    # State builder
    # ------------------------------------------------------------------

    def _observe(self, task: Task, edge: EdgeServer, net_quality: float) -> tuple:
        """
        Discrete state = (queue_bin, task_size_bin, network_quality_bin)

        All three dimensions vary meaningfully:
          - queue_bin: 0 (empty) … 3 (saturated)
          - size_bin:  0 (small) … 2 (large)
          - net_bin:   0 (poor)  … 2 (good)
        """
        queue_bin = int(np.digitize(edge.queue_length, QUEUE_BINS))
        size_bin  = int(np.digitize(task.size_bits,    TASK_SIZE_BINS))
        net_bin   = int(np.digitize(net_quality,       NETWORK_QUALITY_BINS))
        return (queue_bin, size_bin, net_bin)

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def _write_csv(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task_id','arrival_time','size_bits','complexity',
                        'action','latency','energy'])
            for t in self.completed_tasks:
                w.writerow([
                    t.task_id, round(t.arrival_time, 4),
                    round(t.size_bits, 1), round(t.complexity, 1),
                    ACTION_NAMES[t.action_taken],
                    round(t.latency, 6), round(t.energy, 8),
                ])
