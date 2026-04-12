# =============================================================================
# environment/simulation.py
#
# The main simulation environment.
# Ties together: task generator, edge server, cloud server, network model.
# The RL agent plugs in via the `agent` parameter — swap any agent freely.
#
# Usage:
#   sim = Simulation(agent=MyAgent())
#   results = sim.run()
# =============================================================================

import simpy
import numpy as np
import csv
import os
from typing import Optional, List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.config import (
    RANDOM_SEED, SIM_DURATION, TASK_ARRIVAL_RATE,
    TASK_SIZE_MIN, TASK_SIZE_MAX,
    TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX,
    EDGE_BANDWIDTH, EDGE_PROPAGATION_DELAY,
    CLOUD_BANDWIDTH, CLOUD_PROPAGATION_DELAY,
    DEVICE_CPU_SPEED, DEVICE_POWER,
    EDGE_CPU_SPEED, CLOUD_CPU_SPEED,
    TRANSMISSION_POWER,
    ACTION_LOCAL, ACTION_EDGE, ACTION_CLOUD, ACTION_NAMES,
    W_LATENCY, W_ENERGY,
)
from src.environment.task_generator import Task, TaskGenerator
from src.environment.edge_server import EdgeServer
from src.environment.cloud_server import CloudServer
from src.environment.network_model import local_cost, edge_cost, cloud_cost


class Simulation:
    """
    Discrete-event simulation of a three-tier MEC system.

    The agent observes the current state (queue length, network quality,
    task properties) and decides where to process each task.
    After processing, the agent receives a reward and updates itself.

    Args:
        agent          : any object with an act(state) method
                         (QLearningAgent, baseline, etc.)
        seed           : random seed for reproducibility
        log_path       : if given, write per-task CSV log here
        network_quality: 0.0–1.0, scales bandwidth (simulates poor connection)
    """

    def __init__(
        self,
        agent=None,
        seed: int = RANDOM_SEED,
        log_path: Optional[str] = None,
        network_quality: float = 1.0,
    ):
        self.agent = agent
        self.seed = seed
        self.log_path = log_path
        self.network_quality = network_quality  # 0–1 scaling factor on bandwidth

        self.completed_tasks: List[Task] = []
        self._episode_reward = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, duration: float = SIM_DURATION) -> List[Task]:
        """
        Run the simulation for `duration` seconds.
        Returns the list of completed Task objects.
        """
        self.completed_tasks = []
        self._episode_reward = 0.0

        rng = np.random.default_rng(self.seed)
        env = simpy.Environment()
        edge = EdgeServer(env)
        cloud = CloudServer(env)

        env.process(self._task_arrival_loop(env, edge, cloud, rng))
        env.run(until=duration)

        if self.log_path:
            self._write_csv()

        return self.completed_tasks

    @property
    def total_reward(self) -> float:
        return self._episode_reward

    # ------------------------------------------------------------------
    # Internal SimPy processes
    # ------------------------------------------------------------------

    def _task_arrival_loop(self, env, edge, cloud, rng):
        """Poisson arrivals — main generator loop.
        Network quality varies slowly over time (simulates signal fluctuation).
        """
        task_id = 0
        while True:
            inter_arrival = rng.exponential(1.0 / TASK_ARRIVAL_RATE)
            yield env.timeout(inter_arrival)

            # Slowly vary network quality (random walk, clamped 0.2–1.0)
            delta = rng.uniform(-0.05, 0.05)
            self.network_quality = float(np.clip(self.network_quality + delta, 0.2, 1.0))

            task_id += 1
            task = Task(
                task_id=task_id,
                arrival_time=env.now,
                size_bits=float(rng.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX)),
                complexity=float(rng.uniform(TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX)),
            )

            env.process(self._handle_task(env, task, edge, cloud, rng))

    def _handle_task(self, env, task: Task, edge: EdgeServer, cloud: CloudServer, rng):
        """
        Process a single task:
          1. Build state observation
          2. Ask agent for action
          3. Execute action (compute latency + energy)
          4. Give reward to agent
          5. Log the task
        """
        # --- Build state ---
        state = self._observe(task, edge, rng)

        # --- Agent decides ---
        if self.agent is not None:
            action = self.agent.act(state)
        else:
            action = ACTION_LOCAL   # default if no agent

        task.action_taken = action

        # --- Execute action and measure cost ---
        if action == ACTION_LOCAL:
            latency, energy = local_cost(task.size_bits, task.complexity)
            yield env.timeout(latency)

        elif action == ACTION_EDGE:
            # SimPy handles actual queue wait; formula computes full cost
            queue_wait = yield env.process(edge.process(task))
            latency, energy = edge_cost(task.size_bits, task.complexity, queue_wait)

        else:  # ACTION_CLOUD
            yield env.process(cloud.process(task))
            latency, energy = cloud_cost(task.size_bits, task.complexity)

        # --- Record results ---
        task.latency     = latency
        task.energy      = energy
        task.finish_time = env.now

        # --- Reward ---
        reward = -(W_LATENCY * task.latency + W_ENERGY * task.energy)
        self._episode_reward += reward

        # --- Update agent ---
        if self.agent is not None and hasattr(self.agent, 'learn'):
            next_state = self._observe(task, edge, rng)
            self.agent.learn(state, action, reward, next_state)

        self.completed_tasks.append(task)
        yield env.timeout(0)

    def _observe(self, task: Task, edge: EdgeServer, rng) -> tuple:
        """
        Build the discrete state tuple the agent sees.

        State = (queue_bin, task_size_bin, network_quality_bin)

        This matches Table 5.1 in the proposal.
        """
        from src.config import QUEUE_BINS, TASK_SIZE_BINS, NETWORK_QUALITY_BINS
        queue_bin   = int(np.digitize(edge.queue_length, QUEUE_BINS))
        size_bin    = int(np.digitize(task.size_bits,    TASK_SIZE_BINS))
        net_bin     = int(np.digitize(self.network_quality, NETWORK_QUALITY_BINS))
        return (queue_bin, size_bin, net_bin)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_csv(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'task_id', 'arrival_time', 'finish_time',
                'size_bits', 'complexity', 'action',
                'latency', 'energy',
            ])
            for t in self.completed_tasks:
                writer.writerow([
                    t.task_id, round(t.arrival_time, 4), round(t.finish_time, 4),
                    round(t.size_bits, 2), round(t.complexity, 2),
                    ACTION_NAMES[t.action_taken],
                    round(t.latency, 6), round(t.energy, 6),
                ])
