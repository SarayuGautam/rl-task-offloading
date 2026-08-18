# =============================================================================
# environment/simulation.py
# Month 2 update: network quality now changes WITHIN each episode (mobility).
# New state dimensions: velocity_bin, quality_trend_bin.
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
    VELOCITY_BINS, QUALITY_TREND_BINS,
    QUALITY_TRAIN_MIN, QUALITY_TRAIN_MAX,
)
from src.environment.task_generator import Task
from src.environment.edge_server import EdgeServer
from src.environment.cloud_server import CloudServer
from src.environment.network_model import local_cost, edge_cost, cloud_cost


class Simulation:
    """
    Discrete-event simulation of a three-tier MEC system.

    Month 2 changes:
    - network_quality now changes every N tasks within an episode
      (simulates device moving through areas of varying signal strength)
    - velocity added as a simulation parameter - higher velocity = more
      volatile quality changes (vehicle vs pedestrian vs stationary)
    - State now includes velocity_bin and quality_trend_bin
    - use_mobility=False gives Month 1 behaviour for backward compatibility
    """

    def __init__(
        self,
        agent=None,
        seed: int = RANDOM_SEED,
        log_path: Optional[str] = None,
        network_quality: Optional[float] = None,
        velocity: float = 0.0,
        use_mobility: bool = False,
        w_latency: float = W_LATENCY,
        w_energy: float = W_ENERGY,
    ):
        self.agent        = agent
        self.seed         = seed
        self.log_path     = log_path
        self._fixed_quality = network_quality
        self.velocity     = velocity          # 0.0=stationary, 1.0=fast vehicle
        self.use_mobility = use_mobility
        # Reward weights - overridable so the Pareto sweep can vary the
        # latency/energy preference without editing config.py.
        self.w_latency    = w_latency
        self.w_energy     = w_energy

        self.completed_tasks: List[Task] = []
        self._episode_reward = 0.0

    def run(self, duration: float = 500.0) -> List[Task]:
        self.completed_tasks = []
        self._episode_reward = 0.0

        rng = np.random.default_rng(self.seed)

        if self._fixed_quality is not None:
            net_quality = self._fixed_quality
        else:
            # Sample the FULL quality range so that every network-quality bin,
            # including the poor-signal bin (quality < 0.4), is reachable during
            # training. See the note in config.py: the previous range (0.5, 1.0)
            # left half the state space unvisited.
            net_quality = float(rng.uniform(QUALITY_TRAIN_MIN, QUALITY_TRAIN_MAX))

        env   = simpy.Environment()
        edge  = EdgeServer(env)
        cloud = CloudServer(env)

        # Shared mutable state for network quality (changes during episode)
        state_container = {"quality": net_quality, "prev_quality": net_quality}

        env.process(self._arrival_loop(env, edge, cloud, rng, state_container))
        env.run(until=duration)

        if self.log_path:
            self._write_csv()

        return self.completed_tasks

    @property
    def total_reward(self) -> float:
        return self._episode_reward

    def _arrival_loop(self, env, edge, cloud, rng, sc):
        task_id = 0
        while True:
            inter_arrival = rng.exponential(1.0 / TASK_ARRIVAL_RATE)
            yield env.timeout(inter_arrival)
            task_id += 1

            # Mobility: update quality every 20 tasks if use_mobility=True
            if self.use_mobility and task_id % 20 == 0:
                sc["prev_quality"] = sc["quality"]
                volatility = 0.03 + 0.07 * self.velocity  # faster = more change
                change = rng.normal(0, volatility)
                sc["quality"] = float(np.clip(sc["quality"] + change, 0.1, 1.0))

            task = Task(
                task_id=task_id,
                arrival_time=env.now,
                size_bits=float(rng.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX)),
                complexity=float(rng.uniform(TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX)),
            )
            env.process(self._handle_task(env, task, edge, cloud, rng, sc))

    def _handle_task(self, env, task: Task, edge: EdgeServer, cloud: CloudServer, rng, sc):
        state = self._observe(task, edge, sc)
        action = self.agent.act(state) if self.agent else ACTION_EDGE
        task.action_taken = action

        if action == ACTION_LOCAL:
            latency, energy = local_cost(task.size_bits, task.complexity)
            yield env.timeout(latency)
        elif action == ACTION_EDGE:
            queue_wait = yield env.process(edge.process(task))
            latency, energy = edge_cost(task.size_bits, task.complexity,
                                        sc["quality"], queue_wait)
        else:
            yield env.process(cloud.process(task))
            latency, energy = cloud_cost(task.size_bits, task.complexity, sc["quality"])

        task.latency     = latency
        task.energy      = energy
        task.finish_time = env.now

        reward = -(self.w_latency * latency + self.w_energy * energy)
        self._episode_reward += reward

        if self.agent and hasattr(self.agent, 'learn'):
            next_state = self._observe(task, edge, sc)
            self.agent.learn(state, action, reward, next_state)

        self.completed_tasks.append(task)
        yield env.timeout(0)

    def _observe(self, task: Task, edge: EdgeServer, sc: dict) -> tuple:
        """
        Month 1 state: (queue_bin, size_bin, net_bin)
        Month 2 state: (queue_bin, size_bin, net_bin, velocity_bin, trend_bin)
        """
        queue_bin = int(np.digitize(edge.queue_length, QUEUE_BINS))
        size_bin  = int(np.digitize(task.size_bits, TASK_SIZE_BINS))
        net_bin   = int(np.digitize(sc["quality"], NETWORK_QUALITY_BINS))

        if self.use_mobility:
            vel_bin   = int(np.digitize(self.velocity, VELOCITY_BINS))
            trend     = sc["quality"] - sc["prev_quality"]
            trend_bin = int(np.digitize(trend, QUALITY_TREND_BINS))
            return (queue_bin, size_bin, net_bin, vel_bin, trend_bin)

        return (queue_bin, size_bin, net_bin)

    def _write_csv(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task_id','arrival_time','size_bits','complexity',
                        'action','latency','energy'])
            for t in self.completed_tasks:
                w.writerow([t.task_id, round(t.arrival_time,4),
                            round(t.size_bits,1), round(t.complexity,1),
                            ACTION_NAMES[t.action_taken],
                            round(t.latency,6), round(t.energy,8)])
