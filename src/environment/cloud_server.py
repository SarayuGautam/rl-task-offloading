# =============================================================================
# environment/cloud_server.py
#
# Models a distant cloud data center - effectively unlimited compute power
# but high fixed propagation delay.
# No queue needed: cloud always accepts immediately.
# =============================================================================

import simpy
from src.config import CLOUD_CPU_SPEED, CLOUD_PROPAGATION_DELAY
from src.environment.network_model import compute_time


class CloudServer:
    """
    Simulates a remote cloud data center.

    No resource contention - unlimited parallel capacity.
    Network transmission and propagation delays are modeled by Simulation;
    this process models the cloud computation time only.
    """

    def __init__(self, env: simpy.Environment):
        self.env = env

    def process(self, task) -> float:
        """
        SimPy process: simulate cloud computation.
        Returns 0.0 for compatibility; cloud never queues.
        """
        proc_time = compute_time(task.complexity, CLOUD_CPU_SPEED)
        yield self.env.timeout(proc_time)
        return 0.0   # no queue wait ever
