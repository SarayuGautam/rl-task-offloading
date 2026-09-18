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
    The cost is purely the propagation delay (CLOUD_PROPAGATION_DELAY = 80 ms),
    which is added analytically in network_model.cloud_cost rather than
    advanced here in simulation time.
    """

    def __init__(self, env: simpy.Environment):
        self.env = env

    def process(self, task) -> float:
        """
        SimPy process: simulate propagation + computation.
        Returns queue_wait = 0 always (cloud never queues).
        """
        proc_time = compute_time(task.complexity, CLOUD_CPU_SPEED)
        # propagation delay (round trip is counted in network_model.cloud_cost)
        yield self.env.timeout(proc_time)
        return 0.0   # no queue wait ever
