# =============================================================================
# environment/edge_server.py
#
# Models the edge server as a SimPy Resource — tasks queue up when busy.
# This is the key difference from cloud (which has unlimited capacity).
# =============================================================================

import simpy
from src.config import EDGE_QUEUE_CAPACITY, EDGE_CPU_SPEED
from environment.network_model import compute_time


class EdgeServer:
    """
    Simulates a nearby MEC server with limited capacity.

    Uses simpy.Resource so tasks automatically queue when the server
    is busy — this creates realistic queue_wait times that the RL agent
    must learn to avoid.

    Attributes:
        resource     : SimPy Resource (capacity=1 means one task at a time)
        queue_length : current number of tasks waiting
    """

    def __init__(self, env: simpy.Environment):
        self.env = env
        # capacity=1: process one task at a time, rest queue up
        self.resource = simpy.Resource(env, capacity=1)

    @property
    def queue_length(self) -> int:
        """Number of tasks currently waiting (not being processed)."""
        return len(self.resource.queue)

    @property
    def is_busy(self) -> bool:
        return self.resource.count > 0

    def process(self, task):
        """
        SimPy process: request the server, wait if busy, compute, release.
        Returns queue_wait_time in seconds via StopIteration value.
        """
        arrival_at_server = self.env.now

        with self.resource.request() as req:
            yield req
            queue_wait = self.env.now - arrival_at_server
            proc_time = compute_time(task.complexity, EDGE_CPU_SPEED)
            yield self.env.timeout(proc_time)

        return queue_wait
