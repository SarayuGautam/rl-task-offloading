# =============================================================================
# environment/edge_server.py
#
# Models the edge server as a SimPy Resource - tasks queue up when busy.
# This is the key difference from cloud (which has unlimited capacity).
# =============================================================================

import simpy
from src.config import EDGE_CPU_SPEED
from src.environment.network_model import compute_time


class EdgeServer:
    """
    Simulates a nearby MEC server: ONE server, FIFO, UNBOUNDED queue.

    Uses simpy.Resource so tasks automatically queue when the server
    is busy - this creates realistic queue_wait times that the Q-learning agent
    must learn to avoid.

    NOTE (fix, Aug 2026): `capacity=1` is the number of SERVERS, not a queue
    bound - simpy.Resource queues are unbounded. This class previously
    imported EDGE_QUEUE_CAPACITY (=10) without ever using it, and the project
    report described the edge as having "a finite FIFO queue (capacity 10)".
    That was never implemented: no task is ever rejected. The import has been
    removed and the report corrected to say the queue is unbounded. The system
    is stable at the rates studied (lambda = 6/s against mu ~ 14.5/s,
    rho ~ 0.41), so an unbounded queue is not a modelling problem - but it must
    be described accurately.

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
        """
        Number of tasks in the SYSTEM: waiting plus the one in service.

        NOTE (fix, Sep 2026): this previously returned len(self.resource.queue),
        which counts only tasks WAITING and excludes the task currently being
        served. The agent's queue_bin=0 therefore conflated two states that
        imply very different costs: "server idle" (queue wait ~ 0) and "server
        busy, nobody waiting" (queue wait ~ one mean service time, ~69 ms). At
        the load studied (rho ~ 0.41) those cases occur in roughly a 59/41
        split inside the same bin, and the distinction is exactly what decides
        edge versus cloud. Counting the in-service task restores it.
        """
        return len(self.resource.queue) + self.resource.count

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
