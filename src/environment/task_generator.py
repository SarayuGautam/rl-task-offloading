# =============================================================================
# environment/task_generator.py
#
# Defines the Task data structure used throughout the simulation.
#
# Tasks arrive following a Poisson process (the standard model for random,
# independent network traffic). The actual arrival loop lives in
# simulation.py (_arrival_loop); this file only defines what a Task *is*.
#
# Module 1 connection: task size (bits) and complexity (CPU cycles) are the
# core inputs to our delay and energy formulas.
# =============================================================================

from dataclasses import dataclass


@dataclass
class Task:
    """
    Represents a single computational task.

    Attributes:
        task_id      : unique identifier
        arrival_time : when the task appeared (simulation seconds)
        size_bits    : data to be transmitted (bits) - affects transmission delay
        complexity   : CPU cycles needed (millions) - affects computation time
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
