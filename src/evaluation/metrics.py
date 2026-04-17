# =============================================================================
# evaluation/metrics.py — Performance metrics for Module 6B
# =============================================================================
import numpy as np
from typing import List
from src.environment.task_generator import Task


def avg_latency(tasks: List[Task]) -> float:
    return np.mean([t.latency for t in tasks])

def avg_energy(tasks: List[Task]) -> float:
    return np.mean([t.energy for t in tasks])

def composite_cost(tasks: List[Task], w_lat=0.7, w_eng=0.3) -> float:
    return w_lat * avg_latency(tasks) + w_eng * avg_energy(tasks)

def action_distribution(tasks: List[Task]) -> dict:
    from collections import Counter
    from src.config import ACTION_NAMES
    counts = Counter(t.action_taken for t in tasks)
    total = len(tasks)
    return {ACTION_NAMES[a]: round(100 * counts[a] / total, 1) for a in [0,1,2]}

def improvement_over_baseline(agent_tasks, baseline_tasks) -> float:
    """Percentage improvement in composite cost (positive = better)."""
    agent_cost    = composite_cost(agent_tasks)
    baseline_cost = composite_cost(baseline_tasks)
    return (baseline_cost - agent_cost) / baseline_cost * 100

def summary(tasks: List[Task], label: str = "") -> dict:
    return {
        "label":        label,
        "n_tasks":      len(tasks),
        "avg_latency":  round(avg_latency(tasks), 5),
        "avg_energy":   round(avg_energy(tasks), 7),
        "composite":    round(composite_cost(tasks), 5),
        "actions":      action_distribution(tasks),
    }
