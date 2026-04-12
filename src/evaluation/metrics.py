# =============================================================================
# evaluation/metrics.py
# Module 6B: Performance Metrics
# =============================================================================

import numpy as np
from typing import List
from src.environment.task_generator import Task
from src.config import W_LATENCY, W_ENERGY, ACTION_NAMES


def compute_metrics(tasks: List[Task]) -> dict:
    """
    Compute all performance metrics from a list of completed tasks.

    Returns a dict with:
        avg_latency  : mean task latency (seconds)
        avg_energy   : mean energy consumption (Joules)
        avg_cost     : weighted composite cost (W_LATENCY*lat + W_ENERGY*energy)
        std_latency  : standard deviation of latency
        action_dist  : fraction of each action taken (Local/Edge/Cloud)
        total_tasks  : number of completed tasks
    """
    if not tasks:
        return {}

    latencies = np.array([t.latency for t in tasks])
    energies  = np.array([t.energy  for t in tasks])
    costs     = W_LATENCY * latencies + W_ENERGY * energies

    action_counts = {0: 0, 1: 0, 2: 0}
    for t in tasks:
        action_counts[t.action_taken] += 1

    n = len(tasks)
    action_dist = {ACTION_NAMES[a]: round(action_counts[a] / n, 3) for a in action_counts}

    return {
        'avg_latency'  : float(np.mean(latencies)),
        'std_latency'  : float(np.std(latencies)),
        'avg_energy'   : float(np.mean(energies)),
        'std_energy'   : float(np.std(energies)),
        'avg_cost'     : float(np.mean(costs)),
        'std_cost'     : float(np.std(costs)),
        'action_dist'  : action_dist,
        'total_tasks'  : n,
    }


def improvement_over(ql_metrics: dict, baseline_metrics: dict, metric='avg_latency') -> float:
    """
    Percentage improvement of Q-Learning over a baseline.
    Positive = Q-Learning is better (lower latency/cost).
    """
    baseline_val = baseline_metrics[metric]
    ql_val       = ql_metrics[metric]
    if baseline_val == 0:
        return 0.0
    return (baseline_val - ql_val) / baseline_val * 100


def run_all_baselines(duration: float = 2000, seed: int = 42) -> dict:
    """
    Convenience: run all baselines + return metrics dict.
    """
    from src.environment.simulation import Simulation
    from src.agent.baselines import (AlwaysLocalAgent, AlwaysEdgeAgent,
                                      AlwaysCloudAgent, RandomAgent)
    agents = {
        'Always Local'  : AlwaysLocalAgent(),
        'Always Edge'   : AlwaysEdgeAgent(),
        'Always Cloud'  : AlwaysCloudAgent(),
        'Random'        : RandomAgent(seed),
    }
    results = {}
    for name, agent in agents.items():
        sim = Simulation(agent=agent, seed=seed)
        tasks = sim.run(duration=duration)
        results[name] = compute_metrics(tasks)
    return results


def print_comparison_table(all_results: dict, ql_key='Q-Learning'):
    """Pretty-print a comparison table to the terminal."""
    print(f"\n{'Agent':<16} {'Avg Latency':>12} {'Avg Energy':>11} {'Avg Cost':>10} {'Actions (L/E/C)':>20}")
    print('  ' + '-' * 73)
    for name, m in all_results.items():
        d = m['action_dist']
        dist_str = f"{d.get('Local',0):.0%}/{d.get('Edge',0):.0%}/{d.get('Cloud',0):.0%}"
        marker = ' ←' if name == ql_key else ''
        print(f"  {name:<14} {m['avg_latency']:>12.4f}s {m['avg_energy']:>11.6f}J "
              f"{m['avg_cost']:>10.4f} {dist_str:>20}{marker}")

    if ql_key in all_results:
        print()
        ql = all_results[ql_key]
        for name, m in all_results.items():
            if name == ql_key:
                continue
            imp_l = improvement_over(ql, m, 'avg_latency')
            imp_c = improvement_over(ql, m, 'avg_cost')
            sign_l = '+' if imp_l > 0 else ''
            sign_c = '+' if imp_c > 0 else ''
            print(f"  vs {name:<13}  latency {sign_l}{imp_l:.1f}%   cost {sign_c}{imp_c:.1f}%")
