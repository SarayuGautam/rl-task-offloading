"""Run ONE seed of the 5-seed validation and append it to a checkpoint file.

Designed to fit inside a single short-lived tool call: each invocation trains
and evaluates exactly one seed (~2-3 min), writes its result into
repro/sample_outputs/table_4_3_seed_results.json, and exits. Re-running with the same seed is a
no-op if that seed is already checkpointed. `finalize_stats.py` reads the
checkpoint once all 5 are present and computes Table 4.3.
"""
import json
import os
import sys

from src.evaluation.statistical_tests import SEEDS, train_agent, run_baseline
from src.environment.simulation import Simulation
from src.agent.baselines import AlwaysEdgeAgent, AlwaysCloudAgent, GreedyHeuristicAgent
from src.evaluation.metrics import composite_cost
from src.config import EVAL_NETWORK_QUALITY, eval_sim_seed

CKPT = "repro/sample_outputs/table_4_3_seed_results.json"


def load():
    if os.path.exists(CKPT):
        return json.load(open(CKPT))
    return {}


def save(d):
    json.dump(d, open(CKPT, "w"), indent=1)


def run_one(seed: int):
    data = load()
    if str(seed) in data:
        print(f"seed {seed} already checkpointed: {data[str(seed)]}")
        return

    agent = train_agent(seed)
    tasks = Simulation(agent=agent, seed=eval_sim_seed(seed),
                       network_quality=EVAL_NETWORK_QUALITY).run(duration=1_000)
    row = {
        "Q-Learning":       composite_cost(tasks),
        "Always Cloud":     run_baseline(AlwaysCloudAgent(), seed),
        "Always Edge":      run_baseline(AlwaysEdgeAgent(), seed),
        "Greedy Heuristic": run_baseline(GreedyHeuristicAgent(), seed),
    }
    data[str(seed)] = row
    save(data)
    print(f"seed {seed} done: {row}")
    print(f"checkpointed seeds so far: {sorted(int(k) for k in data)}")


if __name__ == "__main__":
    run_one(int(sys.argv[1]))
