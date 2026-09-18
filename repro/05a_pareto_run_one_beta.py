"""Run ONE beta (energy weight) of the Pareto sweep and append to a checkpoint."""
import json
import os
import sys

from src.evaluation.pareto import train_and_eval, BETAS

CKPT = "repro/sample_outputs/pareto_sweep_results.json"


def load():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {}


def save(d):
    json.dump(d, open(CKPT, "w"), indent=1)


def run_one(beta: float):
    data = load()
    key = f"{beta:.2f}"
    if key in data:
        print(f"beta={key} already checkpointed: {data[key]}")
        return
    r = train_and_eval(1.0 - beta, beta)
    data[key] = r
    save(data)
    print(f"beta={key} done: {r}")
    print(f"checkpointed betas so far: {sorted(data.keys())}")


if __name__ == "__main__":
    run_one(float(sys.argv[1]))
