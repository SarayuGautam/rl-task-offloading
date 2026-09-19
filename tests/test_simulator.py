"""Simulator checks against closed-form results (run: python -m pytest -q)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (TASK_SIZE_MIN, TASK_SIZE_MAX, TASK_COMPLEXITY_MIN,  # noqa: E402
                        TASK_COMPLEXITY_MAX, TASK_ARRIVAL_RATE, EDGE_CPU_SPEED,
                        CLOUD_CPU_SPEED, DEVICE_CPU_SPEED, DEVICE_POWER,
                        EDGE_PROPAGATION_DELAY, CLOUD_PROPAGATION_DELAY,
                        TRANSMISSION_POWER, EVAL_NETWORK_QUALITY, SEED_STRIDE,
                        EVAL_SEED_OFFSET, train_sim_seed, eval_sim_seed)
from src.environment.simulation import Simulation  # noqa: E402
from src.environment.network_model import shannon_rate  # noqa: E402
from src.agent.baselines import AlwaysLocalAgent, AlwaysEdgeAgent, AlwaysCloudAgent  # noqa: E402
from src.evaluation.metrics import avg_latency, avg_energy  # noqa: E402
from src.evaluation.statistical_tests import SEEDS, TRAIN_EPISODES  # noqa: E402

Q = EVAL_NETWORK_QUALITY
E_SIZE = (TASK_SIZE_MIN + TASK_SIZE_MAX) / 2
E_C = (TASK_COMPLEXITY_MIN + TASK_COMPLEXITY_MAX) / 2
E_TX = E_SIZE / shannon_rate(Q)


def _run(agent, duration=10_000, seed=eval_sim_seed(42)):
    return Simulation(agent=agent, seed=seed, network_quality=Q).run(duration=duration)


def test_shannon_endpoints():
    assert abs(shannon_rate(0.1) - 137_503) < 10          # ~137 kbps at -10 dB
    assert abs(shannon_rate(1.0) / 1e6 - 9.967) < 0.01    # ~10 Mbps at 30 dB


def test_always_local_matches_closed_form():
    tasks = _run(AlwaysLocalAgent())
    assert abs(avg_latency(tasks) / (E_C / DEVICE_CPU_SPEED) - 1) < 0.02
    assert abs(avg_energy(tasks) / (DEVICE_POWER * E_C / DEVICE_CPU_SPEED) - 1) < 0.02


def test_always_cloud_matches_closed_form():
    tasks = _run(AlwaysCloudAgent())
    expected = E_TX + CLOUD_PROPAGATION_DELAY + E_C / CLOUD_CPU_SPEED
    assert abs(avg_latency(tasks) / expected - 1) < 0.01


def test_always_edge_matches_mg1_pollaczek_khinchine():
    """The edge is an M/G/1 queue with uniform service; P-K gives the mean wait."""
    a, b = TASK_COMPLEXITY_MIN / EDGE_CPU_SPEED, TASK_COMPLEXITY_MAX / EDGE_CPU_SPEED
    es, es2 = (a + b) / 2, (a * a + a * b + b * b) / 3
    rho = TASK_ARRIVAL_RATE * es
    wq = TASK_ARRIVAL_RATE * es2 / (2 * (1 - rho))
    expected = E_TX + EDGE_PROPAGATION_DELAY + wq + es
    tasks = _run(AlwaysEdgeAgent())
    assert abs(avg_latency(tasks) / expected - 1) < 0.03


def test_offload_energy_does_not_depend_on_destination():
    """Section 4.5: device energy for a task is the same whether it goes to edge or cloud."""
    e = {t.task_id: t.energy for t in _run(AlwaysEdgeAgent(), duration=500)}
    c = {t.task_id: t.energy for t in _run(AlwaysCloudAgent(), duration=500)}
    common = set(e) & set(c)
    assert len(common) > 2_500
    assert all(e[i] == c[i] for i in common)
    assert abs(np.mean(list(e.values())) / (TRANSMISSION_POWER * E_TX) - 1) < 0.02


def test_seed_blocks_are_disjoint():
    ranges = [set(train_sim_seed(s, ep) for ep in range(TRAIN_EPISODES)) for s in SEEDS]
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            assert not ranges[i] & ranges[j]
        assert eval_sim_seed(SEEDS[i]) not in set().union(*ranges)
    assert SEED_STRIDE > EVAL_SEED_OFFSET > TRAIN_EPISODES


def test_uplink_flag_off_is_default_and_deterministic():
    t1 = [(t.task_id, t.latency) for t in _run(AlwaysEdgeAgent(), duration=200)]
    t2 = [(t.task_id, t.latency) for t in _run(AlwaysEdgeAgent(), duration=200)]
    assert t1 == t2
