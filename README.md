# Reinforcement Learning for Task Offloading in 3-tier Device-Edge-Cloud Systems

**EGPG 600 — Problem Assessment Project**
Sarayu Gautam, Kathmandu University, Department of CSE

---

## What this project does

Builds a Q-Learning agent that learns to decide — for each incoming task — whether to process it **locally** on the device, offload to a nearby **edge server**, or send to the distant **cloud**. The goal is to minimize both latency and energy consumption.

## Project structure

```
src/
  config.py              ← All hyperparameters in one place
  environment/
    task_generator.py    ← Poisson task arrivals
    network_model.py     ← Delay & energy formulas (Module 1 math)
    edge_server.py       ← SimPy queue-based edge server
    cloud_server.py      ← Infinite-capacity cloud server
    simulation.py        ← Main SimPy environment
  agent/
    base_agent.py        ← Abstract interface (swap agents freely)
    q_learning_agent.py  ← Core Q-Learning (Module 3)
    baselines.py         ← Always-local/edge/cloud/random
    dqn_agent.py         ← Thesis stub (PyTorch DQN — TODO)
  evaluation/
    metrics.py           ← Latency, energy, composite cost
    visualizer.py        ← Learning curves, heatmaps
    statistical_tests.py ← Confidence intervals, hypothesis tests
    queue_validation.py  ← M/M/1 validity check (simulation correctness)
    pareto.py            ← Latency-energy Pareto trade-off sweep (§2.7 contribution)
main.py                  ← CLI runner
```

## Channel model (Shannon-Hartley)

Transmission rate is **not** constant — it follows `rate = B · log₂(1 + SNR)`,
where SNR is derived from `network_quality`. A poor signal means a lower rate,
so the same task takes longer and costs more energy to offload. This is what
makes the agent's `network_quality` state causally affect the reward (and what
gives the Month 2 mobility work its effect). See `src/environment/network_model.py`.

## Quick start

```bash
# The venv was created under an old path; install with python -m pip:
./venv/bin/python -m pip install -r requirements.txt

# Run all baselines + Q-Learning (compare strategies)
./venv/bin/python main.py --agent all

# Train Q-Learning agent and save the 3 thesis charts
./venv/bin/python main.py --agent qlearning --episodes 2000 --charts

# Month 2 mobility mode (network quality changes within an episode)
./venv/bin/python main.py --agent qlearning --mobility --velocity 0.8

# 5-seed statistical validation (95% confidence intervals)
./venv/bin/python main.py --agent stats

# Validate the simulation engine against M/M/1 queuing theory
./venv/bin/python main.py --agent validate

# Pareto latency-energy trade-off curve (sweeps the reward weights)
./venv/bin/python main.py --agent pareto

# Run a single baseline
./venv/bin/python main.py --agent random
```

## Git milestones

| Tag | What's done |
|-----|-------------|
| `v0.1-simulation-env` | Simulation runs 10,000 tasks, CSV logs correct, all baselines working |
| `v0.2-qlearning-agent` | Q-Learning agent learns, reward improves over episodes, Q-table saves |
| `v1.0-project-complete` | Q-Learning beats all baselines by ≥10%, DQN scaffold ready for thesis |

## Key design decision

All agents implement `BaseAgent.act()` and `BaseAgent.learn()`. The simulation never imports a specific agent class — you can swap `QLearningAgent` → `DQNAgent` without touching `simulation.py`. This is the extensible architecture for the 15-credit thesis.
