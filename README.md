# Adaptive Reinforcement Learning for Task Offloading in Edge-Cloud Federated Systems

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
main.py                  ← CLI runner
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all baselines (compare strategies)
python main.py --agent all

# Train Q-Learning agent (500 episodes)
python main.py --agent qlearning --episodes 500

# Run a single baseline
python main.py --agent random
```

## Git milestones

| Tag | What's done |
|-----|-------------|
| `v0.1-simulation-env` | Simulation runs 10,000 tasks, CSV logs correct, all baselines working |
| `v0.2-qlearning-agent` | Q-Learning agent learns, reward improves over episodes, Q-table saves |
| `v1.0-project-complete` | Q-Learning beats all baselines by ≥10%, DQN scaffold ready for thesis |

## Key design decision

All agents implement `BaseAgent.act()` and `BaseAgent.learn()`. The simulation never imports a specific agent class — you can swap `QLearningAgent` → `DQNAgent` without touching `simulation.py`. This is the extensible architecture for the 15-credit thesis.
