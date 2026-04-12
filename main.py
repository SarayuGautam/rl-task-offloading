# =============================================================================
# main.py — CLI entry point
#
# Run any experiment from command line:
#   python main.py --agent qlearning --episodes 500
#   python main.py --agent random
#   python main.py --agent always_edge
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.environment.simulation import Simulation
from src.agent.q_learning_agent import QLearningAgent
from src.agent.baselines import AlwaysLocalAgent, AlwaysEdgeAgent, AlwaysCloudAgent, RandomAgent
from src.config import NUM_EPISODES, SIM_DURATION, RANDOM_SEED


def get_agent(name: str):
    agents = {
        "qlearning":    QLearningAgent,
        "local":        AlwaysLocalAgent,
        "edge":         AlwaysEdgeAgent,
        "cloud":        AlwaysCloudAgent,
        "random":       RandomAgent,
    }
    if name not in agents:
        print(f"Unknown agent '{name}'. Choose from: {list(agents.keys())}")
        sys.exit(1)
    return agents[name]()


def run_baseline(agent_name: str, log_dir: str):
    agent = get_agent(agent_name)
    sim = Simulation(
        agent=agent,
        seed=RANDOM_SEED,
        log_path=os.path.join(log_dir, f"{agent_name}.csv"),
    )
    tasks = sim.run(duration=SIM_DURATION)
    avg_latency = sum(t.latency for t in tasks) / len(tasks)
    avg_energy  = sum(t.energy  for t in tasks) / len(tasks)
    print(f"[{agent_name:12s}]  tasks={len(tasks)}  "
          f"avg_latency={avg_latency:.4f}s  avg_energy={avg_energy:.4f}J  "
          f"total_reward={sim.total_reward:.2f}")


def run_qlearning(episodes: int, log_dir: str):
    agent = QLearningAgent()
    print(f"Training Q-Learning agent for {episodes} episodes...")

    for ep in range(episodes):
        sim = Simulation(
            agent=agent,
            seed=RANDOM_SEED + ep,   # different seed each episode
            log_path=None,
        )
        sim.run(duration=100)        # short episodes during training
        agent.end_episode()

        if (ep + 1) % 100 == 0:
            avg_r = sum(agent.episode_rewards[-100:]) / 100
            print(f"  Episode {ep+1:4d}/{episodes}  "
                  f"avg_reward={avg_r:.2f}  epsilon={agent.epsilon:.3f}")

    # Save trained agent
    qtable_path = os.path.join(log_dir, "qtables", "qlearning_trained.npz")
    agent.save(qtable_path)

    # Final evaluation run (no exploration)
    agent.epsilon = 0.0
    sim = Simulation(
        agent=agent,
        seed=RANDOM_SEED,
        log_path=os.path.join(log_dir, "qlearning_eval.csv"),
    )
    tasks = sim.run(duration=SIM_DURATION)
    avg_latency = sum(t.latency for t in tasks) / len(tasks)
    avg_energy  = sum(t.energy  for t in tasks) / len(tasks)
    print(f"\n[qlearning eval]  tasks={len(tasks)}  "
          f"avg_latency={avg_latency:.4f}s  avg_energy={avg_energy:.4f}J  "
          f"total_reward={sim.total_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description="RL Task Offloading Experiment Runner")
    parser.add_argument("--agent",    type=str, default="qlearning",
                        help="Agent to run: qlearning | local | edge | cloud | random | all")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help="Training episodes (Q-Learning only)")
    parser.add_argument("--log_dir",  type=str, default="experiments/results",
                        help="Directory to save CSV logs")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "qtables"), exist_ok=True)

    if args.agent == "all":
        print("=== Running all baselines ===")
        for name in ["local", "edge", "cloud", "random"]:
            run_baseline(name, args.log_dir)
        print("\n=== Training Q-Learning ===")
        run_qlearning(args.episodes, args.log_dir)
    elif args.agent == "qlearning":
        run_qlearning(args.episodes, args.log_dir)
    else:
        run_baseline(args.agent, args.log_dir)


if __name__ == "__main__":
    main()
