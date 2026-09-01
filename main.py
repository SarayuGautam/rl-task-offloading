# =============================================================================
# main.py - CLI entry point
#
# Run any experiment from command line:
#   python main.py --agent qlearning --episodes 500   # train Q-Learning
#   python main.py --agent qlearning --charts         # train + save 3 charts
#   python main.py --agent qlearning --mobility --velocity 0.8   # mobility mode
#   python main.py --agent random                     # one baseline
#   python main.py --agent all                        # all baselines + Q-Learning
#   python main.py --agent stats                      # 5-seed statistical validation
#   python main.py --agent validate                   # M/M/1 simulation validity check
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.environment.simulation import Simulation
from src.agent.q_learning_agent import QLearningAgent
from src.agent.baselines import AlwaysLocalAgent, AlwaysEdgeAgent, AlwaysCloudAgent, RandomAgent
from src.config import NUM_EPISODES, SIM_DURATION, RANDOM_SEED, EVAL_NETWORK_QUALITY
from src.evaluation.statistical_tests import EVAL_SEED_OFFSET


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
        network_quality=EVAL_NETWORK_QUALITY,   # fixed, documented eval channel
    )
    tasks = sim.run(duration=SIM_DURATION)
    avg_latency = sum(t.latency for t in tasks) / len(tasks)
    avg_energy  = sum(t.energy  for t in tasks) / len(tasks)
    print(f"[{agent_name:12s}]  tasks={len(tasks)}  "
          f"avg_latency={avg_latency:.4f}s  avg_energy={avg_energy:.4f}J  "
          f"total_reward={sim.total_reward:.2f}")


def run_qlearning(episodes: int, log_dir: str,
                  mobility: bool = False, velocity: float = 0.0,
                  charts: bool = False):
    agent = QLearningAgent()
    mode = f"mobility (velocity={velocity})" if mobility else "static"
    print(f"Training Q-Learning agent for {episodes} episodes [{mode}]...")

    for ep in range(episodes):
        sim = Simulation(
            agent=agent,
            seed=RANDOM_SEED + ep,   # different seed each episode
            log_path=None,
            velocity=velocity,
            use_mobility=mobility,
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

    # Training coverage - reported so that any claim about what the policy
    # "learned" in a given state can be checked against whether that state was
    # ever actually visited.
    expected = 324 if mobility else 36
    cov = agent.coverage(expected_states=expected)
    print(f"\n[coverage] states visited {cov['states_visited']}/{expected} "
          f"({cov['state_coverage_pct']}%)  |  state-actions visited "
          f"{cov['state_actions_visited']}/{cov['state_actions_total']}  |  "
          f"fully-explored states {cov['fully_explored_states']}  |  "
          f"learning steps {cov['total_steps']:,}")

    # Final evaluation run (no exploration)
    # Use EVAL_SEED_OFFSET to ensure evaluation workload is disjoint from training
    eval_seed = RANDOM_SEED + EVAL_SEED_OFFSET
    agent.epsilon = 0.0
    sim = Simulation(
        agent=agent,
        seed=eval_seed,
        log_path=os.path.join(log_dir, "qlearning_eval.csv"),
        velocity=velocity,
        use_mobility=mobility,
        network_quality=None if mobility else EVAL_NETWORK_QUALITY,
    )
    tasks = sim.run(duration=SIM_DURATION)
    avg_latency = sum(t.latency for t in tasks) / len(tasks)
    avg_energy  = sum(t.energy  for t in tasks) / len(tasks)
    print(f"\n[qlearning eval]  tasks={len(tasks)}  "
          f"avg_latency={avg_latency:.4f}s  avg_energy={avg_energy:.4f}J  "
          f"total_reward={sim.total_reward:.2f}  states_learned={len(agent.q_table)}")

    if charts:
        generate_charts(agent, log_dir)


def generate_charts(agent: QLearningAgent, log_dir: str):
    """Save the 3 thesis charts: learning curve, comparison bar, Q-table heatmap."""
    from src.evaluation.visualizer import learning_curve, comparison_bar, qtable_heatmap
    from src.evaluation.metrics import summary
    from src.evaluation.statistical_tests import EVAL_SEED_OFFSET

    chart_dir = os.path.join(log_dir, "charts")
    eval_seed = RANDOM_SEED + EVAL_SEED_OFFSET

    # 1. Learning curve from this agent's training history
    learning_curve(agent.episode_rewards, os.path.join(chart_dir, "learning_curve.png"))

    # 2. Comparison bar - re-run each baseline + the trained agent on one seed
    results = {}
    for label, a in [("Always Local", AlwaysLocalAgent()),
                     ("Always Edge", AlwaysEdgeAgent()),
                     ("Always Cloud", AlwaysCloudAgent()),
                     ("Random", RandomAgent()),
                     ("Q-Learning", agent)]:
        tasks = Simulation(agent=a, seed=eval_seed,
                           network_quality=EVAL_NETWORK_QUALITY).run(duration=SIM_DURATION)
        results[label] = summary(tasks, label)
    comparison_bar(results, "avg_latency", os.path.join(chart_dir, "comparison_latency.png"))

    # 3. Q-table heatmap (static 3-D state only)
    qtable_heatmap(dict(agent.q_table), os.path.join(chart_dir, "qtable_heatmap.png"),
                   visit_counts=dict(agent.visit_counts))


def run_stats():
    """5-seed statistical validation with 95% confidence intervals."""
    from src.evaluation.statistical_tests import run_all
    run_all()


def run_queue_validation():
    """Validate the SimPy queue engine against M/M/1 queuing theory."""
    from src.evaluation.queue_validation import run_validation
    run_validation()


def run_pareto():
    """Sweep latency/energy weights and plot the Pareto trade-off curve."""
    from src.evaluation.pareto import run_pareto as _run
    _run()


def main():
    parser = argparse.ArgumentParser(description="RL Task Offloading Experiment Runner")
    parser.add_argument("--agent",    type=str, default="qlearning",
                        help="qlearning | local | edge | cloud | random | all | stats | validate | pareto")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help="Training episodes (Q-Learning only)")
    parser.add_argument("--log_dir",  type=str, default="experiments/results",
                        help="Directory to save CSV logs")
    parser.add_argument("--mobility", action="store_true",
                        help="Enable within-episode network quality changes (Month 2)")
    parser.add_argument("--velocity", type=float, default=0.0,
                        help="Device velocity 0.0=stationary .. 1.0=fast vehicle (mobility mode)")
    parser.add_argument("--charts",   action="store_true",
                        help="Generate the 3 thesis charts after Q-Learning training")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "qtables"), exist_ok=True)

    if args.agent == "validate":
        run_queue_validation()
    elif args.agent == "pareto":
        run_pareto()
    elif args.agent == "stats":
        run_stats()
    elif args.agent == "all":
        print("=== Running all baselines ===")
        for name in ["local", "edge", "cloud", "random"]:
            run_baseline(name, args.log_dir)
        print("\n=== Training Q-Learning ===")
        run_qlearning(args.episodes, args.log_dir,
                      mobility=args.mobility, velocity=args.velocity, charts=args.charts)
    elif args.agent == "qlearning":
        run_qlearning(args.episodes, args.log_dir,
                      mobility=args.mobility, velocity=args.velocity, charts=args.charts)
    else:
        run_baseline(args.agent, args.log_dir)


if __name__ == "__main__":
    main()
