"""Regenerate Table 4.2 (fixed channel) and the new mobility comparison."""
import json
from src.environment.simulation import Simulation
from src.agent.q_learning_agent import QLearningAgent
from src.agent.baselines import (AlwaysLocalAgent, AlwaysEdgeAgent, AlwaysCloudAgent,
                                 RandomAgent, GreedyHeuristicAgent)
from src.evaluation.metrics import avg_latency, avg_energy, composite_cost, action_distribution
from src.config import (RANDOM_SEED, SIM_DURATION, EVAL_NETWORK_QUALITY,
                        train_sim_seed, eval_sim_seed)

EPISODES = 2_000


def train(mobility=False, velocity=0.0):
    agent = QLearningAgent(seed=RANDOM_SEED)
    for ep in range(EPISODES):
        Simulation(agent=agent, seed=train_sim_seed(RANDOM_SEED, ep),
                   use_mobility=mobility, velocity=velocity).run(duration=100)
        agent.end_episode()
    agent.epsilon = 0.0
    return agent


def evaluate(agent, mobility=False, velocity=0.0):
    tasks = Simulation(agent=agent, seed=eval_sim_seed(RANDOM_SEED),
                       network_quality=None if mobility else EVAL_NETWORK_QUALITY,
                       use_mobility=mobility, velocity=velocity).run(duration=SIM_DURATION)
    return dict(n=len(tasks), lat=avg_latency(tasks), eng=avg_energy(tasks),
                cost=composite_cost(tasks), acts=action_distribution(tasks))


def block(title, mobility, velocity):
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")
    ql = train(mobility=mobility, velocity=velocity)
    cov = ql.coverage(expected_states=324 if mobility else 36)
    rows = [("Q-Learning", ql),
            ("Greedy Heuristic", GreedyHeuristicAgent()),
            ("Always-Cloud", AlwaysCloudAgent()),
            ("Always-Edge", AlwaysEdgeAgent()),
            ("Random", RandomAgent()),
            ("Always-Local", AlwaysLocalAgent())]
    out = {}
    print(f"  {'Strategy':<18}{'n':>7}{'latency':>10}{'energy':>12}{'cost':>10}   actions")
    for name, a in rows:
        r = evaluate(a, mobility=mobility, velocity=velocity)
        out[name] = r
        print(f"  {name:<18}{r['n']:>7}{r['lat']:>10.4f}{r['eng']:>12.6f}"
              f"{r['cost']:>10.5f}   {r['acts']}")
    q = out["Q-Learning"]["cost"]
    print()
    for ref in ("Greedy Heuristic", "Always-Cloud", "Always-Edge"):
        c = out[ref]["cost"]
        print(f"  QL vs {ref:<18}: {(c - q) / c * 100:+.2f}%")
    print(f"\n  coverage: {cov['states_visited']}/{cov.get('expected_states')} states "
          f"({cov['state_coverage_pct']}%), {cov['state_actions_visited']}/"
          f"{cov['state_actions_total']} state-actions, "
          f"{cov['total_steps']:,} steps, {cov['fully_explored_states']} fully explored")
    return out, cov


if __name__ == "__main__":
    static, cov_s = block("A. FIXED CHANNEL (quality 0.9) - replaces Table 4.2", False, 0.0)
    mob, cov_m = block("B. MOBILITY (velocity 0.8, quality drifts in-episode) - NEW", True, 0.8)
    json.dump({"static": static, "mobility": mob,
               "cov_static": cov_s, "cov_mobility": cov_m},
              open("repro/sample_outputs/results.json", "w"), indent=1, default=str)
