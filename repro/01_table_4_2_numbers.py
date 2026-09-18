from src.environment.simulation import Simulation
from src.agent.q_learning_agent import QLearningAgent
from src.agent.baselines import (AlwaysLocalAgent, AlwaysEdgeAgent, AlwaysCloudAgent,
                                 RandomAgent, GreedyHeuristicAgent)
from src.evaluation.metrics import avg_latency, avg_energy, composite_cost, action_distribution
from src.config import RANDOM_SEED, SIM_DURATION, EVAL_NETWORK_QUALITY, train_sim_seed, eval_sim_seed

agent = QLearningAgent(seed=RANDOM_SEED)
for ep in range(2000):
    Simulation(agent=agent, seed=train_sim_seed(RANDOM_SEED, ep)).run(duration=100)
    agent.end_episode()
agent.epsilon = 0.0

rows = [("Q-Learning", agent), ("Greedy Heuristic", GreedyHeuristicAgent()),
        ("Always-Cloud", AlwaysCloudAgent()), ("Always-Edge", AlwaysEdgeAgent()),
        ("Random", RandomAgent()), ("Always-Local", AlwaysLocalAgent())]

eseed = eval_sim_seed(RANDOM_SEED)
for name, a in rows:
    tasks = Simulation(agent=a, seed=eseed, network_quality=EVAL_NETWORK_QUALITY).run(duration=SIM_DURATION)
    lat, eng, cost = avg_latency(tasks), avg_energy(tasks), composite_cost(tasks)
    print(f"{name:18s} n={len(tasks):6d}  lat={lat:.6f}  eng={eng:.8f}  cost={cost:.6f}  "
          f"{action_distribution(tasks)}")
