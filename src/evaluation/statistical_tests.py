# =============================================================================
# evaluation/statistical_tests.py
#
# Month 2 - Statistical validation.
# Run Q-learning across multiple seeds and compute 95% confidence intervals.
# Reports whether the measured improvement is statistically significant.
# =============================================================================

import numpy as np
from scipy import stats
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.environment.simulation import Simulation
from src.agent.q_learning_agent import QLearningAgent
from src.agent.baselines import (AlwaysLocalAgent, AlwaysEdgeAgent,
                                 AlwaysCloudAgent, RandomAgent,
                                 GreedyHeuristicAgent)
from src.evaluation.metrics import composite_cost, summary
from src.config import (EVAL_NETWORK_QUALITY, SEED_STRIDE, EVAL_SEED_OFFSET,
                        train_sim_seed, eval_sim_seed)

SEEDS = [42, 123, 456, 789, 999]
TRAIN_EPISODES = 2_000
EVAL_DURATION  = 1_000

assert SEED_STRIDE > TRAIN_EPISODES, "seed stride must exceed episode count"
assert TRAIN_EPISODES < EVAL_SEED_OFFSET < SEED_STRIDE, "eval seed must be disjoint"


def train_agent(seed: int) -> QLearningAgent:
    """Train one Q-learning agent with a given seed."""
    agent = QLearningAgent(seed=seed)
    for ep in range(TRAIN_EPISODES):
        sim = Simulation(agent=agent, seed=train_sim_seed(seed, ep))
        sim.run(duration=100)
        agent.end_episode()
    agent.freeze()   # greedy AND frozen: evaluation must not update the Q-table
    return agent


def eval_seed(seed: int) -> int:
    """Evaluation seed, held disjoint from the training seed range."""
    return eval_sim_seed(seed)


def run_baseline(agent, seed: int) -> float:
    sim = Simulation(agent=agent, seed=eval_seed(seed),
                     network_quality=EVAL_NETWORK_QUALITY)
    tasks = sim.run(duration=EVAL_DURATION)
    return composite_cost(tasks)


def confidence_interval(values: list) -> tuple:
    """
    95% CI: mean +/- t_crit * std / sqrt(n).

    With only n=5 seeds we use the Student-t critical value (t_0.025,4 = 2.776),
    NOT the large-sample z-value of 1.96. Using 1.96 here would understate the
    interval width for such a small sample.
    """
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    sem  = float(np.std(arr, ddof=1)) / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))   # two-tailed 95%
    return mean, t_crit * sem


def paired_t_test(ql_costs: list, base_costs: list) -> dict:
    """
    Paired (dependent) two-tailed t-test of Q-learning vs the best baseline.

    Each seed produces a matched pair (ql_cost, base_cost) run under the SAME
    random conditions, so we test the per-seed DIFFERENCES rather than two
    independent means. This is the formal significance test backing the CI.

        H0: mean(base_cost - ql_cost) = 0   (no improvement)
        H1: mean(base_cost - ql_cost) != 0

    Differences are defined as (base - ql) so a POSITIVE mean difference means
    Q-learning is cheaper (better). The t-statistic is computed by hand so the
    formula is transparent for the thesis, then cross-checked against
    scipy.stats.ttest_rel.

    Returns t, df, p-value, mean difference, its 95% CI, and Cohen's d_z.
    """
    ql   = np.array(ql_costs, dtype=float)
    base = np.array(base_costs, dtype=float)
    diffs = base - ql                       # positive = QL better

    n      = len(diffs)
    df     = n - 1
    d_mean = float(np.mean(diffs))
    d_std  = float(np.std(diffs, ddof=1))   # sample std, divides by n-1
    sem    = d_std / np.sqrt(n)

    t_stat = d_mean / sem                          # by-hand t-statistic
    p_val  = 2.0 * float(stats.t.sf(abs(t_stat), df))   # two-tailed p-value

    # Cross-check against scipy's built-in paired test (should match t_stat/p_val)
    t_scipy, p_scipy = stats.ttest_rel(base, ql)

    t_crit = float(stats.t.ppf(0.975, df=df))      # = 2.776 for df=4
    ci_lo  = d_mean - t_crit * sem
    ci_hi  = d_mean + t_crit * sem

    cohens_dz = d_mean / d_std if d_std > 0 else float("inf")  # paired effect size

    return {
        "diffs":      diffs.tolist(),
        "d_mean":     d_mean,
        "d_std":      d_std,
        "t_stat":     t_stat,
        "df":         df,
        "p_value":    p_val,
        "t_scipy":    float(t_scipy),
        "p_scipy":    float(p_scipy),
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
        "cohens_dz":  cohens_dz,
        "significant": p_val < 0.05,
    }


def run_all(verbose: bool = True) -> dict:
    """
    Run full statistical validation across all SEEDS.

    Improvement is measured against the BEST fixed baseline per seed (the
    lowest-composite-cost of Always Edge / Always Cloud / Greedy Heuristic),
    so the headline number is honest rather than measured against a weaker
    baseline. A NEGATIVE improvement is a real outcome and is reported as
    such: the greedy heuristic is a strong competitor and may win.

    Returns dict with mean, CI, and improvement stats.
    """
    ql_costs, base_costs, base_names = [], [], []

    print(f"Running {len(SEEDS)} seeds - this takes a few minutes...\n")

    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({i+1}/{len(SEEDS)}): training...", end=" ", flush=True)

        # Train Q-learning
        agent = train_agent(seed)
        sim = Simulation(agent=agent, seed=eval_seed(seed),
                         network_quality=EVAL_NETWORK_QUALITY)
        tasks = sim.run(duration=EVAL_DURATION)
        ql_cost = composite_cost(tasks)
        ql_costs.append(ql_cost)

        # Best fixed baseline for the same seed (lowest composite cost)
        candidates = {
            "Always Edge":      run_baseline(AlwaysEdgeAgent(),      seed),
            "Always Cloud":     run_baseline(AlwaysCloudAgent(),     seed),
            "Greedy Heuristic": run_baseline(GreedyHeuristicAgent(), seed),
        }
        best_name = min(candidates, key=candidates.get)
        best_cost = candidates[best_name]
        base_costs.append(best_cost)
        base_names.append(best_name)

        imp = (best_cost - ql_cost) / best_cost * 100
        print(f"QL={ql_cost:.5f}  best={best_name}({best_cost:.5f})  improvement={imp:+.1f}%")

    # Compute CIs
    ql_mean, ql_ci     = confidence_interval(ql_costs)
    base_mean, base_ci = confidence_interval(base_costs)

    improvements = [(b - q) / b * 100 for q, b in zip(ql_costs, base_costs)]
    imp_mean, imp_ci   = confidence_interval(improvements)

    # Formal paired t-test on the raw composite costs (the matched pairs).
    ttest = paired_t_test(ql_costs, base_costs)

    results = {
        "seeds":         SEEDS,
        "ql_costs":      ql_costs,
        "base_costs":    base_costs,
        "base_names":    base_names,
        "improvements":  improvements,
        "ql_mean":       ql_mean,
        "ql_ci":         ql_ci,
        "base_mean":     base_mean,
        "base_ci":       base_ci,
        "imp_mean":      imp_mean,
        "imp_ci":        imp_ci,
        "ttest":         ttest,
        "significant":   ttest["significant"],   # backed by the paired t-test
    }

    if verbose:
        best_label = max(set(base_names), key=base_names.count)
        print()
        print("=" * 55)
        print("  STATISTICAL VALIDATION RESULTS")
        print("=" * 55)
        print(f"  Q-learning composite cost:    {ql_mean:.5f} ± {ql_ci:.5f}")
        print(f"  Best baseline ({best_label}): {base_mean:.5f} ± {base_ci:.5f}")
        print(f"  Improvement vs best baseline: {imp_mean:+.2f}% ± {imp_ci:.2f}%")
        print(f"  95% CI lower bound:           {imp_mean - imp_ci:+.2f}%")
        print()
        print("  PAIRED t-TEST  (H0: QL cost == baseline cost)")
        print("-" * 55)
        print(f"  Mean cost difference (base-QL): {ttest['d_mean']:+.5f} "
              f"(std {ttest['d_std']:.5f})")
        print(f"  95% CI of difference:           "
              f"[{ttest['ci_lo']:+.5f}, {ttest['ci_hi']:+.5f}]")
        print(f"  t({ttest['df']}) = {ttest['t_stat']:.3f}    "
              f"p = {ttest['p_value']:.5f}  (two-tailed)")
        print(f"  Cohen's d_z (effect size):      {ttest['cohens_dz']:.3f}")
        print(f"  scipy cross-check:  t={ttest['t_scipy']:.3f}  "
              f"p={ttest['p_scipy']:.5f}")
        print()
        if results["significant"]:
            print(f"  [PASS] STATISTICALLY SIGNIFICANT - p = {ttest['p_value']:.5f} < 0.05")
        else:
            print(f"  [FAIL] Not significant - p = {ttest['p_value']:.5f} >= 0.05")
        print("=" * 55)

    return results


if __name__ == "__main__":
    run_all()
