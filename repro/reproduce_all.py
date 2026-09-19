"""
reproduce_all.py - one command that regenerates every number, table and figure
in the report from the code in src/.

    python repro/reproduce_all.py            # all stages (about 30 min on 2 cores)
    python repro/reproduce_all.py main       # 5-seed results, Table 4.2/4.3, section 4.4
    python repro/reproduce_all.py figures    # redraw figures from saved results only

Stages: validate | main | pareto | mobility | sensitivity | uplink | queuefix | finalize | figures.
Each training run is checkpointed as one JSON file in results/raw/. A run that is
already checkpointed is skipped, so the pipeline can be interrupted and resumed.
Delete results/raw/ to force a clean re-run. Everything is deterministic given the
seeds in src/config.py and src/evaluation/statistical_tests.py.

Protocol (unchanged from the original study, except for evaluation freezing):
  * training: 2,000 episodes x 100 simulated s, channel quality ~ U(0.1, 1.0)
    per episode, epsilon-greedy with per-episode decay;
  * evaluation: greedy AND frozen (agent.freeze(): epsilon = 0, no learning),
    channel quality fixed at 0.9, workload seed disjoint from training;
  * 5 seeds (42, 123, 456, 789, 999), each on its own disjoint block of
    simulation seeds (seed * 10,000 + episode).
"""
import json
import os
import platform
import subprocess
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from scipy import stats

from src.config import (RANDOM_SEED, SIM_DURATION, EVAL_NETWORK_QUALITY,
                        LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_DECAY,
                        TASK_SIZE_MIN, TASK_SIZE_MAX, TASK_COMPLEXITY_MIN,
                        TASK_COMPLEXITY_MAX, TASK_ARRIVAL_RATE, EDGE_CPU_SPEED,
                        CLOUD_CPU_SPEED, DEVICE_CPU_SPEED, DEVICE_POWER,
                        EDGE_PROPAGATION_DELAY, CLOUD_PROPAGATION_DELAY,
                        TRANSMISSION_POWER, W_LATENCY, W_ENERGY,
                        train_sim_seed, eval_sim_seed)
from src.environment.simulation import Simulation
from src.environment.network_model import shannon_rate
from src.agent.q_learning_agent import QLearningAgent
from src.agent.baselines import (AlwaysLocalAgent, AlwaysEdgeAgent, AlwaysCloudAgent,
                                 RandomAgent, GreedyHeuristicAgent)
from src.evaluation.metrics import avg_latency, avg_energy, composite_cost, action_distribution
from src.evaluation.statistical_tests import SEEDS, TRAIN_EPISODES, EVAL_DURATION
from src.evaluation import pareto as pareto_mod

RES = os.path.join(ROOT, "results")
RAW = os.path.join(RES, "raw")
FIG = os.path.join(RES, "figures")
EPISODE_DURATION = 100
STRATEGIES = ["Q-learning", "Greedy Heuristic", "Always-Cloud", "Always-Edge",
              "Random", "Always-Local"]
# One-at-a-time sensitivity around the single reported configuration.
SENSITIVITY = {
    "alpha=0.05": dict(alpha=0.05), "alpha=0.30": dict(alpha=0.30),
    "gamma=0.0": dict(gamma=0.0), "gamma=0.5": dict(gamma=0.5), "gamma=0.99": dict(gamma=0.99),
    "eps_decay=0.995": dict(eps_decay=0.995), "eps_decay=0.999": dict(eps_decay=0.999),
}


# ----------------------------------------------------------------------------- helpers
def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def _load(path):
    with open(path) as f:
        return json.load(f)


def train(seed, episodes=TRAIN_EPISODES, alpha=LEARNING_RATE, gamma=DISCOUNT_FACTOR,
          eps_decay=EPSILON_DECAY, sim_kwargs=None):
    """Train one agent exactly as statistical_tests.train_agent does (optionally
    with one hyperparameter changed), then freeze it for evaluation."""
    sim_kwargs = sim_kwargs or {}
    agent = QLearningAgent(alpha=alpha, gamma=gamma, seed=seed)
    agent.epsilon_decay = eps_decay
    for ep in range(episodes):
        Simulation(agent=agent, seed=train_sim_seed(seed, ep), **sim_kwargs).run(
            duration=EPISODE_DURATION)
        agent.end_episode()
    agent.freeze()
    return agent


def make_agent(name, seed, ql=None):
    return {"Q-learning": lambda: ql,
            "Greedy Heuristic": GreedyHeuristicAgent,
            "Always-Cloud": AlwaysCloudAgent,
            "Always-Edge": AlwaysEdgeAgent,
            "Random": lambda: RandomAgent(seed=seed),
            "Always-Local": AlwaysLocalAgent}[name]()


def evaluate(agent, seed, duration, sim_kwargs=None):
    tasks = Simulation(agent=agent, seed=eval_sim_seed(seed),
                       network_quality=EVAL_NETWORK_QUALITY,
                       **(sim_kwargs or {})).run(duration=duration)
    return {"n": len(tasks), "latency": float(avg_latency(tasks)),
            "energy": float(avg_energy(tasks)), "cost": float(composite_cost(tasks)),
            "actions_pct": action_distribution(tasks)}


def convergence_episode(rewards, window=50, tail=200, tol_frac=0.05):
    """First episode after which the 50-episode moving average stays within 5 %
    of its final level (mean of the last 200 moving-average values)."""
    r = np.asarray(rewards, dtype=float)
    ma = np.convolve(r, np.ones(window) / window, mode="valid")
    final = ma[-tail:].mean()
    tol = tol_frac * abs(final)
    inside = np.abs(ma - final) <= tol
    # last index where the MA was outside the band; convergence is the next one
    outside = np.where(~inside)[0]
    idx = 0 if len(outside) == 0 else int(outside[-1]) + 1
    return int(idx + window), float(final)


def qtable_payload(agent):
    states = sorted(agent.q_table.keys())
    return {"states": [list(map(int, s)) for s in states],
            "q": [agent.q_table[s].tolist() for s in states],
            "visits": [agent.visit_counts[s].tolist() for s in states]}


def mean_ci(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    m = float(a.mean())
    return {"mean": m, "sd": sd, "ci95_half": half, "ci95": [m - half, m + half], "n": n}


def paired(ql, base):
    """Paired two-sided t-test on raw costs; diff = base - ql (positive = QL cheaper)."""
    ql, base = np.asarray(ql, float), np.asarray(base, float)
    d = base - ql
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(len(d)))
    p = 2 * stats.t.sf(abs(t), len(d) - 1)
    t_sp, p_sp = stats.ttest_rel(base, ql)
    imp = (base - ql) / base * 100
    return {"improvement_pct": mean_ci(imp), "per_seed_pct": imp.tolist(),
            "t": float(t), "df": len(d) - 1, "p": float(p),
            "t_scipy": float(t_sp), "p_scipy": float(p_sp),
            "cohens_dz": float(d.mean() / sd)}


# ----------------------------------------------------------------------------- jobs
def job_seed(seed):
    """Main 5-seed run: train, freeze, evaluate the agent and all five baselines."""
    out = os.path.join(RAW, f"seed_{seed}.json")
    if os.path.exists(out):
        return out
    ql = train(seed)
    res = {"seed": seed, "episode_rewards": ql.episode_rewards,
           "coverage": ql.coverage(expected_states=36),
           "convergence_episode": convergence_episode(ql.episode_rewards)[0],
           "eval_1000s": {name: evaluate(make_agent(name, seed, ql), seed, EVAL_DURATION)
                          for name in STRATEGIES},
           "qtable": qtable_payload(ql)}
    if seed == RANDOM_SEED:   # Table 4.2: the long single-seed evaluation
        res["eval_table42"] = {name: evaluate(make_agent(name, seed, ql), seed, SIM_DURATION)
                               for name in STRATEGIES}
    _dump(res, out)
    return out


def job_sensitivity(args):
    label, seed = args
    out = os.path.join(RAW, "sensitivity", f"{label}_seed_{seed}.json")
    if os.path.exists(out):
        return out
    ql = train(seed, **SENSITIVITY[label])
    _dump({"label": label, "seed": seed, "eval": evaluate(ql, seed, EVAL_DURATION),
           "coverage": ql.coverage(expected_states=36),
           "convergence_episode": convergence_episode(ql.episode_rewards)[0]}, out)
    return out


def job_pareto(beta):
    out = os.path.join(RAW, "pareto", f"beta_{beta:.2f}.json")
    if os.path.exists(out):
        return out
    _dump(pareto_mod.train_and_eval(1.0 - beta, beta), out)   # train_and_eval freezes
    return out


def job_mobility(_=None):
    out = os.path.join(RAW, "mobility.json")
    if os.path.exists(out):
        return out
    ql = train(RANDOM_SEED, sim_kwargs=dict(use_mobility=True, velocity=0.8))
    _dump({"seed": RANDOM_SEED, "velocity": 0.8,
           "coverage": ql.coverage(expected_states=324)}, out)
    return out


def job_uplink(seed):
    """Robustness check for finding F-09: uplink time simulated in event time."""
    out = os.path.join(RAW, "uplink", f"seed_{seed}.json")
    if os.path.exists(out):
        return out
    kw = dict(uplink_in_event_time=True)
    ql = train(seed, sim_kwargs=kw)
    _dump({"seed": seed, "eval": {name: evaluate(make_agent(name, seed, ql), seed,
                                                 EVAL_DURATION, sim_kwargs=kw)
                                  for name in ["Q-learning", "Greedy Heuristic",
                                               "Always-Cloud", "Always-Edge"]}}, out)
    return out


def job_queuefix(seed):
    """History check for report section 5.1: the pre-fix edge observable counted only
    WAITING tasks (not the one in service). Re-measured here with frozen evaluation."""
    out = os.path.join(RAW, "queuefix", f"seed_{seed}.json")
    if os.path.exists(out):
        return out
    from src.environment.edge_server import EdgeServer
    fixed = EdgeServer.queue_length
    EdgeServer.queue_length = property(lambda self: len(self.resource.queue))
    try:
        ql = train(seed)
        res = {"seed": seed, "coverage": ql.coverage(expected_states=36),
               "eval_table42": {n: evaluate(make_agent(n, seed, ql), seed, SIM_DURATION)
                                for n in ["Q-learning", "Always-Cloud"]}}
    finally:
        EdgeServer.queue_length = fixed
    _dump(res, out)
    return out


def _dispatch(job):
    kind, arg = job
    return {"seed": job_seed, "sens": job_sensitivity, "pareto": job_pareto,
            "mobility": job_mobility, "uplink": job_uplink, "queuefix": job_queuefix}[kind](arg)


def run_jobs(jobs, workers=None):
    workers = workers or max(1, min(len(jobs), os.cpu_count() or 1))
    with Pool(workers) as pool:
        for path in pool.imap_unordered(_dispatch, jobs):
            print("  done:", os.path.relpath(path, ROOT), flush=True)


# ----------------------------------------------------------------------------- stages
def stage_validate():
    """Simulator checks against closed-form results (see tests/test_simulator.py)."""
    from src.evaluation.queue_validation import run_validation
    mm1 = run_validation(verbose=False)
    rate = float(shannon_rate(EVAL_NETWORK_QUALITY))
    e_size = (TASK_SIZE_MIN + TASK_SIZE_MAX) / 2
    e_c = (TASK_COMPLEXITY_MIN + TASK_COMPLEXITY_MAX) / 2
    a, b = TASK_COMPLEXITY_MIN / EDGE_CPU_SPEED, TASK_COMPLEXITY_MAX / EDGE_CPU_SPEED
    es, es2 = (a + b) / 2, (a * a + a * b + b * b) / 3            # uniform service moments
    rho = TASK_ARRIVAL_RATE * es
    wq = TASK_ARRIVAL_RATE * es2 / (2 * (1 - rho))                  # Pollaczek-Khinchine
    theory = {"Always-Edge": e_size / rate + EDGE_PROPAGATION_DELAY + wq + es,
              "Always-Cloud": e_size / rate + CLOUD_PROPAGATION_DELAY + e_c / CLOUD_CPU_SPEED,
              "Always-Local": e_c / DEVICE_CPU_SPEED}
    energy_theory = {"offload": TRANSMISSION_POWER * e_size / rate,
                     "local": DEVICE_POWER * e_c / DEVICE_CPU_SPEED}
    sim = {}
    for name in theory:
        lat = [evaluate(make_agent(name, s), s, SIM_DURATION)["latency"] for s in SEEDS]
        sim[name] = {"theory_s": theory[name], "sim_mean_s": float(np.mean(lat)),
                     "rel_err_pct": float(abs(np.mean(lat) - theory[name]) / theory[name] * 100)}
    _dump({"mm1_standalone": mm1, "edge_rho": rho, "edge_Wq_PK_s": wq,
           "energy_theory_J": energy_theory, "closed_form_vs_sim": sim}, os.path.join(RES, "validation.json"))
    for k, v in sim.items():
        print(f"  {k:13s} theory {v['theory_s']:.5f} s  sim {v['sim_mean_s']:.5f} s  err {v['rel_err_pct']:.2f}%")


def stage_main():
    run_jobs([("seed", s) for s in SEEDS])


def stage_pareto():
    run_jobs([("pareto", b) for b in pareto_mod.BETAS])


def stage_mobility():
    run_jobs([("mobility", None)], workers=1)


def stage_sensitivity():
    run_jobs([("sens", (lab, s)) for lab in SENSITIVITY for s in SEEDS])


def stage_uplink():
    run_jobs([("uplink", s) for s in SEEDS])


def stage_queuefix():
    run_jobs([("queuefix", RANDOM_SEED)], workers=1)


def stage_finalize():
    seeds = [_load(os.path.join(RAW, f"seed_{s}.json")) for s in SEEDS]
    cost = {n: [r["eval_1000s"][n]["cost"] for r in seeds] for n in STRATEGIES}
    lat = {n: [r["eval_1000s"][n]["latency"] for r in seeds] for n in STRATEGIES}
    eng = {n: [r["eval_1000s"][n]["energy"] for r in seeds] for n in STRATEGIES}
    table43 = {"per_strategy": {n: {"cost": mean_ci(cost[n]), "latency": mean_ci(lat[n]),
                                    "energy": mean_ci(eng[n]), "per_seed_cost": cost[n]}
                                for n in STRATEGIES},
               "vs": {n: paired(cost["Q-learning"], cost[n]) for n in STRATEGIES[1:]},
               "ql_actions_pct": [r["eval_1000s"]["Q-learning"]["actions_pct"] for r in seeds],
               "coverage": [r["coverage"] for r in seeds],
               "convergence_episode": [r["convergence_episode"] for r in seeds]}
    # Holm-Bonferroni over the five QL-vs-baseline tests
    ps = sorted((v["p"], n) for n, v in table43["vs"].items())
    m = len(ps)
    running = 0.0
    for i, (p, n) in enumerate(ps):
        running = max(running, min(1.0, (m - i) * p))
        table43["vs"][n]["p_holm"] = running
    s42 = seeds[SEEDS.index(RANDOM_SEED)]
    table42 = s42["eval_table42"]
    diag = diagnostics(s42)
    out = {"table_4_2_seed42_10000s": table42, "table_4_3_five_seeds_1000s": table43,
           "section_4_4_seed42": diag}
    if os.path.isdir(os.path.join(RAW, "pareto")):
        rows = [_load(os.path.join(RAW, "pareto", f"beta_{b:.2f}.json")) for b in pareto_mod.BETAS
                if os.path.exists(os.path.join(RAW, "pareto", f"beta_{b:.2f}.json"))]
        e = [r["avg_energy"] for r in rows]
        out["pareto"] = {"rows": rows, "energy_min": min(e), "energy_max": max(e),
                         "train_episodes": pareto_mod.TRAIN_EPISODES,
                         "eval_duration_s": pareto_mod.EVAL_DURATION, "seed": 42}
    if os.path.exists(os.path.join(RAW, "mobility.json")):
        out["mobility"] = _load(os.path.join(RAW, "mobility.json"))
    sens_dir = os.path.join(RAW, "sensitivity")
    if os.path.isdir(sens_dir):
        sens = {"default (alpha=0.15, gamma=0.9, eps_decay=0.998)": _sens_row(cost["Q-learning"], cost, seeds)}
        for lab in SENSITIVITY:
            files = [os.path.join(sens_dir, f"{lab}_seed_{s}.json") for s in SEEDS]
            if all(os.path.exists(f) for f in files):
                rs = [_load(f) for f in files]
                sens[lab] = _sens_row([r["eval"]["cost"] for r in rs], cost, rs)
        out["sensitivity"] = sens
    up_dir = os.path.join(RAW, "uplink")
    if os.path.isdir(up_dir):
        files = [os.path.join(up_dir, f"seed_{s}.json") for s in SEEDS]
        if all(os.path.exists(f) for f in files):
            ups = [_load(f) for f in files]
            uc = {n: [u["eval"][n]["cost"] for u in ups] for n in ups[0]["eval"]}
            out["uplink_in_event_time"] = {
                "per_strategy_cost": {n: mean_ci(v) for n, v in uc.items()},
                "vs": {n: paired(uc["Q-learning"], uc[n]) for n in uc if n != "Q-learning"}}
    qf = os.path.join(RAW, "queuefix", f"seed_{RANDOM_SEED}.json")
    if os.path.exists(qf):
        q = _load(qf)
        e = q["eval_table42"]
        q["improvement_vs_cloud_pct"] = (e["Always-Cloud"]["cost"] - e["Q-learning"]["cost"]) / e["Always-Cloud"]["cost"] * 100
        new = out["table_4_2_seed42_10000s"]
        q["fixed_observable_improvement_vs_cloud_pct"] = (new["Always-Cloud"]["cost"] - new["Q-learning"]["cost"]) / new["Always-Cloud"]["cost"] * 100
        out["queue_observable_history"] = q
    out["manifest"] = manifest()
    _dump(out, os.path.join(RES, "summary.json"))
    print_summary(out)


def _sens_row(ql_costs, base_cost, runs):
    return {"cost": mean_ci(ql_costs),
            "vs_Always-Cloud": paired(ql_costs, base_cost["Always-Cloud"])["improvement_pct"],
            "vs_Greedy": paired(ql_costs, base_cost["Greedy Heuristic"])["improvement_pct"],
            "coverage_sa": [r["coverage"]["state_actions_visited"] for r in runs],
            "convergence_episode": [r["convergence_episode"] for r in runs]}


def diagnostics(s42):
    """Section 4.4 numbers from the seed-42 Q-table (training-end = frozen table)."""
    qt = s42["qtable"]
    states = [tuple(s) for s in qt["states"]]
    q = {s: np.array(v) for s, v in zip(states, qt["q"])}
    vis = {s: np.array(v) for s, v in zip(states, qt["visits"])}
    allstates = [(a, b, c) for a in range(4) for b in range(3) for c in range(3)]
    unvisited = [s for s in allstates if s not in vis or not (vis[s] > 0).any()]
    well = [s for s in states if (vis[s] >= 30).all()]
    rng = {a: [float(min(q[s][a] for s in well)), float(max(q[s][a] for s in well))]
           for a in range(3)} if well else {}
    names = "LEC"
    grid = {}
    for s in states:
        v = vis[s]
        tried = v >= 30
        low = False
        if not tried.any():
            tried, low = v > 0, True
        if not tried.any():
            continue
        grid[str(s)] = names[int(np.argmax(np.where(tried, q[s], -np.inf)))] + ("?" if low else "")
    greedy_any = {str(s): names[int(np.argmax(np.where(vis[s] > 0, q[s], -np.inf)))]
                  for s in states if (vis[s] > 0).any()}
    return {"coverage": s42["coverage"], "unvisited_states": [list(s) for s in unvisited],
            "n_well_trained_30": len(well), "q_range_well_trained": {"local": rng.get(0), "edge": rng.get(1),
                                                                      "cloud": rng.get(2)},
            "min_visits_per_action_by_queue_bin": {
                qb: int(min(vis[s][vis[s] > 0].min() for s in states if s[0] == qb))
                for qb in sorted({s[0] for s in states})},
            "greedy_display_rule": grid, "greedy_any_visit": greedy_any,
            "local_ever_greedy": any(v == "L" for v in greedy_any.values()),
            "convergence_episode": s42["convergence_episode"],
            "table42_ql_actions_pct": s42["eval_table42"]["Q-learning"]["actions_pct"]}


def manifest():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src"], cwd=ROOT).decode().strip())
    except Exception:
        commit, dirty = "unknown", None
    import matplotlib, simpy
    return {"git_commit": commit, "src_dirty": dirty, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": __import__("scipy").__version__,
            "simpy": simpy.__version__, "matplotlib": matplotlib.__version__,
            "seeds": SEEDS, "train_episodes": TRAIN_EPISODES, "eval_duration_s": EVAL_DURATION,
            "table42_eval_duration_s": SIM_DURATION, "eval_quality": EVAL_NETWORK_QUALITY,
            "weights": [W_LATENCY, W_ENERGY]}


def print_summary(out):
    t42 = out["table_4_2_seed42_10000s"]
    print("\nTable 4.2 (seed 42, 10,000 s, frozen greedy evaluation)")
    for n in STRATEGIES:
        r = t42[n]
        print(f"  {n:17s} n={r['n']:6d} lat={r['latency']:.4f} eng={r['energy']:.6f} "
              f"cost={r['cost']:.4f} {r['actions_pct']}")
    t = out["table_4_3_five_seeds_1000s"]
    print("\nTable 4.3 (5 seeds, 1,000 s each)")
    for n in STRATEGIES:
        c = t["per_strategy"][n]["cost"]
        print(f"  {n:17s} cost {c['mean']:.5f}  sd {c['sd']:.5f}  95%CI +/-{c['ci95_half']:.5f}")
    for n, v in t["vs"].items():
        i = v["improvement_pct"]
        print(f"  QL vs {n:17s} {i['mean']:+7.2f}%  sd {i['sd']:.2f}  CI [{i['ci95'][0]:+.2f}, "
              f"{i['ci95'][1]:+.2f}]  t({v['df']})={v['t']:.2f}  p={v['p']:.2e}  "
              f"p_holm={v['p_holm']:.2e}  dz={v['cohens_dz']:.2f}  per-seed {np.round(v['per_seed_pct'], 2).tolist()}")
    print("  convergence episodes:", t["convergence_episode"])
    print("  coverage:", [(c["states_visited"], c["state_actions_visited"]) for c in t["coverage"]])
    d = out["section_4_4_seed42"]
    print("\nSection 4.4 (seed 42):", {k: d[k] for k in ("n_well_trained_30", "q_range_well_trained",
                                                           "local_ever_greedy", "convergence_episode",
                                                           "table42_ql_actions_pct",
                                                           "min_visits_per_action_by_queue_bin")})
    print("  unvisited:", d["unvisited_states"])
    if "pareto" in out:
        p = out["pareto"]
        print(f"\nPareto: energy min {p['energy_min']:.10e} max {p['energy_max']:.10e}")
        for r in p["rows"]:
            print(f"  w_lat={r['w_latency']:.2f} lat={r['avg_latency']:.5f} eng={r['avg_energy']:.6e}")
    if "mobility" in out:
        print("\nMobility coverage:", out["mobility"]["coverage"])
    if "sensitivity" in out:
        print("\nSensitivity (5 seeds each):")
        for lab, r in out["sensitivity"].items():
            print(f"  {lab:48s} cost {r['cost']['mean']:.5f}+/-{r['cost']['ci95_half']:.5f}  "
                  f"vs Cloud {r['vs_Always-Cloud']['mean']:+.2f}%  vs Greedy {r['vs_Greedy']['mean']:+.2f}%  "
                  f"SA {r['coverage_sa']}  conv {r['convergence_episode']}")
    if "queue_observable_history" in out:
        q = out["queue_observable_history"]
        print(f"\nOld queue observable (seed 42): QL vs Cloud {q['improvement_vs_cloud_pct']:+.2f}% "
              f"(fixed: {q['fixed_observable_improvement_vs_cloud_pct']:+.2f}%), coverage {q['coverage']}")
    if "uplink_in_event_time" in out:
        print("\nUplink-in-event-time robustness:")
        for n, v in out["uplink_in_event_time"]["vs"].items():
            print(f"  QL vs {n:17s} {v['improvement_pct']['mean']:+.2f}% +/- {v['improvement_pct']['ci95_half']:.2f}")


def stage_figures():
    from repro import figures
    figures.make_all(_load(os.path.join(RES, "summary.json")), RAW, FIG)


STAGES = {"validate": stage_validate, "main": stage_main, "pareto": stage_pareto,
          "mobility": stage_mobility, "sensitivity": stage_sensitivity,
          "uplink": stage_uplink, "queuefix": stage_queuefix, "finalize": stage_finalize,
          "figures": stage_figures}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["validate", "main", "pareto", "mobility", "sensitivity",
                              "uplink", "queuefix", "finalize", "figures"]
    for name in wanted:
        print(f"\n=== stage: {name} ===", flush=True)
        STAGES[name]()
