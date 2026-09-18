# =============================================================================
# evaluation/queue_validation.py
#
# Simulation validity check - REQUIRED step named in the literature review
# (Sections 2.4 and 2.8: "validity ... must be established through comparison
# against queuing theory predictions ... through comparison with the M/M/1
# queue model").
#
# WHAT THIS PROVES:
#   Our edge server is a SimPy single-server resource with a FIFO queue.
#   If our discrete-event engine is correct, an M/M/1 configuration
#   (Poisson arrivals + exponential service + one server) must reproduce
#   the closed-form M/M/1 formulas from queuing theory.
#
# M/M/1 THEORY (Kendall notation: Markovian arrivals / Markovian service / 1 server):
#   ρ   = λ / μ                 utilisation (must be < 1 for a stable queue)
#   L   = ρ / (1 − ρ)           avg number of tasks in the system
#   W   = 1 / (μ − λ)           avg time in system (wait + service)
#   Wq  = ρ / (μ − λ)           avg time waiting in queue (before service)
#
# We run the SimPy queue and check the empirical averages match these
# formulas within a small tolerance. Matching = the simulator is valid.
# =============================================================================

import simpy
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def simulate_mm1(arrival_rate: float, service_rate: float,
                 duration: float = 50_000, seed: int = 42) -> dict:
    """
    Run a single-server M/M/1 queue in SimPy and measure empirical averages.

    Args:
        arrival_rate (lambda) : mean tasks arriving per second (Poisson)
        service_rate (mu)     : mean tasks served per second (exponential)
        duration              : simulation seconds (longer = tighter estimate)
        seed                  : RNG seed for reproducibility

    Returns:
        dict with empirical W (time in system) and Wq (time in queue)
    """
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)

    waits_in_system, waits_in_queue = [], []

    def customer(env, name):
        arrived = env.now
        with server.request() as req:
            yield req
            wait_q = env.now - arrived          # time spent waiting in queue
            service_time = rng.exponential(1.0 / service_rate)
            yield env.timeout(service_time)
        waits_in_queue.append(wait_q)
        waits_in_system.append(env.now - arrived)

    def arrivals(env):
        i = 0
        while True:
            yield env.timeout(rng.exponential(1.0 / arrival_rate))
            i += 1
            env.process(customer(env, i))

    env.process(arrivals(env))
    env.run(until=duration)

    return {
        "n_served":  len(waits_in_system),
        "W_emp":     float(np.mean(waits_in_system)),
        "Wq_emp":    float(np.mean(waits_in_queue)),
    }


def mm1_theory(arrival_rate: float, service_rate: float) -> dict:
    """Closed-form M/M/1 predictions."""
    lam, mu = arrival_rate, service_rate
    rho = lam / mu
    return {
        "rho":   rho,
        "W_th":  1.0 / (mu - lam),
        "Wq_th": rho / (mu - lam),
    }


def run_validation(verbose: bool = True) -> dict:
    """
    Validate the SimPy queue against M/M/1 theory at several load levels.
    Returns a dict of results; flags overall pass/fail at 5% tolerance.
    """
    # Service rate fixed; sweep arrival rate to test light → heavy load.
    mu = 14.5                       # ~matches edge: 1 / (mean_complexity / EDGE_CPU_SPEED)
    arrival_rates = [3.0, 6.0, 9.0, 12.0]   # rho = 0.21, 0.41, 0.62, 0.83
    tolerance = 0.05                # 5% relative error allowed

    rows, all_pass = [], True
    for lam in arrival_rates:
        th  = mm1_theory(lam, mu)
        emp = simulate_mm1(lam, mu)
        err_W  = abs(emp["W_emp"]  - th["W_th"])  / th["W_th"]
        err_Wq = abs(emp["Wq_emp"] - th["Wq_th"]) / th["Wq_th"] if th["Wq_th"] > 0 else 0.0
        passed = err_W < tolerance
        all_pass = all_pass and passed
        rows.append({
            "lambda": lam, "rho": th["rho"],
            "W_th": th["W_th"], "W_emp": emp["W_emp"], "err_W": err_W,
            "Wq_th": th["Wq_th"], "Wq_emp": emp["Wq_emp"], "err_Wq": err_Wq,
            "n": emp["n_served"], "passed": passed,
        })

    if verbose:
        print("=" * 78)
        print("  SIMULATION VALIDATION - SimPy queue vs M/M/1 theory")
        print(f"  Service rate mu = {mu}/s   |   tolerance = {tolerance*100:.0f}% relative error")
        print("=" * 78)
        print(f"  {'lambda':>7} {'rho':>5} {'W_theory':>10} {'W_sim':>10} "
              f"{'err':>7} {'served':>8} {'verdict':>9}")
        for r in rows:
            print(f"  {r['lambda']:>7.1f} {r['rho']:>5.2f} {r['W_th']:>10.4f} "
                  f"{r['W_emp']:>10.4f} {r['err_W']*100:>6.1f}% {r['n']:>8d} "
                  f"{'PASS' if r['passed'] else 'FAIL':>9}")
        print("=" * 78)
        if all_pass:
            print("  RESULT: VALID - SimPy queue matches M/M/1 theory within tolerance.")
        else:
            print("  RESULT: MISMATCH - investigate before trusting simulation outputs.")
        print("=" * 78)

    return {"rows": rows, "all_pass": all_pass}


if __name__ == "__main__":
    run_validation()
