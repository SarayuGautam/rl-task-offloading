# Defense prep: Q-Learning for Task Offloading (EGPG 600)

Every number here comes from `results/summary.json` (frozen-policy evaluation). Each is traceable
through `AUDIT.md` §4.

## 60-second pitch

Phones and IoT devices can't run heavy tasks fast, but sending everything to the cloud is slow. Edge
servers sit in between. So for every task there is a choice: run it locally, at the edge, or in the cloud.
I framed that choice as an MDP and trained a **tabular Q-learning** agent in a SimPy simulator. The
simulator has Poisson arrivals, a Shannon-Hartley uplink and a real edge queue. I validated it against
queueing theory: M/M/1 within 0.39 %, and an end-to-end M/G/1 check within 0.21 %.

Across five independent seeds, with the policy frozen at evaluation, the agent is **15.6 % cheaper
than the best fixed strategy** (always-cloud; 95 % CI 13.5–17.7 %, p < 0.001). It learns an
interpretable rule on its own: use the edge when it's idle, the cloud when it's busy, never compute
locally.

I also report where it falls short. An untrained greedy heuristic that knows each task's CPU demand
is still **6.3 % cheaper**. And the latency-energy frontier is **degenerate**, because offload energy
doesn't depend on the destination. Both results tell us what the thesis needs: a richer state, and
continuous control.

## Key numbers

| What | Value |
|---|---|
| MDP | 36 states (edge load 4 × task size 3 × channel quality 3), 3 actions; reward = −(0.7 L + 0.3 E) |
| Hyperparameters | α 0.15, γ 0.9, ε 1.0 → 0.05 (× 0.998 per episode; floor at ep 1,497), Q₀ = 0 |
| Training | 2,000 episodes × 100 s (~600 tasks) ≈ 1.2 M updates per seed; channel q ~ U(0.1, 1) |
| Evaluation | frozen greedy, q = 0.9, 1,000 s per seed (Table 4.3); 10,000 s for seed 42 (Table 4.2) |
| Seeds | 42, 123, 456, 789, 999; disjoint simulation-seed blocks (seed × 10,000 + episode) |
| Q-learning cost (5 seeds) | 0.05856 ± 0.00120 (SD) |
| vs Always-Cloud | **+15.60 %**, CI [13.47, 17.74], t(4) = 20.31, p = 3.5e-5 (Holm 1.0e-4), d_z = 9.08 |
| vs Always-Edge (proposal target ≥ 10 %) | +19.84 %, CI [16.68, 22.99], so the target is met |
| vs Greedy Heuristic | **−6.32 %**, CI [−9.08, −3.57], t(4) = −6.38, p = 0.003, d_z = −2.85 |
| Heuristic vs Always-Cloud | +20.62 %, so Q-learning recovers ≈ 76 % of the heuristic's gain |
| Seed 42 (Table 4.2) | Q-learning 0.0850 s / cost 0.0595; heuristic 0.0784 / 0.0550; cloud 0.0990 / 0.0694; edge 0.1038 / 0.0727; local 1.1016 / 0.9364 |
| Policy | edge when idle (44/45 states), cloud at 3–5 tasks (45/45), mixed at 1–2 tasks (39 cloud / 6 edge); never local |
| Action split (seed 42) | 78.5 % edge, 21.5 % cloud, 0 % local |
| Coverage | 27/36 states, 81/81 reachable state-action pairs (all seeds); unvisited = the 9 states with ≥ 6 tasks at the edge |
| Convergence (5 % band) | seed 42: ep 1,364; median 1,428; range 1,364–1,602 |
| Q-values (18 well-trained states) | Q(local) −1.68 … −1.15; Q(edge), Q(cloud) −0.79 … −0.52 |
| Simulator checks | M/M/1 max error 0.39 %; P-K Always-Edge 0.1039 vs 0.1037 s; cloud 0.0990 vs 0.0990 s; local 1.1000 vs 1.0999 s |
| Pareto | 7 weightings; energy constant at 1.9439e-4 J; latency 0.0821–0.0909 s (seed 42, 1,500 episodes) |
| Uplink-in-event-time check | vs cloud +15.79 % [15.34, 16.23]; vs heuristic −4.89 % [−5.13, −4.64]; conclusions unchanged |
| Mobility probe | 81 of 324 states visited (25.0 %) under the same budget |
| Sensitivity (Table 4.4, 5 seeds each) | vs cloud +15.5 … +16.9 % and vs heuristic −4.6 … −6.4 % for every variant (α 0.05/0.30, γ 0/0.5/0.99, ε-decay 0.995/0.999); every CI keeps its sign |
| Old edge observable (history, seed 42) | vs cloud +9.39 % → +14.18 % after the fix; coverage 67/78 → 81/81 state-action pairs |

## What changed since the submitted version (say this first if asked)

In the final audit I found that the **Q-table was still being updated during "greedy" evaluation**. I
froze it (`agent.freeze()`, with a unit test) and re-ran every evaluation. Training was unaffected.
Coverage, convergence and the Q-values are identical.

| | Submitted report | Corrected |
|---|---|---|
| vs Always-Cloud | 17.03 % ± 0.22 | **15.60 %** [13.47, 17.74] |
| vs Greedy Heuristic | −4.53 % | **−6.32 %** [−9.08, −3.57] |
| t / d_z vs cloud | 225.96 / 101.05 | 20.31 / 9.08 |
| Action split | 70.8 / 29.2 | 78.5 / 21.5 (seed 42) |

The suspicious d_z = 101 in the old version was the symptom: continued learning made the five agents
converge. I also corrected three descriptive errors. The heuristic does **not** see "the same state":
it sees the CPU demand. The M/M/1 check validated only the queue primitive, so I added an end-to-end
M/G/1 check. And the §4.4 policy exceptions were described wrongly.

## 10 hardest questions

1. **"An untrained heuristic beats you. Why Q-learning?"** It isn't a fair fight, and the report says
   so. The heuristic gets the exact cost model plus each task's CPU demand and exact queue count. The agent
   sees 3 binned variables and no CPU demand. Even so, it recovers ≈ 76 % of the heuristic's gain from
   rewards alone. That matters when the cost model is unknown or wrong. The fix for the gap is to put CPU
   demand into the state: that's thesis step 1.
2. **"Why did your headline drop from 17 % to 15.6 %?"** Evaluation leakage (above). Frozen evaluation is
   the correct protocol, and a test now enforces it.
3. **"Is this even an MDP? One decision barely affects the next."** Decisions are coupled only
   through edge load, so the problem is close to a contextual bandit. Table 4.4 shows this directly:
   γ = 0 (fully myopic) gives cost 0.05815 vs 0.05856 for γ = 0.9, within noise. The planning horizon
   matters far less than the missing CPU-demand information.
4. **"Did you tune α, γ, ε?"** No. The single configuration was fixed before evaluation. Table 4.4 varies
   each one at a time over 5 seeds (variants fixed before running), and every variant stays 15.5–16.9 %
   cheaper than always-cloud and 4.6–6.4 % dearer than the heuristic. α = 0.05 is best (16.94 %, gap
   −4.63 %), which fits the "Q-value noise in near-tied states" explanation. I did **not** switch to it;
   its CI overlaps the default's.
5. **"How do you know the simulator is right?"** The queueing core matches M/M/1 within 0.39 %.
   End to end, Always-Edge matches the Pollaczek-Khinchine M/G/1 prediction within 0.21 %, and cloud and
   local match their closed forms within 0.01 %. These run as unit tests.
6. **"Five seeds? A t-test on n = 5?"** It's a paired design: every strategy sees the same workload per
   seed. I use t-based CIs (t = 2.776) and Holm correction across 5 comparisons, and the sign is consistent
   on every seed (cloud: +13.59 … +17.75 %; heuristic: −3.95 … −8.81 %). I report CIs first, following
   Henderson et al. (2018).
7. **"d_z = 9 looks absurd."** d_z = mean/SD of the paired differences. The common random numbers make
   the SD small. It measures consistency, not practical size; the percentage is the practical effect.
8. **"Why is local never chosen? Bug?"** No. A mean task needs 550 Mcycles / 500 Mcycles/s = 1.1 s
   locally, while even the worst channel uploads it in ≈ 40 ms. Q(local) ≤ −1.15 vs Q(offload) ≥ −0.79.
   Always-Local costs 0.94 against 0.059 for Q-learning.
9. **"Your Pareto 'frontier' is one point. Failure?"** It's a structural result. Device energy when
   offloading is P_tx · t_tx, and that is the same for edge and cloud. So among the actions the agent
   actually uses, energy can't move. A real trade-off needs continuous knobs (CPU frequency, transmit
   power): DDPG in the thesis.
10. **"Unvisited states? Generalisation?"** 9/36 states, all with ≥ 6 tasks at the edge, never occur at
    ρ ≈ 0.41. They're shown as untrained, and the fallback is edge. Tables don't generalise, which is
    exactly the DQN motivation. The mobility probe already drops coverage to 25 % of 324 states.

Backup questions:
- **Next-state definition.** s′ is observed at task completion and reuses the completed task's size, which
  biases the bootstrap. This is acknowledged in §5.2; the γ = 0 row removes the bootstrap entirely.
- **Edge queue ignores upload time.** Measured: simulating upload in event time gives +15.79 % vs cloud
  and −4.89 % vs heuristic. Same conclusions.
- **Train/test contamination.** Evaluation seeds are disjoint from all training blocks (a unit test asserts
  this), and the policy is frozen.

## Honest limitations

- A tabular, binned state that omits CPU demand (partial observability). It is beaten by an informed
  heuristic.
- The constant α leaves near-tied states (1–2 tasks at the edge) seed-dependent.
- There is only one evaluation channel (q = 0.9), a single device, and simulation only. Local
  execution has no contention, so its latency is a lower bound.
- The Pareto sweep uses one seed. The degeneracy is structural, not statistical.
- No DQN, no optimal-policy bound: there is **no claim** of a performance ceiling or of a crossover point.
- Title change and re-signing of the certificate: to be confirmed with the supervisor (TODO(HUMAN)).
