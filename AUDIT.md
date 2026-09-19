# AUDIT — defense-final branch

Audit of the code and of the report `EGPG600_Report_SarayuGautam.docx` (original kept as-is;
edits go to `EGPG600_Report_SarayuGautam_v2.docx`). Rows: `severity | location | problem | fix`.
Severity: P0 = wrong result/claim, P1 = misleading or irreproducible, P2 = presentation.
Status tags: VERIFIED / FIXED / TODO(HUMAN).

## 1. Triage map (Phase 1)

- Algorithm: tabular Q-learning (dict Q-table, 36-state base space, 3 actions). There is no DQN or Double-Q
  implementation: `src/agent/dqn_agent.py` is a stub. `act()` returns a random action, and the network and
  training step raise `NotImplementedError`. It is not wired into `main.py`.
- Environment: SimPy event-driven, continuing task. There is no terminal state; an episode is a
  100 s time-limit truncation. Poisson arrivals (λ = 6/s), uniform size and complexity, Shannon uplink,
  one-server FIFO edge queue with uniform service (M/G/1), infinite-server cloud.
- Entry points: `main.py --agent {qlearning,all,stats,validate,pareto,<baseline>}`. The old `repro/01..05_*.py`
  scripts produced the numbers in the report.
- Config: `src/config.py` holds the only hyperparameter set. Protocol constants are in
  `statistical_tests.py` (5 seeds, 2,000 episodes, 1,000 s eval) and `pareto.py` (1,500 episodes, 2,000 s eval).
- Results in the report came from `repro/sample_outputs/*.json` and the charts. **Environment check:** the
  pre-fix code in this sandbox reproduces the seed-42 checkpoint exactly
  (Q-learning 0.0574315, Always-Cloud 0.0693729, Always-Edge 0.0724863, Greedy 0.0550904). So the saved
  artifacts come from this code.
- Report: 5 chapters (Intro, Literature Review, Methodology, Results, Conclusion), 4 result figures
  (image1 = Fig 4.1 comparison, image4 = Fig 4.2 learning curve, image3 = Fig 4.3 heatmap,
  image6 = Fig 4.4 Pareto), 3 diagram figures (3.1–3.3; no editable source in the repo), and 4 tables.
- Branches: `main` = squashed commit `d50171c` (the patch `qlearning-fixes.patch` is already applied). Also
  `origin/fix/step1-simulated-network-time` (unmerged; see F-09) and
  `origin/fix/state-space-coverage-and-eval-consistency` (older history).

## 2. Findings

| # | Sev | Location | Problem | Fix / status |
|---|---|---|---|---|
| F-01 | P0 | `simulation.py:158-160`; every evaluation call site | "Greedy evaluation" still calls `agent.learn()`, so the Q-table keeps updating on the evaluation workload. `main.generate_charts` also re-evaluates the agent on the same seed it just learned from. | FIXED: `QLearningAgent.freeze()` (ε = 0, learning off) is used at every evaluation site. All evaluation numbers were re-run. |
| F-02 | P0 | Report: Abstract, §4.3 | Says the Greedy Heuristic "observes the same state as the agent". It does not: it reads the raw task size, the CPU complexity (not in the agent's state at all), the exact queue count and the exact channel quality. | FIXED in v2 wording |
| F-03 | P0 | Report §4.1 | The "simulator validation" validates a standalone `simpy.Resource` M/M/1 queue with exponential service. The experiments use uniform service (M/G/1) inside `Simulation`. | FIXED: §4.1 reworded; added an M/G/1 Pollaczek–Khinchine check and closed-form cloud/local checks of the real `Simulation` (tests + report) |
| F-04 | P1 | `repro/*.py` | Crash with `ModuleNotFoundError: src` when run as documented (`python repro/…`). | FIXED: replaced by a single `repro/reproduce_all.py` |
| F-05 | P1 | `repro/sample_outputs/table_4_3_seed_results.json` | Seeds 42 and 123 are stored rounded to 5 dp; the other seeds are full precision. The mixed provenance was used for Table 4.3. | FIXED: regenerated at full precision |
| F-06 | P1 | `README.md`, `PROJECT_STATUS.md` | Stale headline (+9.13 % ± 1.05 %, 65/72 coverage) contradicts the report (+17.03 %). They also link files outside the repo. | FIXED: README rewritten; PROJECT_STATUS marked superseded |
| F-07 | P1 | `requirements.txt` | Pins `torch`, `jupyter` and `pandas`, none of which are imported. | FIXED: minimal pinned set |
| F-08 | P1 | Report §3.5 vs `q_learning_agent.py:121-139` | The code bootstraps with max over actions *tried* in s′ (0 if none). The report shows the textbook max over all actions. | FIXED: documented in the §3.5 algorithm box |
| F-09 | P1 | `origin/fix/step1-simulated-network-time` (unmerged) | On `main`, an edge task joins the queue at generation time; uplink and propagation are added analytically, not in event time. | Kept `main` semantics (the report describes them). Impact measured with an opt-in flag; see §4 |
| F-10 | P1 | Report Abstract, §5.1 | "± 0.22 %" is a 95 % CI half-width, but the text does not say so. | FIXED: every ± is labelled |
| F-11 | P1 | Report §4.5 | Pareto agents use 1,500 training episodes and 2,000 s evaluation (not 2,000 episodes). The report is silent on this. | FIXED: stated |
| F-12 | P1 | Report §5.3 | Mobility coverage "81 of 324 (25.0 %)" has no saved artifact. | Re-run; see ledger |
| F-13 | P2 | Fig 3.1 (image5) | The "State observation" box lists queue size and network quality but omits task size. | TODO(HUMAN): no diagram source in the repo; noted in the caption |
| F-14 | P1 | Report §3.4.1 | Says "the 24 [states] with an edge queue length below six are reachable". Queue < 6 covers 3 bins × 9 = 27 states (24 was left over from the old 24/36 coverage). | FIXED |
| F-15 | P1 | Report §4.4 (policy description) | "Only exception: at intermediate quality, large tasks prefer the edge for one more queue level." The frozen seed-42 table has two different exceptions (1–2 tasks: small tasks at ok quality; large tasks at good quality), and the exceptions differ across seeds. | FIXED from `results/raw/seed_*.json` |
| F-16 | P1 | Report §4.3.3 | Explains the jump of d_z to 101 by the queue-observable fix. The actual cause was the evaluation leakage (F-01): with a frozen policy, d_z = 9.08. | FIXED |
| F-17 | P1 | Report §5.2 | "Always-Local latency of 1.099 s" disagrees with Table 4.2 (1.1016 s). | FIXED (uses the Table 4.2 value) |
| F-18 | P1 | Report §5.3 | "preliminary experiments on this pipeline are already under way": nothing in the repo supports this. | REMOVED; TODO(HUMAN) reinstate only with evidence |
| F-19 | P2 | Report §2.3/§2.7 | Notation clash: α/β as cost weights vs α as the learning rate; T as delay vs T as the transition function. | FIXED: w_lat/w_eng, L, P(s′\|s,a); math set as OMML equations |
| F-20 | P2 | Report §2.6.1/§2.9 | Universal literature claims ("No equivalent Q-Learning baseline exists …"). | Softened to "in the literature reviewed" |
| F-21 | P2 | Report §2.6.2 | The DDQN sentence has no citation (van Hasselt et al., 2016). | TODO(HUMAN): add the citation if wanted (not verified online here) |
| F-22 | P2 | Report | Mixed US/UK spelling (utilization/modeled/optimizes). | FIXED to UK spelling in running text |
| F-23 | P2 | Report metadata | No docProps/core.xml (no title metadata). | FIXED: title/creator added |
| F-24 | P1 | Report Declaration/Certificate | Both carry the project title. The retitle changes a document already signed by the supervisor. | TODO(HUMAN): confirm the title change with supervisor/department; re-sign if required |

## 3. Decisions and assumptions (logged instead of asking)

- D-1 Kept `main` simulator semantics (edge queue entered at generation time, uplink added analytically) for all
  reported results. The report describes these semantics, and the unmerged branch was never used for any result. The branch's change is available behind
  `UPLINK_IN_EVENT_TIME` and measured as a robustness check (5 seeds).
- D-2 Protocol unchanged (seeds, episodes, α, γ, ε schedule, eval durations, eval quality). The only change is freezing the
  Q-table during evaluation. No run was dropped, and no seed or hyperparameter was tuned. The sensitivity variants were fixed
  before any of them ran.
- D-3 The chapter structure of the university template is kept (Intro / Lit review / Method / Results / Conclusion).
  The requested paper-style elements were mapped onto it: Background → §2.3 (MDP + Bellman optimality), Method →
  §3.5 + Algorithm 1, Discussion (incl. failures) → new §4.7, Sensitivity → new §4.6 + Table 4.4, Limitations → §5.2.
- D-4 Figures are embedded as 300-dpi PNG, because the .docx template and LibreOffice PDF export handle PNG robustly. Vector PDFs of every
  figure are in `results/figures/` for slides/LaTeX.
- D-5 Retitle policy: "Q-Learning" in the title page, declaration, certificate, headings and figure titles; "Q-learning" in running text
  and captions. "Reinforcement learning" is kept where it names the field (background §2.6, acknowledgement, future
  DRL work, the abbreviation list) and in cited titles. The repo name `rl-task-offloading` and CLI identifiers
  (`--agent qlearning`, `run_qlearning`) are unchanged, to keep paths and imports intact.
- D-6 Phases 2 and 3 touched the same files, so they were committed together (commit `7332f74`).
- D-7 Deleted files (all recoverable from git history): `repro/01_…py`–`repro/05b_…py`, `repro/sample_outputs/**`
  (pre-fix artifacts incl. a 59,942-line eval CSV), `qlearning-fixes.patch` (already applied on `main`).

## 4. Claim ledger (report v1 claim → source → status)

Sources: `cfg` = `src/config.py`; `S` = `results/summary.json`; `V` = `results/validation.json`;
`raw` = `results/raw/`. Every v2 number is generated from S/V by `report/build_v2.py`, so none is typed
by hand. Status: VERIFIED (v1 claim holds) · FIXED (v1 claim was wrong or stale; v2 value shown) · TODO(HUMAN).

### Model, method and hyperparameters (Abstract, §1–§3, Table 3.1)
| Claim (v1) | Source | Status |
|---|---|---|
| λ = 6 tasks/s Poisson; size 1,000–10,000 bits; complexity 100–1,000 Mcycles | cfg:8,31-34; simulation.py `_arrival_loop` | VERIFIED |
| Device 500 Mcycles/s, 0.5 W; edge 8,000 Mcycles/s, 5 ms; cloud 30,000 Mcycles/s, 80 ms; P_tx 0.3 W | cfg:37-51 | VERIFIED |
| B = 1 MHz; SNR −10…30 dB → ≈137 kbit/s … ≈10 Mbit/s | cfg:59-61; `tests/test_simulator.py::test_shannon_endpoints` | VERIFIED |
| Cost = 0.7·latency + 0.3·energy; reward = −cost | cfg:102-103; simulation.py `_handle_task` | VERIFIED |
| μ ≈ 14.5/s, ρ ≈ 0.41 | 1/0.06875 = 14.55; 6 × 0.06875 = 0.4125 (under Always-Edge) | VERIFIED (qualifier added) |
| State: 4×3×3 = 36, bins [1,3,6], [3000,7000], [0.4,0.75]; mobility 324 | cfg:107-120 | VERIFIED; "queue" = tasks in system (edge_server.py:55), wording FIXED |
| "24 with queue below six are reachable" | 3 bins × 9 = 27 | FIXED (F-14) |
| α 0.15, γ 0.90, ε 1.0 → 0.05, × 0.998/episode, 2,000 episodes | cfg:94-99; q_learning_agent.py `end_episode` | VERIFIED; floor reached at ep 1,497 (`test_epsilon_schedule`) |
| 100 s episodes, ~600 updates/episode, ~1.2 M updates | main.py/statistical_tests.py `duration=100`; S coverage total_steps 1,197,168–1,199,429 | VERIFIED |
| Training quality ~U(0.1, 1.0) per episode; eval quality 0.9 | cfg:74-80; simulation.py `run` | VERIFIED |
| Seeds 42,123,456,789,999; seed × 10,000 + episode; disjoint eval | cfg:18-29; statistical_tests.py:23; `test_seed_blocks_are_disjoint` | VERIFIED |
| Q-update Q ← Q + α[r + γ max Q(s′,·) − Q] | q_learning_agent.py `learn()`: max over *tried* actions, 0 if none | FIXED (documented, Algorithm 1) (F-08) |
| Zero (optimistic) init + masked greedy | q_learning_agent.py `_greedy`; `test_greedy_ignores_untried_actions` | VERIFIED |
| Off-policy convergence of the update rule | `test_converges_to_q_star_off_policy` (tiny MDP, Q* by value iteration, atol 1e-3) | VERIFIED |
| No terminal state; 100 s = truncation; always bootstrap | simulation.py (no done flag) | VERIFIED (now stated) |
| "Evaluated greedily (ε = 0)" | the Q-table kept learning during eval | FIXED: `freeze()` (F-01), `test_freeze_stops_learning_and_exploration` |
| Heuristic "observes the same state" | baselines.py `set_context` (raw size, complexity, exact queue, quality) | FIXED (F-02) |
| Heuristic wait estimate = queue × mean service time | baselines.py:87 | VERIFIED |

### Results (Chapter 4)
| Claim (v1) | v2 value / source | Status |
|---|---|---|
| Table 4.1 (0.0870/0.0870, 0.1176/0.1176, 0.1818/0.1811, 0.4000/0.3988), max 0.39 % at ρ = 0.62, 150,428–600,524 tasks | V `mm1_standalone` (re-run, identical) | VERIFIED |
| "Simulator valid basis for experiments" (from M/M/1 only) | V `closed_form_vs_sim`: Edge P-K 0.1039 vs 0.1037 s (0.21 %), Cloud 0.0990 vs 0.0990 s (0.001 %), Local 1.1000 vs 1.0999 s (0.01 %) | FIXED (F-03) + tests |
| Table 4.2 Q-learning 0.0817 s / 0.000195 J / 0.0573 | S `table_4_2_seed42_10000s`: 0.0850 / 0.000195 / 0.0595 | FIXED |
| Table 4.2 Heuristic 0.0784/0.0550; Cloud 0.0990/0.0694; Edge 0.1038/0.0727; Random 0.4290/0.1846/0.3557; Local 1.1016/0.5508/0.9364; ~59,940 tasks | same key (baselines are unaffected by F-01) | VERIFIED |
| Same energy for all four offloading strategies | S; `test_offload_energy_does_not_depend_on_destination` | VERIFIED |
| Q-learning 0.05757 ± 0.00019; Cloud 0.06939 ± 0.00007; Heuristic 0.05508 ± 0.00027 | S t4.3: QL 0.05856 (SD 0.00120); Cloud 0.06939 (SD 0.00006); Heur 0.05508 (SD 0.00021) | FIXED (QL); VERIFIED (baselines) |
| vs Cloud 17.03 % ± 0.22 [16.81, 17.24], per-seed 16.83–17.21 | 15.60 % [13.47, 17.74], per-seed 13.59–17.75 | FIXED |
| vs Edge 21.20 % ± 1.35 | 19.84 % [16.68, 22.99] | FIXED |
| vs Heuristic −4.53 % ± 0.44 [−4.96, −4.09], per-seed −5.00…−4.12 | −6.32 % [−9.08, −3.57], per-seed −8.81…−3.95 | FIXED |
| t(4) = 225.96, p = 2.3e-9; t(4) = −29.92, p = 7.4e-6 | 20.31, 3.5e-5 (Holm 1.0e-4); −6.38, 3.1e-3 (Holm 3.1e-3) | FIXED |
| d = 101.05 / −13.38 and the "likely reason" narrative | d_z = 9.08 / −2.85; the cause was F-01 | FIXED (F-16) |
| t critical 2.776; SciPy cross-check agrees | statistical_tests.py; S `t_scipy` = `t` | VERIFIED |
| Split 70.8 % edge / 29.2 % cloud; never local | seed 42: 78.5 / 21.5 / 0; five-seed 70.3–78.2 % edge | FIXED / VERIFIED (never local) |
| "Only exception: intermediate quality, large tasks" | raw seed_*.json: idle→edge 44/45, 3–5→cloud 45/45, 1–2 band 39 cloud / 6 edge; seed 42 exceptions (1–2, small, ok) and (1–2, large, good) | FIXED (F-15) |
| Convergence by episode 1,364 | S: seed 42 = 1,364; median 1,428; range 1,364–1,602 | VERIFIED (+5-seed) |
| 27/36 states (75.0 %), 81 pairs; unvisited = queue ≥ 6 | S coverage (all 5 seeds); `unvisited_states` | VERIFIED |
| 18 states ≥ 30 visits; Q(local) −1.68…−1.15; Q(edge/cloud) −0.79…−0.52 | S `section_4_4_seed42` | VERIFIED |
| 550 Mcycles → 1.1 s locally | (100+1000)/2 / 500 | VERIFIED |
| Pareto: 7 weightings, energy 1.9439e-4 J identical | raw/pareto (re-run with frozen eval: energies bit-identical) | VERIFIED; 1,500 episodes / 2,000 s now stated (F-11) |
| Pareto latency range | 0.0821–0.0909 s; four weightings learn the same policy | NEW |

### Conclusions (Chapter 5) and other claims
| Claim (v1) | v2 value / source | Status |
|---|---|---|
| Contribution numbers (17.03 %, 21.20 %, −4.53 %, t = 225.96) | as above | FIXED |
| "Queue observable fix raised improvement 8.7 % → 17.4 %, coverage 65/72 → 81/81" | raw/queuefix (frozen eval, seed 42): 9.39 % → 14.18 %; 67/78 → 81/81 state-action pairs | FIXED |
| "eight-line heuristic" | `GreedyHeuristicAgent.act` is 12 lines | FIXED ("short heuristic") |
| Always-Local latency 1.099 s (§5.2) | Table 4.2: 1.1016 s | FIXED (F-17) |
| Device overload ≈ 6.6× | 6 × 1.1 = 6.6 | VERIFIED |
| Mobility coverage 81 of 324 (25.0 %) | raw/mobility.json: 81/324 | VERIFIED (re-run) |
| "preliminary experiments … under way" | no evidence | REMOVED; TODO(HUMAN) (F-18) |
| "no equivalent Q-learning baseline exists" | literature claim | softened (F-20) |
| Figures 4.1–4.4 | regenerated by repro/figures.py from S/raw (Okabe-Ito, 300 dpi + vector PDF) | FIXED |
| Figures 3.1–3.3 (diagrams) | no source files | TODO(HUMAN) F-13 (caption note added) |

### References (verified online 2026-09-19 unless noted)
| Entry | Status |
|---|---|
| Alfakih 2020 (IEEE Access 8, 54074); Bi 2021 (TWC 20(11) 7519); Chen X. 2016 (ToN); Chen & Liu 2022 (Sensors 22(13) 4738); Hwang 2021 (arXiv:2107.05015); Peng 2024 (Comput. Sci. Rev.); Shi 2023 (Sensors 23(17) 7595); Shuai & Xie 2024 (IJCS, doi 10.1002/dac.5691); Sonmez 2018 (ETT); Souza 2023 (FGCS 148); You 2017 (TWC) | VERIFIED (title/venue/DOI) |
| Dash 2026 | VERIFIED; completed with SN Comput. Sci. 7, Art. 99 |
| Sun 2026 | VERIFIED; completed with J. Supercomput. 82(5), Art. 345 |
| Henderson et al. 2018 (AAAI 32(1), doi 10.1609/aaai.v32i1.11694) | ADDED, VERIFIED |
| Mnih 2015; Watkins & Dayan 1992; Lillicrap 2016; Mao 2017 | canonical records, not re-fetched |
| Watkins 1989 (PhD thesis, Cambridge); Sutton & Barto 2018 (MIT Press, 2nd ed.) | ADDED, canonical records, not re-fetched |
| Shuai & Xie 2024 volume/issue | not given in the report; left as is (TODO(HUMAN) optional) |

## 5. Robustness and history checks (new runs, all frozen evaluation)

| Check | Result | Source |
|---|---|---|
| Uplink + propagation simulated in event time (F-09), 5 seeds | vs Always-Cloud +15.79 % [15.34, 16.23]; vs Greedy −4.89 % [−5.13, −4.64]; conclusions unchanged | S `uplink_in_event_time` |
| Old edge observable (waiting tasks only), seed 42, 10,000 s | vs Always-Cloud +9.39 % (fixed observable: +14.18 %); coverage 67/78 → 81/81 | S `queue_observable_history` |
| Mobility (324 states), seed 42, velocity 0.8 | 81/324 states visited (25.0 %) | raw/mobility.json |
| Hyperparameter sensitivity (7 one-at-a-time variants × 5 seeds) | see Table 4.4 / S `sensitivity` | raw/sensitivity |

## 6. Open TODO(HUMAN)

1. **Title change approval** (F-24): the declaration and supervisor certificate now carry the new title
   "Q-Learning for Task Offloading in 3-Tier Device-Edge-Cloud Systems". Confirm that the department/supervisor
   accept the retitle, and re-sign or re-date if required (the declaration date is still 2026-09-11).
2. **Examiners may hold the submitted version.** Its headline numbers (17.03 %, −4.53 %, d = 101) are superseded.
   Be ready to explain the evaluation fix (DEFENSE_PREP.md, "What changed").
3. **Figure 3.1** (workflow diagram, no source in the repo) omits the task-size bin in its "State observation" box.
   The caption notes this; redraw it if time allows (F-13).
4. **Optional citations**: van Hasselt et al. (2016) for the DDQN sentence (F-21); the volume/issue of Shuai &
   Xie (2024) are not in the reference.
5. **"Preliminary experiments already under way"** (F-18) was removed. Reinstate it only with evidence.
6. **Branches**: `origin/fix/step1-simulated-network-time` is superseded by the `UPLINK_IN_EVENT_TIME` flag plus
   the robustness check. Decide whether to delete it. Nothing was pushed from this session (branch `defense-final`
   is local).
7. **Repository name** `rl-task-offloading` is unchanged (renaming would break the clone URL); rename on GitHub
   if desired.
8. **Slides**: none were provided. If a deck exists, update the numbers from DEFENSE_PREP.md.
