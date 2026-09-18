# Project Status — Overall Note

**Project:** Reinforcement Learning for Task Offloading in 3-tier Device-Edge-Cloud Systems
**Student:** Sarayu Gautam · Kathmandu University, Dept. of CSE · EGPG 600
**Last updated:** 2026-09-12 (final verification pass before the defense)

This is the master status note. It is kept in sync with the code. Whenever the
code changes, update this file. For the defense, read `DEFENSE_GUIDE.md` in the
workspace root — it is the plain-language presentation document.

Every number in the "Verified results" section below was re-derived from the code
and the saved artifacts on 2026-09-01 **and re-verified end to end on 2026-09-12**
(stats re-run and Pareto sweep re-run; both reproduce the report exactly).

---

## Where the project stands

| Phase | Status |
|---|---|
| Proposal | ✅ Submitted (Jan 26, 2026) |
| Simulation environment (SimPy, 3-tier) | ✅ Done, runs ~60k tasks |
| Baseline agents (local/edge/cloud/random) | ✅ Done |
| Q-Learning agent + training loop | ✅ Done |
| Shannon-Hartley channel model (quality → cost) | ✅ Done (2026-06-20) |
| M/M/1 simulation validity check | ✅ Done — max error 0.39% → VALID |
| Full-range quality training + untried-action masking | ✅ Done (commit `667ea29`) |
| Statistical validation (5 seeds, 95% CI) | ✅ Done — **+9.13% ± 1.05%** vs Always Cloud (re-verified 2026-09-12) |
| **Paired t-test** | ✅ **Done** — t(4) = 24.399, p = 0.00002, Cohen's d_z = 10.91 |
| Mobility-aware state space (Month 2) | ⚠️ Implemented and CLI-reachable, **never run for results** |
| Charts (learning curve, comparison, heatmap) | ✅ Done, reproducible via `--charts` |
| Pareto latency-energy trade-off sweep | ✅ **Done — re-run 2026-09-12 at all 7 weightings; confirms degenerate frontier, energy constant at 1.94 × 10⁻⁴ J** |
| DQN implementation | ❌ Stub only (thesis-phase extension) |
| Final report | ✅ Delivered — `EGPG600_Report_SarayuGautam.docx` (2026-09-11; §4.4 corrected 2026-09-12) |

---

## Verified results (re-derived 2026-09-01)

### Headline — 5-seed statistical validation (`--agent stats`)

Re-ran end to end; reproduces the reported figures exactly.

| Quantity | Value |
|---|---|
| Q-Learning composite cost | 0.06300 ± 0.00083 |
| Best baseline (Always Cloud, all 5 seeds) | 0.06933 ± 0.00012 |
| **Improvement vs best baseline** | **+9.13% ± 1.05%** |
| 95% CI on improvement | [+8.08%, +10.18%] — contains the 10% target |
| Paired t-test | t(4) = 24.399, p = 0.00002 (two-tailed) |
| Cohen's d_z | 10.912 |
| Per-seed improvement | +9.4, +8.8, +10.0, +9.5, +7.8 % |

Protocol: `SEEDS = [42, 123, 456, 789, 999]`, `TRAIN_EPISODES = 2_000`,
`EVAL_DURATION = 1_000`, `EVAL_SEED_OFFSET = 100_000` (evaluation workload is
disjoint from every training episode).

**Supersedes the old +9.01% ± 0.41% figure** in previous versions of this file.
That number predates commit `667ea29` (full-range quality training + masking of
untried actions) and must not be quoted anywhere.

### Learned policy and coverage (`experiments/results/qtables/qlearning_trained.npz`)

| Quantity | Value |
|---|---|
| States visited | 24 / 36 (66.7%) |
| State-action pairs visited | 65 / 72 |
| Fully explored states (all 3 actions tried) | 19 |
| Final epsilon | 0.05 |
| Greedy policy over visited states | 16 → Cloud, 8 → Edge, **0 → Local** |

Breakdown by edge-queue bin (`QUEUE_BINS = [1, 3, 6]`):

| Queue bin | States | Visits per tried action | Q(local) |
|---|---|---|---|
| 0 (queue < 1) | 9 | 6,401 – 144,111 | −1.742 … −1.361 |
| 1 (1 ≤ queue < 3) | 9 | 210 – 9,099 | −1.655 … −1.404 |
| 2 (3 ≤ queue < 6) | 6 | **1 – 3** (not meaningful) | −0.127 |
| 3 (queue ≥ 6) | 0 | never reached (ρ ≈ 0.41) | — |

Q(edge) and Q(offload) over bins 0–1: −0.978 … −0.590. Local is strictly
dominated across the whole trained region and is never selected.

---

## ✅ Resolved divergences between the earlier report text and the saved artifacts

Found 2026-09-01 by recomputing from `qlearning_trained.npz`; re-confirmed
2026-09-12. **All four items below were corrected in the report itself on
2026-09-12**, after the report was replaced by an editable
`EGPG600_Report_SarayuGautam.docx`. The list is kept for the record — only an
old printout of the 2026-09-11 PDF would still carry the earlier wording.

1. **Count of barely-visited states.** §4.4 said "three visited states with a
   queue length between three and six... reached only one to three times each."
   The artifact has **six** such states (queue bin 2), each visited 1–3 times.
2. **Visit range for the well-trained region.** §4.4 said "roughly 3,000 and
   144,000 times per action." Across both qualifying bins (0 and 1) the true
   span is **210 – 144,111**. The ~3,000 floor holds only for queue bin 0 alone,
   where the true minimum is 6,401.
3. **Q(local) range.** §4.4 said Q(local) "lies between −1.36 and −1.63." The
   true range over the well-trained bins is **−1.361 to −1.742**.
4. **Unvisited-state claim.** §4.4 said "The 12 unvisited states are precisely
   those with an edge queue length of six or more" — arithmetically impossible
   given 24/36 visited (9 + 9 + 6 by queue bin): **nine** of the 12 unvisited
   states have queue ≥ 6; the other three lie in the 3–6 band alongside the six
   barely-visited states.

The former item on the stale four-point Pareto CSV is **resolved**: re-running
`--agent pareto` on 2026-09-12 regenerated `pareto_data.csv` and
`pareto_curve.png` at all seven weightings. Energy is constant at
1.9433 × 10⁻⁴ J — i.e. **1.94 × 10⁻⁴ J, exactly as the report quotes**.

The qualitative claims in §4.4 and §4.5 were never in question — local really is
never selected, and energy really is invariant over the realised action set.

---

## Scope note: what this project does NOT establish

Recorded because it was over-claimed in an earlier draft of report §2.3.

The project does **not** characterise a transition point between tabular
Q-Learning and neural approximation, and does **not** establish a performance
ceiling for Q-Learning. The evidence:

- `src/agent/dqn_agent.py` is a stub. `act()` returns a uniform random action,
  `_build_network()` and `_train_step()` raise `NotImplementedError`, and the
  class is not registered in `main.py`, so it cannot be run.
- There is no state-space scaling sweep. Only two state configurations exist
  (36-state base, 324-state `--mobility`), and only the base one has ever been
  run for results — every saved Q-table has 3-dimensional state keys.
- There is no oracle or optimal-policy upper bound in `src/agent/baselines.py`;
  the comparison set is four fixed heuristics plus random. Without an upper
  bound, the 9.13% figure is a measurement at one operating point, not a ceiling.

This claim is in scope for the thesis phase, not for EGPG 600.

---

## Repository state

- Branch is clean except for **one uncommitted change to `main.py`**: the final
  evaluation run and the comparison-bar chart now use
  `RANDOM_SEED + EVAL_SEED_OFFSET` instead of `RANDOM_SEED`, so the evaluation
  workload is disjoint from training. This is a methodological fix that the
  statistical validation already applied; commit it so the charts and the stats
  path agree.
- `experiments/results_BACKUP_2026-08-17/` holds pre-fix artifacts (18-state
  Q-tables from before the coverage fix). Keep for provenance, do not quote.

---

## Changelog

### 2026-09-12 — Final verification pass (defense readiness)
- **Corrected §4.4 of the report** (the 2026-09-11 PDF was replaced by an
  editable `EGPG600_Report_SarayuGautam.docx`; four sentences fixed against the
  Q-table artifact): "three" barely-visited states → **six**; visits "roughly
  3,000–144,000" → **210–144,111**; Q(local) "−1.36 to −1.63" → **−1.36 to
  −1.74**; "the 12 unvisited states are precisely those with queue ≥ 6" →
  **nine of the 12** (the rest lie in the 3–6 band, as 24/36 = 9+9+6 requires).
  Verified by pandoc diff that no other text changed, and by a LibreOffice
  render that the document still opens cleanly (42 pages).
- Re-ran `--agent stats` end to end: reproduces +9.13% ± 1.05%, t(4) = 24.399,
  p = 0.00002, Cohen's d_z = 10.912, and all five per-seed improvements
  (+9.4, +8.8, +10.0, +9.5, +7.8%) exactly.
- Re-ran `--agent pareto`: regenerated `pareto_data.csv` and `pareto_curve.png`
  at all seven weightings. Energy is constant at 1.9433 × 10⁻⁴ J → the report's
  1.94 × 10⁻⁴ figure and "degenerate frontier" claim are confirmed against the
  code. The Aug 18 four-row CSV was stale and is now replaced.
- Re-derived the §4.4 coverage and Q-value ranges from `qlearning_trained.npz`;
  the three §4.4 text divergences listed above stand (six barely-visited states,
  210–144,111 visits, Q(local) −1.361 to −1.742).
- Cleaned the workspace: removed the old defense briefing, the master-note
  `.docx` versions and the Archive folder (all superseded), stale April chart
  PNGs, the Trello migration JSON, and LaTeX/`__pycache__`/`.DS_Store` junk.
  Superseded everything with `DEFENSE_GUIDE.md` (workspace root).

### 2026-09-01 — Verification pass
- Re-ran `--agent stats`; confirmed +9.13% ± 1.05%, t(4) = 24.399, p = 0.00002,
  Cohen's d_z = 10.912. Updated the headline in this file from the stale
  +9.01% ± 0.41%.
- Marked the paired t-test complete (it is implemented in
  `statistical_tests.py:paired_t_test`, with a scipy `ttest_rel` cross-check).
- Recomputed coverage and Q-value ranges from the saved Q-table; logged four
  numeric discrepancies against the report text (see Discrepancies above).
- Found the Pareto CSV/PNG on disk to be a stale four-point sweep while the code
  specifies seven weightings.
- Recorded the scope note on the tabular-to-DRL transition claim.

### 2026-06-20 (later) — Shannon channel + Pareto sweep
- **Implemented the Shannon-Hartley channel model** (`network_model.py`,
  `config.py`). Data rate is now `B · log₂(1 + SNR(quality))` instead of a fixed
  rate, so `network_quality` genuinely drives transmission delay and energy.
- Re-ran the 5-seed validation against the best baseline per seed, which is
  Always Cloud rather than Always Edge. Against the weaker Always Edge baseline
  the improvement is ~13%, but the report quotes the harder Always Cloud figure.
- **Added `src/evaluation/pareto.py`** (`--agent pareto`). Finding: the frontier
  is **degenerate**, not merely flat. Device energy under offloading is uplink
  transmission energy, which does not depend on the destination; since local is
  strictly dominated and never selected, energy is invariant over every action
  the agent takes. No weighting can move the operating point. This motivates the
  continuous-control DRL extension.
- **Fixed phantom citations** in the supervisor notes ("Ali et al. 2025",
  "Nieto et al. 2024") — replaced with Mao 2017 / Souza 2023 (simulation) and
  Alfakih 2020 / Chen & Liu 2022 / Shuai 2024 (cost function).

### 2026-06-20 — Pre-meeting refactor
- **Fixed import bug** in `edge_server.py` and `cloud_server.py`
  (`from environment...` → `from src.environment...`).
- **Removed dead code**: the unused `TaskGenerator` class in `task_generator.py`.
- **Added `src/evaluation/queue_validation.py`** — validates the SimPy queue
  engine against M/M/1 theory at four load levels. Max 0.4% error → VALID.
- **Wired existing work into the CLI** (`main.py`): `--mobility` / `--velocity`,
  `--charts`, `--agent stats`, `--agent validate`.

---

## How to run everything (current CLI)

```bash
PY=./venv/bin/python    # the venv was moved; use python -m pip for installs

$PY main.py --agent all                                  # all baselines + Q-Learning
$PY main.py --agent qlearning --episodes 2000 --charts   # train + save 3 charts
$PY main.py --agent qlearning --mobility --velocity 0.8  # Month 2 mobility mode
$PY main.py --agent stats                                # 5-seed validation + t-test
$PY main.py --agent validate                             # M/M/1 validity check
$PY main.py --agent pareto                               # latency-energy Pareto sweep
```

---

## ✅ Remaining for EGPG 600 — all closed

1. ~~Re-run `--agent pareto`~~ — done 2026-09-12; confirms 1.94 × 10⁻⁴ J as
   quoted in §4.5.
2. ~~Correct the §4.4 figures~~ — done 2026-09-12 in the report DOCX (see
   "Resolved divergences" above; the Pareto constant had turned out correct
   all along).
3. ~~Commit the `main.py` eval-seed change~~ — committed as `6258152`.
4. ~~Final report~~ — delivered 2026-09-11.

**Only item left: the defense itself.** Prepare from `DEFENSE_GUIDE.md`.

## ⏭️ Thesis phase (15 credits) — not part of EGPG 600

- DQN implementation (`dqn_agent.py` stub) and DQN-vs-Q-Learning comparison.
- Characterising the tabular-to-DRL transition point and the Q-Learning
  performance ceiling — requires a state-space scaling sweep and an optimal or
  oracle upper bound, neither of which exists in this codebase.
- Continuous-control DRL (DDPG) to expose a genuine latency-energy frontier.
- Multi-agent RL; federated learning.
