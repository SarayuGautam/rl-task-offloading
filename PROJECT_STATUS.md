# Project Status — Overall Note

**Project:** Reinforcement Learning for Task Offloading in 3-tier Device-Edge-Cloud Systems
**Student:** Sarayu Gautam · Kathmandu University, Dept. of CSE · EGPG 600
**Last updated:** 2026-06-20

This is the master status note. It is kept in sync with the code. Whenever the
code changes, update this file and the supervisor `.docx` notes.

---

## Where the project stands

| Phase | Status |
|---|---|
| Proposal | ✅ Submitted (Jan 26, 2026) |
| Simulation environment (SimPy, 3-tier) | ✅ Done, runs ~60k tasks |
| Baseline agents (local/edge/cloud/random) | ✅ Done |
| Q-Learning agent + training loop | ✅ Done, +9.01% over best baseline (Cloud) |
| Statistical validation (5 seeds, 95% CI) | ✅ Done — 9.01% ± 0.41% vs best baseline (Shannon model) |
| Mobility-aware state space (Month 2) | ✅ Done, now reachable from CLI |
| Shannon-Hartley channel model (quality → cost) | ✅ **Done (added 2026-06-20)** |
| **M/M/1 simulation validity check** | ✅ **Done (added 2026-06-20)** — passes within 0.4% |
| Charts (learning curve, comparison, heatmap) | ✅ Done, now reproducible from CLI |
| Pareto latency-energy trade-off sweep | ✅ **Done (added 2026-06-20)** — `--agent pareto`; frontier is weak (see note) |
| Paired t-test on seeds | ❌ Not started (CI exists; t-test planned next) |
| DQN implementation | ❌ Stub only (thesis-phase extension) |
| Thesis document | ❌ Not started (this is EGPG 600 pre-thesis) |

---

## How to run everything (current CLI)

```bash
PY=./venv/bin/python    # the venv was moved; use python -m pip for installs

$PY main.py --agent all                          # all baselines + Q-Learning
$PY main.py --agent qlearning --episodes 2000 --charts   # train + save 3 charts
$PY main.py --agent qlearning --mobility --velocity 0.8  # Month 2 mobility mode
$PY main.py --agent stats                         # 5-seed statistical validation
$PY main.py --agent validate                      # M/M/1 simulation validity check
$PY main.py --agent pareto                        # latency-energy Pareto sweep
```

---

## Changelog

### 2026-06-20 (later) — Shannon channel + Pareto sweep
- **Implemented the Shannon-Hartley channel model** (`network_model.py`,
  `config.py`). Data rate is now `B · log₂(1 + SNR(quality))` instead of a fixed
  rate. This fixes two things at once: (a) the "Shannon's Law" claim in the notes
  is now true, and (b) `network_quality` — previously observed in the state but
  with **zero effect on cost** — now genuinely drives transmission delay & energy,
  which also gives the Month 2 mobility work its effect.
  - Re-ran the 5-seed validation **against the best baseline per seed** (which is
    now Always Cloud, not Always Edge — Shannon makes cloud's cheap-transmission +
    fast-CPU win at good signal). Honest result: **+9.01% ± 0.41%**, CI = [8.60%,
    9.42%], statistically significant. (Against the weaker Always Edge baseline it
    is ~14%, but quoting that would be cherry-picking.)
  - **NOTE on the proposal's "≥10%" target:** met against Always Edge (~14%) but
    NOT against the strongest baseline Always Cloud (9.01%). Flag this honestly to
    the supervisor; the value of the agent is adapting across conditions, and the
    margin over *every* fixed baseline is statistically significant.
- **Fixed phantom citations** in the supervisor notes: "Ali et al. (2025)" and
  "Nieto et al. (2024)" (not in the bibliography) replaced with the real refs —
  Mao 2017 / Souza 2023 (simulation), Alfakih 2020 / Chen & Liu 2022 / Shuai 2024
  (cost function).
- **Added `src/evaluation/pareto.py`** (`--agent pareto`) — sweeps the latency/energy
  weight and plots the trade-off. **Finding:** the frontier is nearly flat. In this
  discrete 3-action setting, latency and energy are *positively coupled* (the slow
  option, local, is also the energy-expensive one; the two viable offload options,
  edge & cloud, have near-identical energy), so there is little genuine trade-off to
  expose. This is an honest, reportable result — and it directly motivates the DRL
  thesis extension, where *continuous* controls (CPU frequency / transmit power)
  create a real latency-energy frontier. See `pareto_curve.png`.

### 2026-06-20 — Pre-meeting refactor
- **Fixed import bug** in `edge_server.py` and `cloud_server.py`
  (`from environment...` → `from src.environment...`). The modules could not be
  imported directly before — it only worked by accident of import order.
- **Removed dead code**: the unused `TaskGenerator` class in `task_generator.py`
  (arrivals are handled by `simulation._arrival_loop`). The `Task` dataclass is kept.
- **Added `src/evaluation/queue_validation.py`** — validates the SimPy queue
  engine against M/M/1 theory at four load levels. This is the validity step the
  literature review (§2.4, §2.8) calls "a required step". Result: max 0.4% error → VALID.
- **Wired existing work into the CLI** (`main.py`):
  - `--mobility` / `--velocity` flags expose the Month 2 mobility state space
  - `--charts` regenerates the 3 thesis charts after training
  - `--agent stats` runs the 5-seed statistical validation
  - `--agent validate` runs the M/M/1 validity check
  Previously these lived only in module files and were run by hand (not reproducible).

---

## ✅ Resolved

- **Title confirmed:** "Reinforcement Learning for Task Offloading in 3-tier
  Device-Edge-Cloud Systems" (per the presentation slide). Drops "Adaptive" and
  "Federated", so the §2.6.4 federated concern no longer applies.
- **Shannon's Law:** now genuinely implemented (rate = B·log₂(1+SNR)); claim is true
  and `network_quality` causally affects cost.
- **Citations:** phantom "Ali/Nieto" refs replaced with real bibliography entries.
- **Pareto curve:** implemented; honest finding is a weak frontier (see changelog).

## ⚠️ Remaining for EGPG 600

1. **Paired t-test** on the 5-seed results (the 95% CI already exists; a t-test gives a
   formal p-value). ~half day.
2. **Final evaluation report + publication-quality charts** — compile the existing
   results (comparison, learning curve, heatmap, M/M/1 table, Pareto, CI) into the
   submission document. ~1–2 days.
3. **Minor wording:** decide whether to keep the near-degenerate Pareto result as an
   honest finding (recommended — it motivates the DRL extension) or extend the energy
   model (e.g., CPU-frequency/DVFS control) to expose a real frontier (arguably
   thesis-phase work).

## ⏭️ Thesis phase (15 credits) — not part of EGPG 600

- DQN implementation (`dqn_agent.py` stub), DQN-vs-Q-Learning comparison, full thesis
  document. Continuous-control DRL (DDPG) is the natural way to expose a real
  latency-energy Pareto frontier.
