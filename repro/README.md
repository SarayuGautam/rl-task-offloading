# Reproduction scripts

These scripts regenerate every corrected number, table and figure in the
revised report from the patched code in `src/`. They are not part of the
application; they are the audit trail for where each reported figure came
from. Run them from the repository root (`python repro/01_...py`), not from
inside `repro/`.

All scripts are deterministic given the fixed seeds in `src/config.py`, so
re-running any of them should reproduce the numbers below exactly.

| Script | Produces | Report location |
|---|---|---|
| `01_table_4_2_numbers.py` | Latency/energy/cost for all six strategies on one matched evaluation seed | Table 4.2, Figure 4.1 |
| `02_fixed_channel_and_mobility.py` | The same comparison at the fixed channel, plus a second run under within-episode mobility | Table 4.2 (fixed-channel block); the mobility figures cited in §5.3's future-work item |
| `03a_table_4_3_run_one_seed.py` | Trains and evaluates ONE seed, appends it to `sample_outputs/table_4_3_seed_results.json` | Table 4.3 (run once per seed: 42, 123, 456, 789, 999) |
| `03b_table_4_3_finalize.py` | Confidence intervals, paired t-tests and Cohen's d from all five checkpointed seeds | Table 4.3, §4.3.1-4.3.3 |
| `04_section_4_4_diagnostics.py` | Convergence episode, per-action Q-value ranges, and the greedy policy grid using the same visit-threshold rule as the heatmap | §4.4 prose |
| `05a_pareto_run_one_beta.py` | Trains and evaluates ONE cost-weighting, appends it to `sample_outputs/pareto_sweep_results.json` | Figure 4.4 (run once per weight: 0.00, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00) |
| `05b_pareto_finalize.py` | Assembles the swept table and regenerates the Pareto chart from all seven checkpointed weights | Table/Figure 4.5 discussion, §4.5 |

`sample_outputs/` holds the actual results these scripts produced for this
revision: the four regenerated chart PNGs used in the report, the trained
Q-table, and the two JSON checkpoints (`table_4_3_seed_results.json`,
`pareto_sweep_results.json`) that `03b` and `05b` read.

## Why `03a` and `05a` checkpoint one seed/weight at a time

Training all five seeds (or all seven Pareto weights) back-to-back is a
long-running job. Splitting it into one-seed and one-weight steps that each
append to a JSON checkpoint means the work survives being interrupted and
resumed, and each step finishes in well under a minute. This is exactly how
the numbers in the report were generated. `03b`/`05b` will tell you which
seeds or weights are still missing if you run them early.

## What changed and why (see the main patch for the actual code)

- **`src/environment/edge_server.py`**: `queue_length` now counts the task
  in service, not just the ones waiting. This was the single highest-impact
  fix: it raised the improvement over Always-Cloud from 8.7% to 17.4% on the
  representative seed (Table 4.2) and completed state-action coverage from
  65/72 to 81/81 (§4.4).
- **`src/config.py`**: training seeds are now `seed * 10,000 + episode`
  instead of `seed + episode`, so the five "independent" seeds no longer
  share up to 96% of their training workload.
- **`src/agent/baselines.py`**: adds `GreedyHeuristicAgent`, a fifth,
  untrained but state-aware baseline that turned out to beat the learned
  policy by 4.5% (§3.7, §4.2, §4.3).
- **`main.py`**, **`src/evaluation/pareto.py`**,
  **`src/evaluation/statistical_tests.py`**: wire the two fixes above and
  the new baseline through the CLI, the Pareto sweep, and the five-seed
  statistics; also fixes `run_baseline` evaluating on a different seed than
  `run_qlearning` did (`main.py --agent all` was comparing baselines and the
  agent on different workloads).
- **`requirements.txt`**: adds `scipy`, which `statistical_tests.py` already
  imported but which was missing, so a clean install could not run
  `python main.py --agent stats`.
