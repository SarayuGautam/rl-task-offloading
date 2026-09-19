# Reproduction

`python repro/reproduce_all.py` regenerates every number, table and figure in the report
(`report/EGPG600_Report_SarayuGautam_v2.docx`) from the code in `src/`. Run it from anywhere; it
changes to the repository root itself.

| Stage | What it produces | Report location |
|---|---|---|
| `validate` | Standalone M/M/1 check; closed-form (M/G/1 Pollaczek-Khinchine, cloud, local) checks of the full simulator → `results/validation.json` | §4.1, Table 4.1 |
| `main` | 5 seeds × (train 2,000 episodes → freeze → evaluate Q-learning + 5 baselines, 1,000 s); seed 42 also 10,000 s → `results/raw/seed_*.json` | Tables 4.2, 4.3; §4.4; Figs 4.1–4.3 |
| `pareto` | 7 cost weightings (seed 42, 1,500 episodes, 2,000 s eval) → `results/raw/pareto/` | §4.5, Fig 4.4 |
| `mobility` | Coverage of the 324-state mobility space (seed 42, velocity 0.8) | §5.3 |
| `sensitivity` | One-at-a-time α ∈ {0.05, 0.30}, γ ∈ {0, 0.5, 0.99}, ε-decay ∈ {0.995, 0.999}; 5 seeds each | §4.6, Table 4.4 |
| `uplink` | Robustness: uplink/propagation simulated in event time (`UPLINK_IN_EVENT_TIME=True`), 5 seeds | §3.3, §4.7 |
| `queuefix` | History check: old edge observable (waiting tasks only), seed 42, frozen evaluation | §5.1 |
| `finalize` | Statistics (mean, SD, 95% t-CI, paired t-tests, Holm correction, Cohen's d_z) → `results/summary.json` | all |
| `figures` | `results/figures/*.png` (300 dpi, embedded in the .docx) and `*.pdf` (vector) | Figs 4.1–4.4 |

Every training run is checkpointed in `results/raw/`. An interrupted run resumes where it stopped;
delete a checkpoint to force that run again. All runs are deterministic given the seeds.

The earlier per-table scripts (`01_…` to `05b_…`) and `sample_outputs/` were removed on 2026-09-19.
They evaluated the agent while it was still learning (AUDIT.md F-01), and they could not be run as
documented (F-04). Both remain available in git history.
