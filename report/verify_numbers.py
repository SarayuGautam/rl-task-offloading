"""
Independent re-verification of the report's headline numbers (Phase 6).

Recomputes the key statistics directly from the raw per-seed checkpoints in
results/raw/ (not from summary.json) and checks that each value appears, with the
same rounding, in the text of report/EGPG600_Report_SarayuGautam_v2.docx. It also
checks that stale numbers from the first version appear only in the explicitly
labelled "before the correction" sentences.

    python report/verify_numbers.py
"""
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
DOCX = os.path.join(ROOT, "report", "EGPG600_Report_SarayuGautam_v2.docx")
SEEDS = [42, 123, 456, 789, 999]

text = subprocess.run(["pandoc", DOCX, "-t", "plain", "--wrap=none"], capture_output=True, text=True,
                      check=True).stdout
text = text.replace("−", "-")
runs = {s: json.load(open(os.path.join(RAW, f"seed_{s}.json"))) for s in SEEDS}
cost = {n: np.array([runs[s]["eval_1000s"][n]["cost"] for s in SEEDS])
        for n in runs[42]["eval_1000s"]}
fails = []


def check(label, value_str):
    ok = value_str.replace("−", "-") in text
    print(f"  [{'OK ' if ok else 'MISSING'}] {label}: {value_str}")
    if not ok:
        fails.append(label)


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


print("Five-seed comparisons (recomputed from results/raw):")
for ref, lab in (("Always-Cloud", "cloud"), ("Always-Edge", "edge"), ("Greedy Heuristic", "heuristic")):
    imp = (cost[ref] - cost["Q-learning"]) / cost[ref] * 100
    m, lo, hi = ci(imp)
    t, p = stats.ttest_rel(cost[ref], cost["Q-learning"])
    check(f"mean improvement vs {lab}", f"{m:.2f}%")
    check(f"95% CI vs {lab}", f"[{lo:.2f}%, {hi:.2f}%]")
    check(f"t(4) vs {lab}", f"t(4) = {t:.2f}")
qm = cost["Q-learning"]
check("Q-learning cost mean ± SD", f"{qm.mean():.5f} ± {qm.std(ddof=1):.5f}")
t42 = runs[42]["eval_table42"]
for n, r in t42.items():
    check(f"Table 4.2 {n} latency", f"{r['latency']:.4f}")
    check(f"Table 4.2 {n} cost", f"{r['cost']:.4f}")
check("seed-42 edge share", f"{t42['Q-learning']['actions_pct']['Edge']:.1f}% of tasks to the edge")
conv = [runs[s]["convergence_episode"] for s in SEEDS]
check("convergence seed 42", f"{conv[0]:,}")
check("convergence median", f"median {int(np.median(conv)):,}")
cov = [runs[s]["coverage"]["states_visited"] for s in SEEDS]
assert len(set(cov)) == 1
check("coverage", f"{cov[0]} of the 36 states")
par = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, "pareto", "*.json")))]
e = {r["avg_energy"] for r in par}
assert len(e) == 1, "Pareto energies differ"
check("Pareto energy", f"{e.pop() * 1e4:.4f} × 10")
mob = json.load(open(os.path.join(RAW, "mobility.json")))["coverage"]["states_visited"]
check("mobility coverage", f"{mob} of the 324 states")
val = json.load(open(os.path.join(ROOT, "results", "validation.json")))
for n, v in val["closed_form_vs_sim"].items():
    check(f"closed-form {n} simulated", f"{v['sim_mean_s']:.4f} s")

print("\nStale v1 numbers (allowed only in the labelled 'before the correction' sentences):")
for stale in ["225.96", "2.3 × 10⁻⁹", "0.05757", "70.8%", "29.2%", "21.20%", "−13.38", "-13.38", "1.099 s",
              "the 24 with", "0.0817", "0.0573"]:
    hits = [m.start() for m in re.finditer(re.escape(stale.replace("−", "-")), text)]
    status = "OK " if not hits else "FOUND"
    print(f"  [{status}] {stale!r}: {len(hits)} occurrence(s)")
    if hits:
        fails.append(f"stale {stale}")
for allowed in ["17.03%", "101.05", "4.53%"]:
    n = text.count(allowed)
    print(f"  [info] {allowed!r} appears {n}x (must only be in the evaluation-correction notes)")

print("\nRESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
