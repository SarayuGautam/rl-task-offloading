"""Finalize Table 4.3 from repro/sample_outputs/table_4_3_seed_results.json (5 disjoint seeds)."""
import json
from src.evaluation.statistical_tests import SEEDS, confidence_interval, paired_t_test

CKPT = "repro/sample_outputs/table_4_3_seed_results.json"

if __name__ == "__main__":
    data = json.load(open(CKPT))
    missing = [s for s in SEEDS if str(s) not in data]
    if missing:
        raise SystemExit(f"missing seeds: {missing}")

    cols = {k: [data[str(s)][k] for s in SEEDS]
            for k in ("Q-Learning", "Always Cloud", "Always Edge", "Greedy Heuristic")}

    print(f"{'Seed':<8}" + "".join(f"{k:<18}" for k in cols))
    for i, s in enumerate(SEEDS):
        print(f"{s:<8}" + "".join(f"{cols[k][i]:<18.5f}" for k in cols))

    print("\nMean +/- 95% CI (composite cost):")
    means = {}
    for k, v in cols.items():
        m, ci = confidence_interval(v)
        means[k] = (m, ci)
        print(f"  {k:<18} {m:.5f} +/- {ci:.5f}")

    ql = cols["Q-Learning"]
    print("\n" + "=" * 70)
    print("  TABLE 4.3 (revised): Q-Learning vs each reference, 5 disjoint seeds")
    print("=" * 70)
    summary = {}
    for ref in ("Always Cloud", "Always Edge", "Greedy Heuristic"):
        base = cols[ref]
        imps = [(b - q) / b * 100 for q, b in zip(ql, base)]
        m, ci = confidence_interval(imps)
        t = paired_t_test(ql, base)
        print(f"\n  vs {ref}")
        print(f"    per-seed improvement %: {[round(i,2) for i in imps]}")
        print(f"    mean improvement:  {m:+.2f}% +/- {ci:.2f}%   "
              f"95% CI [{m-ci:+.2f}%, {m+ci:+.2f}%]")
        print(f"    paired t-test:     t({t['df']}) = {t['t_stat']:.3f}   "
              f"p = {t['p_value']:.6f}  (scipy: t={t['t_scipy']:.3f}, p={t['p_scipy']:.6f})")
        print(f"    Cohen's d_z:       {t['cohens_dz']:.2f}")
        print(f"    verdict:           {'SIGNIFICANT' if t['significant'] else 'NOT significant'}"
              f"  ({'QL cheaper' if m > 0 else 'QL more expensive'})")
        summary[ref] = dict(imp_mean=m, imp_ci=ci, t=t['t_stat'], p=t['p_value'],
                            d=t['cohens_dz'], per_seed=imps)

    json.dump({"means": {k: v for k, v in means.items()}, "vs": summary},
              open("repro/sample_outputs/table_4_3_final.json", "w"), indent=1, default=str)
