import json
from src.evaluation.pareto import BETAS, _plot, _save_csv

data = json.load(open("repro/sample_outputs/pareto_sweep_results.json"))
rows = [data[f"{b:.2f}"] for b in BETAS]
for r in rows:
    print(f"w_lat={r['w_latency']:.2f}  w_eng={r['w_energy']:.2f}  "
          f"latency={r['avg_latency']:.5f}s  energy={r['avg_energy']:.8f}J")

energies = [r["avg_energy"] for r in rows]
print(f"\nenergy min={min(energies):.10f}  max={max(energies):.10f}  "
      f"spread={max(energies)-min(energies):.2e} J")

_save_csv(rows, "repro/sample_outputs/pareto_data.csv")
_plot(rows, "repro/sample_outputs/charts/pareto_curve.png")
