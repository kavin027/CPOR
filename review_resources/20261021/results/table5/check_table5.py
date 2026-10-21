"""Rebuild Table 5 from supplied per-configuration aggregates (not predictions)."""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import csv, json, math

ROOT = Path(__file__).resolve().parent
def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
cells = read("CELL_METRICS.csv")
assert len(cells) == 396
group = defaultdict(list)
for r in cells:
    group[tuple(r[k] for k in ("dataset", "backbone", "seed", "method"))].append(float(r["rmse"]))
group = {k: mean(v) for k, v in group.items()}
assert len(group) == 216
for r in read("GROUP_EXPERIMENT_METRICS.csv"):
    k = tuple(r[x] for x in ("dataset", "backbone", "seed", "method"))
    assert math.isclose(group[k], float(r["rmse"]), rel_tol=1e-12)
averages = defaultdict(list)
for (d, b, seed, method), value in group.items():
    averages[d, b, method].append(value)
assert all(len(v) == 3 for v in averages.values())
averages = {k: mean(v) for k, v in averages.items()}
for r in read("THREE_EXPERIMENT_RMSE.csv"):
    k = tuple(r[x] for x in ("dataset", "backbone", "method"))
    assert math.isclose(averages[k], float(r["mean_rmse"]), rel_tol=1e-12)
summary = json.loads((ROOT / "SUMMARY.json").read_text())
for method in summary["primary_summary"]:
    gains = [100 * (value - group[d, b, seed, "CPOR"]) / value
             for (d, b, seed, m), value in group.items() if m == method]
    assert math.isclose(mean(gains), summary["primary_summary"][method]["mean"], abs_tol=1e-10)
groups = {(d, b) for d, b, m in averages}
best = sum(averages[d, b, "CPOR"] == min(v for (dd, bb, _), v in averages.items()
           if (dd, bb) == (d, b)) for d, b in groups)
assert best == 10
print("PASS: 396 configuration metrics, 216 experiment/group means, 72 table means; CPOR lowest in 10/12 rows.")
