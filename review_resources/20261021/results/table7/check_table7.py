"""Rebuild three-experiment RMSE means from supplied configuration aggregates."""
from pathlib import Path
from collections import defaultdict
from statistics import mean,stdev
import csv,json,math
H=Path(__file__).resolve().parent
def read(n):
    with (H/n).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
cells=read('CELL_METRICS.csv');assert len(cells)==330
g=defaultdict(list)
for r in cells:g[tuple(r[k] for k in ['dataset','backbone','seed','method'])].append(float(r['rmse']))
g={k:mean(v) for k,v in g.items()};assert len(g)==180
for r in read('GROUP_EXPERIMENT_METRICS.csv'):
    assert math.isclose(g[tuple(r[k] for k in ['dataset','backbone','seed','method'])],float(r['rmse']),rel_tol=1e-12)
v=defaultdict(list)
for (d,b,s,m),x in g.items():v[d,b,m].append(x)
assert all(len(x)==3 for x in v.values())
for r in read('THREE_EXPERIMENT_RMSE.csv'):
    assert math.isclose(mean(v[tuple(r[k] for k in ['dataset','backbone','method'])]),float(r['mean_rmse']),rel_tol=1e-12)
stats=json.loads((H/'SUMMARY.json').read_text())['summary']
for m,t in stats.items():
    gains=[100*(1-g[d,b,s,'CPOR']/x) for (d,b,s,method),x in g.items() if method==m]
    assert math.isclose(mean(gains),t['mean'],abs_tol=1e-10)
print('PASS: 330 configuration metrics, 180 group/experiment means, 60 three-experiment means.')
