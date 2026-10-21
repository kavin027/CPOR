"""Verify the delivered aggregates without prediction caches or numerical solvers."""
from pathlib import Path
import csv,json,statistics,math
H=Path(__file__).resolve().parent
rows=list(csv.DictReader((H/'CELL_METRICS.csv').open()))
assert len(rows)==264 and len({(r['cell'],r['seed'],r['arm']) for r in rows})==264
assert sum(r['reused']=='False' for r in rows)==96
acc={}
for r in rows:acc.setdefault((r['dataset'],r['backbone'],r['seed'],r['arm']),[]).append(float(r['rmse']))
g={k:statistics.mean(v) for k,v in acc.items()};assert len(g)==144
for r in csv.DictReader((H/'GROUP_EXPERIMENT_METRICS.csv').open()):
 assert math.isclose(float(r['rmse']),g[r['dataset'],r['backbone'],r['seed'],r['arm']],rel_tol=1e-12)
for r in csv.DictReader((H/'THREE_EXPERIMENT_RMSE.csv').open()):
 values=[g[r['dataset'],r['backbone'],s,r['arm']] for s in ['42','1','2']]
 assert int(r['repeats'])==3 and math.isclose(float(r['mean_rmse']),statistics.mean(values),rel_tol=1e-12)
 assert math.isclose(float(r['sd_rmse']),statistics.stdev(values),rel_tol=1e-10,abs_tol=1e-10)
sm=json.loads((H/'SUMMARY.json').read_text())
for name,a,b in [('root_vs_square_full','RootFull','SquareFull'),('root_vs_square_directional','RootDirectional','SquareDirectional'),('full_vs_directional_root','RootFull','RootDirectional'),('full_vs_directional_square','SquareFull','SquareDirectional')]:
 gains={k[:3]:100*(1-v/g[k[:3]+(b,)]) for k,v in g.items() if k[3]==a}
 assert math.isclose(statistics.mean(gains.values()),sm['comparisons'][name]['mean_gain_pct'],abs_tol=1e-10)
 ag={}
 for k,v in gains.items():ag.setdefault(k[:2],[]).append(v)
 assert sum(statistics.mean(v)>0 for v in ag.values())==sm['comparisons'][name]['positive_groups']
 assert sum(min(v)>0 for v in ag.values())==sm['comparisons'][name]['positive_all_three']
assert sm['reused']==168 and sm['new']==96 and sm['trajectories']==264
print('PASS: 264 trajectories; all three-experiment averages, dispersions and four paired comparisons.')
