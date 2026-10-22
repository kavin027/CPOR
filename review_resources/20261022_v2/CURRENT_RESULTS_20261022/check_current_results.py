"""Portable aggregate check; no cached observations or training required."""
from pathlib import Path
import csv,json,statistics,math,collections,hashlib
H=Path(__file__).resolve().parent
def rows(n):
 with (H/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=1e-9),(a,b)
cells=rows('ALL_CELL_METRICS.csv');keys=['dataset','backbone','window','seed','method']
assert len(set(tuple(r[k] for k in keys) for r in cells))==1947
groups=collections.defaultdict(list)
for r in cells:groups[tuple(r[k] for k in ['dataset','backbone','seed','method'])].append(float(r['rmse']))
g={k:statistics.mean(v) for k,v in groups.items()}
for r in rows('GROUP_MEAN_SD.csv'):
 v=[g[r['dataset'],r['backbone'],i,r['method']] for i in ['42','1','2']];near(r['mean'],statistics.mean(v));near(r['sd'],statistics.stdev(v))
for r in rows('SUMMARY_PAIRED_GAINS.csv'):
 byrun=collections.defaultdict(list);bygroup=collections.defaultdict(list)
 for (ds,bb,i,k),v in g.items():
  if k==r['comparator'] and (r['scope']=='Combined' or bb==r['scope']):
   gain=100*(1-g[ds,bb,i,'CPOR']/v);byrun[i].append(gain);bygroup[ds,bb].append(gain)
 means=[statistics.mean(v) for v in byrun.values()];assert len(means)==3
 near(r['mean_gain_pct'],statistics.mean(means));near(r['sd_gain_pp'],statistics.stdev(means))
 assert int(r['positive_mean_groups'])==sum(statistics.mean(v)>0 for v in bygroup.values())
for r in rows('FOUR_ARM_SUMMARY.csv'):
 byrun=collections.defaultdict(list)
 for (ds,bb,i,k),v in g.items():
  if k==r['comparator'] and (r['scope']=='Combined' or bb==r['scope']):byrun[i].append(100*(1-g[ds,bb,i,r['method']]/v))
 means=[statistics.mean(v) for v in byrun.values()];near(r['mean_gain_pct'],statistics.mean(means));near(r['sd_gain_pp'],statistics.stdev(means))
costs=rows('derived/PROTECTION_COST_RUNS.csv')
for r in costs:
 near(r['cost'],100*(float(r['protected_rmse'])/float(r['plain_rmse'])-1))
for r in rows('derived/PROTECTION_COST_SUMMARY.csv'):
 v=[float(x['cost']) for x in costs if x['method']==r['method']];assert len(v)==36
 near(r['mean'],statistics.mean(v));near(r['median'],statistics.median(v))
for r in rows('derived/RISK_SUMMARY.csv'):
 q=[x for x in rows('derived/RISK_CELLS.csv') if x['method']==r['method']]
 for k in ['node_sequences','prefixes','ever','all_dates','final']:assert int(r[k])==sum(int(x[k]) for x in q)
 for k,num,den in [('ever_pct','ever','node_sequences'),('dates_pct','all_dates','prefixes'),('final_pct','final','node_sequences')]:near(r[k],100*int(r[num])/int(r[den]))
lr=collections.defaultdict(list)
for r in rows('derived/LEVEL_CELLS.csv'):lr[tuple(r[k] for k in ['dataset','backbone','seed','scope','method'])].append(float(r['rmse']))
lr={k:statistics.mean(v) for k,v in lr.items()}
for r in rows('derived/LEVEL_MEAN_SD.csv'):
 v=[lr[r['dataset'],r['backbone'],i,r['scope'],r['method']] for i in ['42','1','2']];near(r['mean'],statistics.mean(v));near(r['std'],statistics.stdev(v))
# Retained 500-update side of the budget comparison (main Table 12).
og=collections.defaultdict(list)
for r in rows('OLD_500_STEP_ALL_RUNS.csv'):
    if r['backbone']=='DLinear':og[r['dataset'],r['seed'],r['method']].append(float(r['rmse']))
og={k:statistics.mean(v) for k,v in og.items()}
for r in rows('OLD_NEW_DLINEAR_COMPARISON.csv'):
    if not r['old_gain_pct']:continue # No matched old result (e.g. ProtectedProCAST).
    vals=[statistics.mean([100*(1-og[ds,i,'CPOR']/v) for (ds,run,k),v in og.items() if run==i and k==r['comparator']]) for i in ['42','1','2']]
    near(r['old_gain_pct'],statistics.mean(vals));near(r['old_sd_pp'],statistics.stdev(vals))
# Group-wise paired means/SDs used in supplementary Table S13.
for r in rows('GROUP_GAIN_MEAN_SD.csv'):
    v=[100*(1-g[r['dataset'],r['backbone'],i,'CPOR']/g[r['dataset'],r['backbone'],i,r['comparator']]) for i in ['42','1','2']]
    near(r['mean'],statistics.mean(v));near(r['sd'],statistics.stdev(v))
for r in rows('derived/RISK_SUMMARY.csv'):
    v=sorted(float(x['final_ratio']) for x in rows('derived/FINAL_NODE_RATIOS.csv') if x['method']==r['method'] and x['final_ratio'] and math.isfinite(float(x['final_ratio'])))
    ix=(len(v)-1)*.95;lo=int(ix);near(r['p95'],v[lo]+(ix-lo)*(v[min(lo+1,len(v)-1)]-v[lo]))
for p,h in json.loads((H/'BUNDLE_HASHES.json').read_text()).items():assert hashlib.sha256((H/p).read_bytes()).hexdigest()==h,p
print(json.dumps(dict(passed=True,metric_rows=len(cells),groups_checked=len(rows('GROUP_MEAN_SD.csv')),paired_summaries=len(rows('SUMMARY_PAIRED_GAINS.csv')),four_arm_summaries=12,level_means_checked=len(rows('derived/LEVEL_MEAN_SD.csv')),risk_methods_checked=13,scope='Aggregate arithmetic and included-file hashes; not raw-data retraining or exact-ledger replay.'),indent=2))
