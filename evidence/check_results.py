"""Current Table 5/7/8 arithmetic checks. Standard library only; no training."""
from pathlib import Path
import runpy,json,hashlib,csv,statistics,math
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'review_resources/20261021'
for n in [5,7,8]:runpy.run_path(str(D/f'results/table{n}/check_table{n}.py'),run_name='__main__')
mapping=json.loads((D/'SOURCE_AND_RESULTS_MAP.json').read_text())
for path,meta in mapping['files'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==meta['sha256'],path
for path,h in json.loads((D/'SHA256.json').read_text()).items():
    assert hashlib.sha256((D/path).read_bytes()).hexdigest()==h,path
current={}
for n in [5,7,8]:current[f'table{n}']=json.loads((D/f'results/table{n}/SUMMARY.json').read_text())
rows=list(csv.DictReader((D/'results/table8/PAIRED_GAINS.csv').open()))
assert len(rows)==144
for name,comp in current['table8']['comparisons'].items():
    vals=[r for r in rows if r['comparison']==name]
    assert len(vals)==36
    for sid,x in comp['experiment_means'].items():
        assert math.isclose(statistics.mean(float(r['gain_pct']) for r in vals if r['seed']==sid),x,abs_tol=1e-10)
# Cross-check current shared methods against the previously public six-dataset CSV.
legacy=list(csv.DictReader((ROOT/'evidence/paired_group_seed_gains.csv').open()))
for ref in ['ProCAST72Warm','ProtectedMLPol','ProtectedCalBest','SquareFull','CovFullFix']:
    vals=[r for r in legacy if r['reference']==ref];assert len(vals)==36
    for r in vals:assert math.isclose(float(r['gain_pct']),100*(1-float(r['c0_rmse'])/float(r['reference_rmse'])),abs_tol=1e-10)
    mean=statistics.mean(float(r['gain_pct']) for r in vals)
    if ref=='ProCAST72Warm':expect=current['table5']['primary_summary']['ProCAST']['mean']
    elif ref=='SquareFull':expect=current['table8']['comparisons']['root_vs_square_full']['mean_gain_pct']
    else:continue
    assert math.isclose(mean,expect,abs_tol=1e-10)
out=dict(version='CPOR-review-resources-v2-20261021',current_tables=[5,7,8],
         four_arm_scope='6 datasets / 12 groups / 3 runs / 264 trajectories',
         source_and_resource_hashes_verified=True,new_training=0,new_predictions=0)
(ROOT/'evidence/AGGREGATE_CHECK.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
