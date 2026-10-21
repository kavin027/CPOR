"""Rebuild six-dataset summary and four-arm table from group RMSEs.
Standard library only. This checks aggregation, not forecast generation.
"""
from pathlib import Path
import csv,json,statistics,math
O=Path(__file__).resolve().parent
def read(name):
    with (O/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
rows=read('paired_group_seed_gains.csv');out={}
for ref in ['ProCAST72Warm','ProtectedMLPol','ProtectedCalBest','SquareFull','CovFullFix']:
    r=[x for x in rows if x['reference']==ref];assert len(r)==36
    for x in r:assert math.isclose(float(x['gain_pct']),100*(1-float(x['c0_rmse'])/float(x['reference_rmse'])),abs_tol=1e-10)
    per_run=[statistics.mean(float(x['gain_pct']) for x in r if int(x['seed'])==k) for k in [42,1,2]]
    assert all(sum(int(x['seed'])==k for x in r)==12 for k in [42,1,2])
    out[ref]=dict(mean_pct=statistics.mean(per_run),sd_between_experiment_means=statistics.stdev(per_run),experiment_means=per_run)
assert round(out['ProCAST72Warm']['mean_pct'],2)==28.47
assert round(out['SquareFull']['mean_pct'],2)==.36
g=read('ablation_group_metrics.csv');cells={(x['dataset'],x['backbone'],x['method']):float(x['rmse']) for x in g}
keys=sorted({(x['dataset'],x['backbone']) for x in g});assert len(keys)==10
arms={a:statistics.mean(100*(1-cells[(d,b,a)]/cells[(d,b,'SquareFull')]) for d,b in keys) for a in ['RootFull','RootDirectional','SquareFull','SquareDirectional']}
geometry=statistics.mean(100*(1-cells[(d,b,'RootFull')]/cells[(d,b,'RootDirectional')]) for d,b in keys)
saved=json.loads((O/'ablation_summary.json').read_text())
for r in saved:assert math.isclose(arms[r['arm']],r['mean'],abs_tol=1e-10)
assert round(geometry,4)==-.0144
result=dict(method='CPOR',full_name='Cumulatively Protected Online Reconciliation',
            six_dataset_summary=out,four_arm_common_square_full=arms,
            arm_display_names={'RootFull':'CPOR (root/full)','RootDirectional':'CPOR (root/directional)',
                               'SquareFull':'Squared-risk (full)','SquareDirectional':'Squared-risk (directional)'},
            root_full_vs_root_directional_pct=geometry,
            scope='Group arithmetic only; archived arm keys are retained for reproducibility; no new forecasts, fitting or training.')
(O/'AGGREGATE_CHECK.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
