from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;O=H.parent;R=O.parent
P=O/'kais_presentation_revision_20261018'
cells=json.loads((O/'kais_review_response_20261016/Online_Resource_1/cells.json').read_text())
index=json.loads((H/'RESULT_INDEX.json').read_text());assert len(index)==132
lookup={(r['cell'],r['seed'],r['method']):r for r in index}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files if k!='dates'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];inputs={}
for c in cells:
    for seed in [42,1,2]:
        d=O/'c0_seed_validation_20261016/results'/c['dataset']/c['backbone']/c['window']/str(seed)
        ep=d/'endpoints.npz';x=load(ep);y=x['truth'][-72:];raw=x['raw'][-72:]
        paths={'CPOR':d/'C0.npz','ProCAST':d/'ProCAST72Warm.npz',
               'Ada-Y':Path(lookup[c['file'],seed,'Ada-Y']['path']),
               'PIR':Path(lookup[c['file'],seed,'PIR']['path'])}
        preds={'Raw':raw,'NNLS':x['b'][-72:]}
        inputs[str(ep)]=sha(ep)
        for m,p in paths.items():
            a=load(p);preds[m]=a['pred']
            if 'truth' in a:np.testing.assert_array_equal(y,a['truth'])
            if 'raw' in a:np.testing.assert_array_equal(raw,a['raw'])
            inputs[str(p)]=sha(p)
        for m,pred in preds.items():
            assert pred.shape==y.shape and np.isfinite(pred).all()
            rows.append(dict(dataset=c['dataset'],backbone=c['backbone'],window=c['window'],
                seed=seed,method=m,rmse=float(np.sqrt(((pred-y)**2).mean(0)).mean()),
                mae=float(np.abs(pred-y).mean()),nodes=y.shape[1],observations=len(y)))
f=pd.DataFrame(rows);assert len(f)==396
g=f.groupby(['dataset','backbone','seed','method'],as_index=False)[['rmse','mae']].mean()
assert len(g)==216
summary=g.groupby(['dataset','backbone','method']).agg(mean_rmse=('rmse','mean'),sd_rmse=('rmse','std'),
                                                      repeats=('rmse','size')).reset_index()
assert (summary.repeats==3).all()
gain=[]
for (ds,bb,seed),z in g.groupby(['dataset','backbone','seed']):
    v=z.set_index('method').rmse
    for m in ['ProCAST','Ada-Y','PIR','Raw','NNLS']:
        gain.append(dict(dataset=ds,backbone=bb,seed=int(seed),comparator=m,
                         gain_pct=100*(1-v['CPOR']/v[m]),
                         external_vs_raw_pct=100*(1-v[m]/v['Raw'])))
gains=pd.DataFrame(gain);totals={}
for m,z in gains.groupby('comparator'):
    runs=z.groupby('seed').gain_pct.mean();groups=z.groupby(['dataset','backbone']).gain_pct.mean()
    totals[m]=dict(mean=float(runs.mean()),experiment_sd=float(runs.std()),
        per_experiment={str(int(k)):float(v) for k,v in runs.items()},
        positive_mean_groups=int((groups>0).sum()),
        all_three_positive=int(z.groupby(['dataset','backbone']).gain_pct.min().gt(0).sum()),
        external_vs_raw_mean=float(z.external_vs_raw_pct.mean()),
        external_vs_raw_positive=int(z.groupby(['dataset','backbone']).external_vs_raw_pct.mean().gt(0).sum()))
# Independently match the previously published aggregate source.
old=pd.read_csv(P/'Review_Evidence_20261020/paired_group_seed_gains.csv')
for method,ref in [('CPOR','ProCAST72Warm'),('ProCAST','ProCAST72Warm'),('Raw','Raw'),('NNLS','Anchor')]:
    z=old[old.reference==ref];q=g[g.method==method]
    v=q.merge(z,on=['dataset','backbone','seed'],validate='one_to_one')
    target='c0_rmse' if method=='CPOR' else 'reference_rmse'
    np.testing.assert_allclose(v.rmse,v[target],rtol=1e-12,atol=1e-10)
assert abs(totals['ProCAST']['mean']-28.4743323683)<1e-8
assert round(totals['Ada-Y']['per_experiment']['42'],3)==31.177
assert round(totals['PIR']['per_experiment']['42'],3)==30.373
for name,df in [('CELL_METRICS.csv',f),('GROUP_EXPERIMENT_METRICS.csv',g),
                ('THREE_EXPERIMENT_RMSE.csv',summary),('PAIRED_GAINS.csv',gains)]:
    df.to_csv(H/name,index=False)
(H/'INPUT_SHA256.json').write_text(json.dumps(inputs,indent=2))
result=dict(complete=True,configurations=22,experiments=3,groups=12,methods=6,
            new_adapter_fits=88,reused_adapter_results=44,
            added_historical_support_fits=len(list((H/'historical_support').rglob('run.json'))),
            scored_backbones_and_cpor_reused=True,primary_summary=totals,
            protocol='Mean window RMSE within group, then mean over three experiments; gains paired within experiment before averaging.',
            original_paper_reproduction=False,parameter_search=False)
(H/'SUMMARY.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
print(summary.pivot(index=['dataset','backbone'],columns='method',values='mean_rmse').round(3).to_string())
