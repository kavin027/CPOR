"""Read-only audit of existing Li penalties and Robust calibration geometry. No model or grid fits."""
from pathlib import Path
import numpy as np,pandas as pd,hashlib,json,subprocess
from threadpoolctl import threadpool_limits
W=Path(__file__).resolve().parent;O=W.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def li():
    d=O/'cpor_li_factor_20261106'
    files={f.stem:f for f in (d/'verified_releases').glob('*.npz')}
    for bb in ['PatchTST','DLinear']: files['Labour_'+bb+'_early']=O/'cpor_labour_early_three_runs_20261107/runs'/bb/'42/recent/LiFactor.npz'
    original=O/'cpor_li_preprint_20261106/author/main_experiment.R'
    assert 'lambda_grid_ridge <- sort(c(0, 10^(seq(-1, 5, by = 0.25))))' in original.read_text(encoding='utf8')
    grid=np.r_[0,10.**np.arange(-1,5.01,.25)]
    rows=[]
    for k,f in sorted(files.items()):
        ds,bb,win=k.split('_')
        with np.load(f) as fi:l=fi['lambda_values']
        assert l.shape==(72,) and np.isin(l,grid).all()
        rows.append(dict(dataset=ds,backbone=bb,window=win,releases=len(l),upper=int((l==1e5).sum()),lower=int((l==0).sum()),source=f.relative_to(O).as_posix(),sha256=sha(f)))
    m=pd.DataFrame(rows);m.to_csv(W/'li_audit.csv',index=False)
    print('LI',len(rows),m.releases.sum(),m.upper.sum(),m.groupby('dataset').upper.sum().to_dict(), 'extra',m.query("dataset=='Labour' and window=='early'").upper.tolist())
    report=pd.read_csv(d/'R_check.csv')
    assert (report.chosen==report.python_lambda).all();print('R crosscheck',report.columns.tolist(),len(report));print(report.max(numeric_only=True).to_dict())
def robust():
    h=O/'cpor_external_20261105';idx=pd.read_csv(O/'cpor_minor_v8_20261022/fresh_evidence/CURRENT_RESULTS_20261022/derived/TRAJECTORY_INDEX.csv').query('seed==42')
    rows=[]
    for r in idx.itertuples():
        d=h/'runs'/r.dataset/r.backbone/r.window;sc=json.loads((d/'spec.json').read_text());p=Path(r.source)/'endpoints.npz'
        assert sha(p)==sc['input_sha256']
        with np.load(p) as inp,np.load(d/'targets.npz') as t:
            x=inp['raw'][:36];xt=inp['raw'][36:];y=inp['truth'][36:]
            _,s,vt=np.linalg.svd(x,full_matrices=False)
            rank=np.linalg.matrix_rank(x)
            assert rank==sc['robust']['rank']
            v=vt[:rank];rel=np.linalg.norm(xt-(xt@v.T)@v)/np.linalg.norm(xt)
            delta=np.max(abs(t['RobustMN']-t['RobustOrth']))
            calc={k:float(np.sqrt(np.mean((t[k]-y)**2,axis=0)).mean()) for k in ('RobustMN','RobustOrth')}
            rows.append(dict(dataset=r.dataset,backbone=r.backbone,window=r.window,rank=rank,n=x.shape[1],missing_dimensions=x.shape[1]-rank,cond_identified=s[0]/s[rank-1],off_training_rowspace_fraction=rel,max_pred_difference=delta,normal_eq=sc['robust']['normal_equation'],**calc))
    m=pd.DataFrame(rows);m.to_csv(W/'robust_audit.csv',index=False)
    print('Robust',len(m), 'rank-deficient',(m.missing_dimensions>0).sum(), 'range by dataset')
    print(m.groupby('dataset')[['rank','n','off_training_rowspace_fraction','cond_identified','RobustMN','RobustOrth']].agg(['min','max']).round(5).to_string())
    print('max normal',m.normal_eq.max())
    ref=pd.read_csv(h/'groups.csv').query("scope=='all'").set_index(['dataset','backbone']).sort_index()
    audit=m.groupby(['dataset','backbone'])[['RobustMN','RobustOrth']].mean().sort_index()
    np.testing.assert_allclose(audit,ref[audit.columns],atol=1e-8)
    print('protection effect',100*(1-ref['RobustMN+P']/ref.RobustMN).mean(),100*(1-ref['RobustMN+P']/ref.RobustMN).median())
    print((h/'native/robust_check.json').read_text())
if __name__=='__main__':
    with threadpool_limits(1):li();robust()

