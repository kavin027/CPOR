from pathlib import Path
import sys,importlib.util,json,hashlib,time,copy
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;O=H.parent;ROOT=O.parent
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
fresh=mod('seed_frozen_core',O/'c0_frozen_confirmation_20261014/run.py')
base=fresh.base;old=fresh.old
cells=json.loads((O/'kais_review_response_20261016/Online_Resource_1/cells.json').read_text())
fresh.HERE=H
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files if k!='dates'}
def key(c):return {k:c[k] for k in ['family','dataset','backbone','window']}
def dest(c,seed):return H/'results'/c['dataset']/c['backbone']/c['window']/str(seed)
def datafile(c):
    if c['dataset']=='M5':return O/'c0_frozen_m5_20261014/observed.csv'
    if c['dataset']=='Labour':return O/'rr_recap_evidence_extension_20261002/major_revision_20261002/external/data/Labour.csv'
    return O/f"procast_joint_safe_20261001_v4/data/{c['dataset']}.csv"
def rawfile(c,seed):
    prefix=O/'rr_recap_evidence_extension_20261002'
    if c['dataset']=='Labour':prefix=prefix/'major_revision_20261002/external'
    return prefix/'forecasts'/c['dataset']/c['backbone']/c['window']/str(seed)/'raw.npz'
def endpoint(c,seed):return Path(str(ROOT/c['endpoint_source']).replace('\\42\\',f'\\{seed}\\'))
def histories(c,seed):return H/'support'/c['dataset']/c['backbone']/str(seed)
def verify_registration():
    r=json.loads((H/'REGISTRATION.json').read_text())
    expected=dict(r['sources'])
    for p in sorted((H/'amendments').glob('V*.json')):
        a=json.loads(p.read_text());assert a['registration_sha256']==sha(H/'REGISTRATION.json')
        for name,d in a['sources'].items():assert expected[name]==d['old'];expected[name]=d['new']
    for p,h in expected.items():assert sha(p)==h,p
    return r
def protected_dev(d,x,D,expert):
    # Preserve the original development driver's calibration QP warm state.
    p=d/'C0.npz'
    if not p.exists():
        a=base.Actor(x['S'],x['v'],'C0');started=time.perf_counter()
        for t in range(36):a.issue(t,x['b'][t],x['ub'][t],D[t],expert[t],None if t==0 else x['truth'][t-1],calibration=True)
        rows=[];logs=[]
        for t in range(36,108):
            z=a.issue(t,x['b'][t],x['ub'][t],D[t],expert[t],x['truth'][t-1],reset=t==36);logs.append(dict(t=t,fit=z.pop('fit')));rows.append(z)
        out={k:np.array([r[k] for r in rows]) for k in rows[0]};checks=old.audit(x,out,1);a.close(x['truth'][-1])
        np.savez_compressed(p,**out);np.savez_compressed(d/'C0_snapshots.npz',**{k:np.array([r[k] for r in a.records]) for k in a.records[0]})
        js(p.with_suffix('.json'),dict(sha256=sha(p),checks=checks,seconds=time.perf_counter()-started,failed_publications=int(out['solver_failed'].sum()),logs=logs))
    snap=load(d/'C0_snapshots.npz');records=[{k:np.array(v[t],copy=True) for k,v in snap.items()} for t in range(36)]
    E=np.sum((snap['published'][:36]-x['truth'][:36])**2,axis=0);kappa=float(np.sqrt(E).mean()/E.mean()) if E.mean()>0 else 1/float(np.sqrt(x['v'].mean()))
    p=d/'SquareFull.npz'
    if not p.exists():
        a=fresh.factorial.Actor(x['S'],x['v'],'SquareFull',records,kappa);rows=[];logs=[];started=time.perf_counter()
        for t in range(36,108):
            z=a.issue(t,x['b'][t],x['ub'][t],D[t],expert[t],None if t==36 else x['truth'][t-1]);logs.append(dict(t=t,fit=z.pop('fit')));rows.append(z)
        out={k:np.array([r[k] for r in rows]) for k in rows[0]};checks=old.audit(x,out,1);a.close(x['truth'][-1])
        np.savez_compressed(p,**out);np.savez_compressed(d/'SquareFull_snapshots.npz',**{k:np.array([r[k] for r in a.records]) for k in a.records[0]})
        js(p.with_suffix('.json'),dict(sha256=sha(p),checks=checks,seconds=time.perf_counter()-started,kappa=kappa,failed_publications=int(out['solver_failed'].sum()),logs=logs))
def dev_context(c,seed):
    x=load(endpoint(c,seed));rel=f"{c['dataset']}/{c['backbone']}/{c['window']}/{seed}"
    d=O/'rr_recap_error_dependence_20261003/results'/rel
    bank=load(d/'bank.npz')['bank'];saved=load(d/'CovFull.npz')
    D,expert=base.make_contexts(x,bank,saved,old)
    return x,D,expert,bank
def protect(x,target):
    A=[old.Fraction(0) for _ in x['v']];qp=old.SafeProjection(x['S'],x['v']);rows=[];us=[]
    for j,t in enumerate(range(36,108)):
        if j:A=old.advance(A,x['b'][t-1],rows[-1],x['truth'][t-1])
        p,u,_=old.one_action(qp,x['S'],x['b'][t],x['ub'][t],target[j],A)
        assert old.certified(x['b'][t],p,A,old.RR,old.K,True)
        rows.append(p);us.append(u)
    old.advance(A,x['b'][-1],rows[-1],x['truth'][-1])
    return np.array(rows),np.array(us)
def controls(d,x,expert,bank):
    file=d/'controls.npz'
    if file.exists():assert sha(file)==json.loads(file.with_suffix('.json').read_text())['sha256'];return
    y=x['truth'];n,J=bank.shape[1:];cal=np.sqrt(((expert[:36]-y[:36,:,None])**2).mean(0)).mean(0);chosen=int(np.argmin(cal))
    w=np.full((n,J),1/J);regrets=np.zeros_like(w);squares=np.zeros_like(w);mixtures=[]
    scaled=bank/np.sqrt(x['v'])[None,:,None];labels=y/np.sqrt(x['v'])[None,:]
    for t in range(108):
        p=(scaled[t]*w).sum(1);mixtures.append(p*np.sqrt(x['v']))
        r=2*(p-labels[t])[:,None]*(p[:,None]-scaled[t]);regrets+=r;squares+=r*r
        q=np.maximum(regrets,0)/(1+squares);total=q.sum(1,keepdims=True);w=np.divide(q,total,out=np.full_like(q,1/J),where=total>0)
    z,_=old.project(x['S'],np.asarray(mixtures),x['v']);ml,uml=protect(x,z[36:]);cb,ucb=protect(x,expert[36:,:,chosen])
    np.savez_compressed(file,MLPol=z[36:],ProtectedMLPol=ml,CalBest=expert[36:,:,chosen],ProtectedCalBest=cb,MLPolWitness=uml,CalBestWitness=ucb)
    js(file.with_suffix('.json'),dict(sha256=sha(file),chosen_expert=chosen,calibration_rmse=cal.tolist(),registration_sha256=sha(H/'REGISTRATION.json')))
