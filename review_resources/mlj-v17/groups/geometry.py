"""Audit attainable node forecasts and replay Independent on Full's pre-release balance."""
import argparse,csv,json,sys,time,os
from fractions import Fraction as F
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import linprog
import pandas as pd
W=Path(__file__).resolve().parent; ROOT=W.parents[2]
V16=ROOT/'output/cpor_review_v16_20261115/groups'
sys.path.insert(0,os.environ.get('CPOR_V16_GROUPS',str(V16)))
from run_groups import Ledger,issue,check_hash,DEFAULT_ARCHIVE,DEFAULT_INDEX
from box import classify
def run(job):
    r,archive,runs=job;ds=r['dataset'];case=r['case']
    if (W/'scratch'/(case+'.npz')).exists():return
    files={k:archive/r[k+'_archive'] for k in ['context','PIRCoherent','PIRFull']}
    for k,f in files.items():check_hash(f,r[k+'_sha256'])
    with np.load(files['context']) as q,np.load(files['PIRCoherent']) as z,np.load(files['PIRFull']) as f,np.load(runs/case/'Independent.npz') as ii:
        S=q['S'];Y=q['truth'][36:].astype(float);B=q['b'][36:];U0=q['ub'][36:];Z=z['pred'];U=z['witness'];F0=f['pred'];WF=f['witness'];I=ii['pred'];v=q['v']
    flags=[];below=[];above=[];wl=[];ps=[];obj=[];actions=[]
    ledger=Ledger(B,Y,Z)
    for t,(b,ub,z,u,p,wf) in enumerate(zip(B,U0,Z,U,F0,WF)):
        how,l,h,ww=classify(S,np.minimum(ub,u),np.maximum(ub,u),p,wf)
        flags.append(how);below.append(l);above.append(h);wl.append(ww)
        peer,pi,*_=issue('Independent',S,v,b,ub,z,u,t,ledger,{})
        assert not ledger.flags(t,pi).any()
        ps.append(peer);obj.append([np.sum((p-z)**2/v),np.sum((peer-z)**2/v)])
        actions.append(np.max(np.abs(p-peer)/(2*(2e-6+1e-8*np.abs(p)))))
        ledger.step(t,ledger.toint(p,t))
    ps=np.array(ps)
    # These are squared-error differences in original units, not relative errors
    # from different nodes pooled across datasets.
    ir=(I-Y)**2;fr=(F0-Y)**2;pr=(ps-Y)**2
    rI=np.sqrt(ir.mean(0));rF=np.sqrt(fr.mean(0));rP=np.sqrt(pr.mean(0))
    den=(rI+rF)*len(Y)
    fn=lambda a:np.divide(a,den,out=np.zeros_like(a),where=den!=0)
    df=fn(ir-fr);state=fn(ir-pr);local=fn(pr-fr)
    assert np.allclose(df.sum(0),rI-rF,rtol=1e-8,atol=2e-10)
    np.savez_compressed(W/'scratch'/(case+'.npz'),classes=flags,below=below,above=above,wit=wl,
      full=rF,ind=rI,samestate=rP,contrib=df,local=local,state=state,obj=obj,actions=actions)
def main():
    q=argparse.ArgumentParser();q.add_argument('--workers',type=int,default=10);q.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE);q.add_argument('--index',type=Path,default=DEFAULT_INDEX);q.add_argument('--runs',type=Path,default=V16/'runs');q.add_argument('--case',default='')
    a=q.parse_args();rs=[r for r in csv.DictReader(a.index.open(encoding='utf-8-sig')) if a.case in r['case']]
    (W/'scratch').mkdir(exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        for i,_ in enumerate(pool.map(run,[(r,a.archive,a.runs) for r in rs])):
            if (i+1)%20==0:print('checked',i+1,flush=True)
    print('DONE',len(rs),flush=True)
if __name__=='__main__':main()

