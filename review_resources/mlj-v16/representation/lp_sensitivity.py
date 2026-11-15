"""Fixed observed endpoints, alternative nonnegative representations. Use the original PIR stream cache."""
from pathlib import Path
import csv,sys,hashlib,os
from fractions import Fraction as F
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from grouped import grouped,partition
R=F(40401,40000)
W=Path(__file__).resolve().parent
def exact(a):return [F.from_float(float(x)) for x in a]
def feasible(b,p,A):
    return all(g<=a for g,a in zip(charge(b,p),A))
def charge(b,p):
    return [x*x-R*y*y if x>R*y else R*(x-y)**2/(R-1) for x,y in zip(p,b)]
def alternative(S,group):
    n,m=S.shape;js=set(group);M=np.zeros((m,m),dtype=int)
    columns=defaultdict(list)
    for k,v in enumerate(S.T.astype(int)):columns[tuple(v)].append(k)
    for j,s in enumerate(S.T.astype(int)):
        options=[(k,) for k in columns[tuple(s)]]
        for k,v in enumerate(S.T.astype(int)):
            if not v.any() or np.any(v>s) or np.array_equal(v,s):continue
            options += [(k,h) for h in columns.get(tuple(s-v),[]) if k<h]
        def rank(opts):
            first=sum((S[:,k] for k in opts if k in js),np.zeros(n))
            return (-np.abs(first-(s if j in js else 0)).sum(),opts)
        M[list(min(options,key=rank)),j]=1
    np.testing.assert_array_equal(S@M,S)
    return M
def job(args):
    archive,ids=args
    ds,bb,run,part=ids;base=archive.joinpath('streams',*ids)
    side=base/('carry' if part=='continuous' else 'single')
    with np.load(base/'context.npz') as f:
        S=f['S'];v=f['v'];Y=f['truth'][36:];B=f['b'][36:];UB=f['ub'][36:]
    with np.load(side/'PIRCoherent.npz') as f: Z=f['pred'];U=f['witness']
    grouped_orig=np.load(side/'PIRTwoGroup.npz')['pred']
    full=np.load(side/'PIRFull.npz')['pred']
    mask=partition(S);M=alternative(S,mask)
    import partition_sensitivity as ps
    ps.setup(W)
    P=[];ab=ps.alternative(S,mask,UB).astype(float);au=ps.alternative(S,mask,U).astype(float)
    for w,p in [(ab,B),(au,Z)]:
        assert w.min()>=0 and np.allclose(w@S.T,p,rtol=1e-8,atol=2e-6)
    A=[F(0)]*S.shape[0];delta=((au-ab)[:,mask]-(U-UB)[:,mask])@S[:,mask].T
    origstate=A.copy();maxdiff=0
    for t,(b,ub,z,u,y) in enumerate(zip(B,ab,Z,au,Y)):
        bq=exact(b);yq=exact(y)
        gen=lambda a,br,ur:grouped(S[:,mask],mask,b,br,z,ur,a,bq,1/v,dict(feasible=feasible,exact=exact),float(R))
        p,w,*_=gen(A,ub,u)
        # Existing routine may fall back to the anchor. Verify the archived
        # original rule on the same stream, with its own settlements.
        po,wo,*_=gen(origstate,UB[t],U[t])
        maxdiff=max(maxdiff,float(abs(po-grouped_orig[t]).max()))
        assert np.allclose(po,grouped_orig[t],rtol=1e-8,atol=2e-6),(ids,t,maxdiff)
        def update(p,state,w):
            pq=exact(p)
            assert feasible(bq,pq,state) and np.min(w)>=0 and np.allclose(S@w,p,rtol=1e-8,atol=2e-6)
            state[:]=[a+R*(b-y0)**2-(v0-y0)**2 for a,b,y0,v0 in zip(state,bq,yq,pq)]
            assert min(state)>=0
        update(p,A,w);update(grouped_orig[t],origstate,np.load(side/'PIRTwoGroup.npz')['witness'][t] if not np.array_equal(po,grouped_orig[t]) else wo);P.append(p)
    def score(q):return np.sqrt(np.mean((q-Y)**2,0)).mean()
    alt,orig,fu=map(score,[np.array(P),grouped_orig,full])
    return dict(dataset=ds,backbone=bb,run=run,part=part,original=orig,alternative=alt,full=fu,n=len(P),geometry_changes=int((np.abs(delta)>1e-8*np.maximum(1,np.abs(Z))).any(1).sum()),changed_witness_dates=int(np.any(ab!=UB,axis=1).sum()),max_original_diff=maxdiff,checks=Y.size,breaches=0)
if __name__=='__main__':
    archive=Path(sys.argv[1]).resolve()
    jobs=[(archive,q.relative_to(archive/'streams').parts[:-1]) for ds in ['RH','ECommerce'] for q in sorted((archive/'streams'/ds).glob('*/*/*/context.npz')) if q.parent.name in ['early','late','continuous']]
    assert len(jobs)==54
    with ProcessPoolExecutor(3) as ex:
        rows=list(ex.map(job,jobs))
    with (W/'lp_streams.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    import pandas as pd
    s=pd.DataFrame(rows);s['protocol']=np.where(s.part=='continuous','continuous','window')
    q=s.groupby(['dataset','protocol','backbone','run'])[['original','alternative','full']].mean()
    q['alt_vs_original']=100*(1-q.alternative/q.original)
    q['full_vs_alt']=100*(1-q.full/q.alternative)
    q['full_vs_original']=100*(1-q.full/q.original)
    q.groupby(['dataset','protocol']).mean().to_csv(W/'lp_summary.csv')
    print(q.groupby(['dataset','protocol']).mean().round(5));print(s.groupby('dataset')[['geometry_changes','n','checks']].sum())

