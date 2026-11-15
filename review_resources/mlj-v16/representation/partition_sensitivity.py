"""Fixed retrospective sensitivity at the original archived observed endpoints.
Run with --archive path_to_frozen_seven_dataset_stream_root --original path_to_original_OR2/fixed_groups
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, sys, json, csv
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog
from sympy import Matrix
from concurrent.futures import ProcessPoolExecutor

H=Path(__file__).resolve().parent
def setup(folder):
    sys.path.insert(0,str(folder))
    global g
    import generate_bounds as g

def alternative(S,mask,initial):
    """Solve an LP to choose a basis; apply its map in exact arithmetic to the original rational endpoints."""
    S=S.astype(int); m=S.shape[1]; cost=np.zeros(m);cost[mask]=-1
    valid=(S!=0).any(0); bounds=[(0,None) if a else (0,0) for a in valid]
    def repaired_basis(ix):
        inv=Matrix(S[:,ix].tolist()).inv()
        return [[F(int(v.p),int(v.q)) for v in row] for row in inv.tolist()]
    inverse=lru_cache(maxsize=80)(repaired_basis)
    obj=[]
    for row in initial:
        old=g.array(row); target=S@old
        scale=max(1.,max(float(abs(q)) for q in target));rhs=np.array([float(q)/scale for q in target])
        res=linprog(cost,A_eq=S,b_eq=rhs,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
        if not res.success:raise ValueError(res.message)
        active=np.flatnonzero(res.x>1e-13)
        seq=dict.fromkeys([*active,*np.flatnonzero(old>0),*np.flatnonzero(valid)])
        chosen=[]
        for j in seq:
            if np.linalg.matrix_rank(S[:,chosen+[j]])>len(chosen):chosen.append(j)
            if len(chosen)==S.shape[0]:break
        inv=inverse(tuple(chosen));new=np.full(m,F(0),dtype=object)
        new[chosen]=[sum(v*q for v,q in zip(rr,target)) for rr in inv]
        # Degenerate basis may produce a tiny negative correction. Stay along the exact null direction.
        t=min([F(1)]+[old[j]/(old[j]-new[j]) for j in chosen if new[j]<0])
        row2=old+t*(new-old)
        assert np.array_equal(S@row2,target) and min(row2)>=0
        obj.append(row2)
    return np.array(obj)

def case(job):
    archive,orig,ds,mode=job;setup(orig)
    f=archive/'streams'/ds/'PatchTST/42/early'
    x=np.load(f/'context.npz'); zf=np.load(f/'single/PIRCoherent.npz')
    S=x['S'].astype(int);n,m=S.shape; B=g.array(x['b'][36:]); Z=g.array(zf['pred']);Y=g.array(x['truth'][36:]);U=g.array(x['ub'][36:]);V=g.array(zf['witness']);P=g.array(np.load(f/'single/PIRFull.npz')['pred'])
    mask=g.partition(S) if mode!='contiguous' else np.arange((m+1)//2)
    origmask=g.partition(S)
    def directions(U,V,mask):return (V-U)[:,mask] @ S[:,mask].T
    dbase=directions(U,V,origmask)
    if mode=='alternate_witness':
        U=alternative(S,mask,x['ub'][36:]);V=alternative(S,mask,zf['witness'])
    d0=directions(U,V,mask);drifts=d0-dbase;steps=int((drifts!=0).any(1).sum())
    A=[F(0)]*n; loss=[F(0)]*n;T=len(Y)
    for t,(b,z,y) in enumerate(zip(B,Z,Y)):
        L=[];Uhi=[]
        for i in range(n):
            r=g.ceil_sqrt(A[i]*(g.RR-1)/g.RR); L.append(max(F(0),b[i]-r))
            Uhi.append(b[i]+r if A[i]<=g.RR*(g.RR-1)*b[i]**2 else g.ceil_sqrt(A[i]+g.RR*b[i]**2))
        pad=[(1+max(abs(bi),abs(zi),abs(hi)))/10**7 for bi,zi,hi in zip(b,z,Uhi)]
        corners=[max(abs(eb),abs(ez),abs(ez-eb)) for eb,ez in zip(S@U[t]-b,S@V[t]-z)]
        assert all(c<=p for c,p in zip(corners,pad))
        a=d0[t];d=z-b-a;pt=[(F(0),F(0)),(F(1),F(0)),(F(1),F(1)),(F(0),F(1))]
        for i in range(n):
            for s,rhs in [(1,Uhi[i]-b[i]+pad[i]),(-1,b[i]-L[i]+pad[i])]:
                if max(0,s*a[i])+max(0,s*d[i])>rhs:pt=g.clip(pt,s*a[i],s*d[i],rhs)
        for i in range(n):
            v=[b[i]+a[i]*v1+d[i]*v2 for v1,v2 in pt]
            loss[i]+=max(F(0),min(v)-pad[i]-y[i],y[i]-max(v)-pad[i])**2
            A[i]+=g.RR*(b[i]-y[i])**2-max(F(0),L[i]-y[i],y[i]-Uhi[i])**2
    low=sum(g.floor_sqrt(q/T) for q in loss)/n;full=sum(g.ceil_sqrt(sum((P[:,i]-Y[:,i])**2)/T) for i in range(n))/n
    print(ds,mode,round(float(low),4),steps,flush=True)
    return dict(dataset=ds,variant=mode,full_upper=float(full),lower=float(low),strict=int(full<low),changed_directions_issues=steps,mask_first=' '.join(map(str,mask)),max_direction_shift=float(max(abs(drifts).ravel())))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--out',type=Path,default=H);args=ap.parse_args()
    pairs=[(ds,v) for ds in ['RH','ECommerce','Labour','AEMO','Wiki2','Traffic','M5'] for v in ['contiguous']+(['alternate_witness'] if ds in ['RH','ECommerce'] else [])]
    with ProcessPoolExecutor(3) as e:rows=list(e.map(case,[(args.archive.resolve(),args.original.resolve(),ds,v) for ds,v in pairs]))
    old=list(csv.DictReader((args.original/'exact_grid.csv').open(encoding='utf8')))
    for row in rows:
        q=next(r for r in old if (r['dataset'],r['backbone'],r['run'],r['part'],r['rule'])==(row['dataset'],'PatchTST','42','early','two'))
        row.update(original_lower=float(q['relaxation_lower']),original_strict=int(q['strict_mean']))
    with (args.out/'partition_sensitivity.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
