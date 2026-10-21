import numpy as np
from scipy.optimize import minimize
from fractions import Fraction
from strict_solver import SafeProjection
from rollout import action,advance,RR,K
from engine import safe_bounds
from controller import excess_bound
from frozen_core import project
KINDS=['InverseFull','DiagonalFull','CovFull','CovSegment','StaticCovFull','UniformFull']

def simplex(x):
    u=np.sort(x,axis=-1)[:,::-1];c=np.cumsum(u,axis=-1)-1
    rho=np.sum(u>c/np.arange(1,x.shape[1]+1),axis=-1)-1
    tau=c[np.arange(len(x)),rho]/(rho+1)
    w=np.maximum(x-tau[:,None],0);w/=w.sum(1,keepdims=True)
    return w

def gaps(Q,w):
    g=2*np.einsum('jkl,jl->jk',Q,w);value=.5*np.sum(g*w,axis=1)
    return np.maximum(0,np.sum(g*w,axis=1)-g.min(1)),value

def solve_weights(Q,diagonal=False):
    n,k,_=Q.shape
    if diagonal:
        w=1/np.diagonal(Q,axis1=1,axis2=2);w/=w.sum(1,keepdims=True)
        gap,obj=gaps(Q,w);return w,gap,obj,0
    L=2*np.linalg.eigvalsh(Q)[:,-1];assert L.min()>0
    w=np.full((n,k),1/k);z=w.copy();alpha=1.
    for iteration in range(512):
        new=simplex(z-2*np.einsum('jkl,jl->jk',Q,z)/L[:,None])
        aa=(1+np.sqrt(1+4*alpha*alpha))/2
        z=new+(alpha-1)/aa*(new-w);w=new;alpha=aa
        if iteration%16==15:
            gap,obj=gaps(Q,w)
            if np.all(gap<=1e-9*(1+obj)):break
    gap,obj=gaps(Q,w);need=np.flatnonzero(gap>1e-8*(1+obj))
    for j in need:
        q=Q[j]
        r=minimize(lambda a:float(a@q@a),w[j],jac=lambda a:2*q@a,method='SLSQP',
            bounds=[(0,1)]*k,constraints=[dict(type='eq',fun=lambda a:a.sum()-1,jac=lambda a:np.ones(k))],
            options=dict(ftol=1e-13,maxiter=1000))
        w[j]=simplex(r.x[None])[0]
        # Polish the identified simplex face rather than relaxing the certificate.
        for threshold in [1e-10,1e-8,1e-6]:
            active=np.flatnonzero(w[j]>threshold)
            if not len(active):continue
            sol=np.linalg.solve(q[np.ix_(active,active)],np.ones(len(active)))
            sol/=sol.sum()
            if sol.min()<0:continue
            trial=np.zeros(k);trial[active]=sol
            oldgap,_=gaps(q[None],w[j][None]);newgap,_=gaps(q[None],trial[None])
            if newgap[0]<oldgap[0]:w[j]=trial
    gap,obj=gaps(Q,w)
    assert np.isfinite(w).all() and w.min()>=0
    assert np.max(gap/(1+obj))<=1e-7, np.max(gap/(1+obj))
    return w,gap,obj,len(need)

def moment(bank,y,v,t,diagonal=False):
    e=(bank[max(0,t-28):t]-y[max(0,t-28):t,:,None])/np.sqrt(v)[None,:,None]
    G=np.einsum('tjk,tjl->jkl',e,e)/len(e)
    diag=np.diagonal(G,axis1=1,axis2=2);scale=np.maximum(diag.mean(1),1e-12)
    Q=np.zeros_like(G) if diagonal else .9*G
    ids=np.arange(G.shape[-1]);Q[:,ids,ids]+=(diag if diagonal else .1*diag)+1e-6*scale[:,None]
    return Q/scale[:,None,None]

def targets(x,bank,kind):
    y,v,S=x['truth'],x['v'],x['S'];n=len(v);k=bank.shape[2];weights=[];gap=[];objective=[];fallback=[]
    static=None
    for t in range(36,len(y)):
        if kind=='UniformFull':
            w=np.full((n,k),1/k);g=np.zeros(n);ob=np.zeros(n);fb=0
        elif kind=='StaticCovFull' and static is not None:w,g,ob,fb=static
        else:
            Q=moment(bank,y,v,36 if kind=='StaticCovFull' else t,kind=='DiagonalFull')
            w,g,ob,fb=solve_weights(Q,kind=='DiagonalFull')
            if kind=='StaticCovFull':static=(w.copy(),g.copy(),ob.copy(),fb)
        weights.append(w);gap.append(g);objective.append(ob);fallback.append(fb)
    weights=np.asarray(weights);mixed=np.sum(bank[36:]*weights,axis=2);z,u=project(S,mixed,v)
    return dict(target=z,latent=u,weights=weights,combination_gap=np.asarray(gap),combination_objective=np.asarray(objective),fallback=np.asarray(fallback),raw_target=mixed)

def release(x,target,kind):
    v=x['v'];n=len(v);qp=SafeProjection(x['S'],v);accounts=[Fraction(0) for _ in v]
    previous=None;reservation=None;stats=[]
    out={k:[] for k in ['pred','witness','available_after','reserved','lo','hi','contraction']}
    for i,t in enumerate(range(36,len(x['truth']))):
        if i:accounts=advance([a+r for a,r in zip(accounts,reservation)],x['b'][t-1],previous,x['truth'][t-1])
        lo,hi=safe_bounds(x['b'][t],np.array([float(a) for a in accounts]))
        p,u,c,info=action(qp,x['S'],x['b'][t],x['ub'][t],target[i],accounts,kind=='CovSegment')
        risk=excess_bound(x['b'][t],p,RR,K)
        for j in range(n):accounts[j]-=risk[j]
        assert min(accounts)>=0
        val=dict(pred=p,witness=u,available_after=[float(a) for a in accounts],reserved=[float(a) for a in risk],lo=lo,hi=hi,contraction=c)
        for key,value in val.items():out[key].append(value)
        stats.append(info);previous=p;reservation=risk
    return {**{key:np.asarray(value) for key,value in out.items()},'stats':stats}
