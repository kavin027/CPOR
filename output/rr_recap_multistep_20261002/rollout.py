from fractions import Fraction
import numpy as np
from strict_solver import SafeProjection
from engine import safe_bounds,numerical_guard
from controller import excess_bound
from method_soc import future_targets

RR=(1+Fraction('.005'))**2;K=RR-1

def advance(accounts,b,p,mu):
    new=[]
    for B,bb,pp,yy in zip(accounts,b,p,mu):
        bb,pp,yy=[Fraction.from_float(float(a)) for a in [bb,pp,yy]]
        value=B+RR*(bb-yy)**2-(pp-yy)**2;assert value>=0;new.append(value)
    return new

def action(solver,S,b,ub,mu,accounts,segment):
    credit=np.array([float(a) for a in accounts]);lo,hi=safe_bounds(b,credit);ql,qh=lo,hi
    if segment:ql=np.maximum(lo,np.minimum(b,mu));qh=np.minimum(hi,np.maximum(b,mu))
    p,u,info=solver.solve(b,mu,ub,ql,qh)
    p,u,c=numerical_guard(b,p,ub,u,S,lo,hi,accounts,RR,K,True)
    if segment:
        relative=np.max(np.maximum(0,np.maximum(np.minimum(b,mu)-p,p-np.maximum(b,mu)))/np.maximum(1,np.maximum(abs(b),abs(mu))))
        assert relative<=1e-8
    return p,u,c,info

def release(x,horizon=3,geometry='full'):
    b,y,S,ub,v=[x[k] for k in ['b','truth','S','ub','v']];segment=geometry=='segment'
    base=SafeProjection(S,v);available=[Fraction(0) for _ in v];published={};reserved={};stats=[]
    out={key:[] for key in ['pred','witness','available_after','reserved','lo','hi','contraction','horizon','choice','branch_cost','nominal_gain','all_contractions']}
    for t in range(36,len(y)):
        if t>36:
            available=advance([available[j]+reserved[t-1][j] for j in range(len(v))],b[t-1],published[t-1],y[t-1])
        H=min(horizon,12-t%12,len(y)-t);mu=future_targets(x,t,H)
        credit=np.array([float(a) for a in available]);lo,hi=safe_bounds(b[t],credit)
        p,u,c,info=action(base,S,b[t],ub[t],mu[0],available,segment)
        candidates=[(p,u,c)];meta=[info];contractions=[c]
        if H>1 and np.any(credit):
            F=np.sum((mu[1:]-b[t+1:t+H])**2/v,axis=0)
            priorities=[1+np.minimum(10,F/(F.mean()+1e-12)),1+np.minimum(10,F/(credit/v+1e-6))]
            for weight in priorities:
                q,w,lam,qinfo=action(SafeProjection(S,v/weight),S,b[t],ub[t],mu[0],available,segment)
                candidates.append((q,w,lam));meta.append(qinfo);contractions.append(lam)
            q=.5*(b[t]+p);w=.5*(ub[t]+u)
            q,w,lam=numerical_guard(b[t],q,ub[t],w,S,lo,hi,available,RR,K,True)
            candidates.extend([(q,w,lam),(b[t].copy(),ub[t].copy(),1.)]);contractions.extend([lam,1.])
        costs=[]
        for q,w,lam in candidates:
            cost=float(np.mean((q-mu[0])**2/v));accounts=advance(available,b[t],q,mu[0])
            for h in range(1,H):
                qh,wh,ch,ih=action(base,S,b[t+h],ub[t+h],mu[h],accounts,segment)
                contractions.append(ch);cost+=float(np.mean((qh-mu[h])**2/v));accounts=advance(accounts,b[t+h],qh,mu[h])
                np.testing.assert_allclose(qh,wh@S.T,rtol=1e-8,atol=2e-6)
            costs.append(cost)
        choice=int(np.argmin(costs));p,u,c=candidates[choice]
        h=excess_bound(b[t],p,RR,K)
        for j in range(len(v)):available[j]-=h[j]
        assert min(available)>=0
        padded=np.full(5,np.nan);padded[:len(costs)]=costs
        values=dict(pred=p,witness=u,available_after=[float(a) for a in available],reserved=[float(a) for a in h],lo=lo,hi=hi,contraction=c,horizon=H,choice=choice,branch_cost=padded,nominal_gain=costs[0]-costs[choice],all_contractions=sum(a<.99 for a in contractions))
        for key,value in values.items():out[key].append(value)
        stats.append(dict(initial_projection=meta,branch_count=len(costs)));published[t]=p;reserved[t]=h
    return {**{key:np.asarray(value) for key,value in out.items()},'stats':stats}
