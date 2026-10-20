from fractions import Fraction
import numpy as np
from strict_solver import SafeProjection
from engine import safe_bounds,numerical_guard
from controller import excess_bound
from frozen_core import project
from planner_soc import Planner

def future_targets(x,t,H):
    # No future endpoint z or future truth access, including inside a block.
    raw=x['raw'][t:t+H];mu,_=project(x['S'],raw+x['z'][t]-raw[0],x['v']);mu[0]=x['z'][t]
    return mu

def release(x,horizon=3,geometry='full'):
    b,y,S,ub,v=[x[k] for k in ['b','truth','S','ub','v']]
    qp=SafeProjection(S,v);planners={h:Planner(S,v,h,geometry=='segment') for h in range(2,horizon+1)}
    rr=(1+Fraction('.005'))**2;k=rr-1;available=[Fraction(0) for _ in v]
    published={};reserved={};stats=[]
    out={key:[] for key in ['pred','witness','available_after','reserved','lo','hi','contraction','horizon','repair_norm','segment_violation','fallback']}
    for t in range(36,len(y)):
        if t>36:
            i=t-1
            for j in range(len(v)):
                bb,yy,pp=[Fraction.from_float(float(a)) for a in [b[i,j],y[i,j],published[i][j]]]
                available[j]+=reserved[i][j]+rr*(bb-yy)**2-(pp-yy)**2
        assert min(available)>=0
        credit=np.array([float(a) for a in available]);lo,hi=safe_bounds(b[t],credit)
        H=min(horizon,12-t%12,len(y)-t);mu=future_targets(x,t,H)
        qlo,qhi=lo.copy(),hi.copy()
        if geometry=='segment':qlo=np.maximum(lo,np.minimum(b[t],mu[0]));qhi=np.minimum(hi,np.maximum(b[t],mu[0]))
        fallback=0;repair=0.;info={}
        if not np.any(credit):p=b[t].copy();u=ub[t].copy();info=dict(status='forced_anchor')
        elif H==1:p,u,info=qp.solve(b[t],mu[0],ub[t],qlo,qhi)
        else:
            try:
                plan,latent,info=planners[H].solve(b[t:t+H],ub[t:t+H],mu,credit,qlo,qhi)
                p,u=plan[0],latent[0]
                # Tiny first-step feasibility errors are repaired as a projection,
                # not by a large contraction along an unrelated direction.
                if np.any(p<qlo) or np.any(p>qhi):
                    fixed,fu,qinfo=qp.solve(b[t],p,ub[t],qlo,qhi)
                    repair=float(np.sqrt(np.mean((fixed-p)**2/v)));p,u=fixed,fu;info['repair_solver']=qinfo
            except RuntimeError as error:
                p,u,baseinfo=qp.solve(b[t],mu[0],ub[t],qlo,qhi);fallback=1
                info=dict(status='planning_failed_use_H1',error=str(error),fallback_info=baseinfo)
        p,u,c=numerical_guard(b[t],p,ub[t],u,S,lo,hi,available,rr,k,True)
        violation=float(np.max(np.maximum(0,np.maximum(np.minimum(b[t],mu[0])-p,p-np.maximum(b[t],mu[0])))/np.maximum(1,np.maximum(abs(b[t]),abs(mu[0])))))
        if geometry=='segment':assert violation<=1e-8
        h=excess_bound(b[t],p,rr,k)
        for j in range(len(v)):available[j]-=h[j]
        assert min(available)>=0
        values=dict(pred=p,witness=u,available_after=[float(a) for a in available],reserved=[float(a) for a in h],lo=lo,hi=hi,contraction=c,horizon=H,repair_norm=repair,segment_violation=violation,fallback=fallback)
        for key,value in values.items():out[key].append(value)
        published[t]=p;reserved[t]=h;stats.append(info)
    return {**{key:np.asarray(value) for key,value in out.items()},'stats':stats}
