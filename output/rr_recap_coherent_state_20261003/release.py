from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from strict_solver import SafeProjection
from rollout import advance,RR,K
from guard_v8 import guard
from engine import safe_bounds
from controller import excess_bound

def certificate(qp,b,p,target,ub,lo,hi):
    xx=(p-b)/qp.s;g=(p-target)/qp.s;obj=float(.5*g@g);lower=(lo-b)/qp.s;upper=(hi-b)/qp.s
    sub=qp.S[:,qp.cols];cap=np.min(np.divide(hi[:,None],sub,out=np.full_like(sub,np.inf),where=sub>0),axis=0)
    lw=-ub[qp.cols]/qp.cs;uw=(np.maximum(cap,ub[qp.cols])-ub[qp.cols])/qp.cs;c=qp.M.T@g;G=np.vstack([qp.M,-qp.M]);rhs=np.r_[upper,-lower]
    lp=linprog(c,A_ub=G,b_ub=rhs,bounds=list(zip(lw,uw)),method='highs')
    if not lp.success:return obj,obj
    mult=np.maximum(0,-lp.ineqlin.marginals);red=c+G.T@mult;dual=red@np.where(red>=0,lw,uw)-mult@rhs
    gap=float(g@xx-dual);assert gap>=-1e-7*(1+obj)
    return obj,min(obj,max(0,gap))

def release(x,target,kind):
    qp=SafeProjection(x['S'],x['v']);accounts=[Fraction(0) for _ in x['v']];previous=None;reservation=None;stats=[]
    out={key:[] for key in ['pred','witness','available_after','reserved','lo','hi','contraction','objective','gap']}
    for i,t in enumerate(range(36,len(x['truth']))):
        if i:accounts=advance([a+r for a,r in zip(accounts,reservation)],x['b'][t-1],previous,x['truth'][t-1])
        b,ub=x['b'][t],x['ub'][t];lo,hi=safe_bounds(b,np.array([float(a) for a in accounts]));ql,qh=lo,hi
        if kind.endswith('Segment'):ql=np.maximum(lo,np.minimum(b,target[i]));qh=np.minimum(hi,np.maximum(b,target[i]))
        p,u,info=qp.solve(b,target[i],ub,ql,qh)
        if info.get('solver')=='Clarabel' and qp.x.value is not None and np.any(qh>ql):
            trial=np.clip(b+qp.s*qp.x.value,ql,qh)
            if np.allclose(trial,x['S']@u,rtol=1e-8,atol=2e-6):p=trial
        p,u,c=guard(b,p,ub,u,x['S'],ql,qh,accounts,RR,K)
        obj,gap=certificate(qp,b,p,target[i],ub,ql,qh)
        risk=excess_bound(b,p,RR,K)
        for j in range(len(accounts)):accounts[j]-=risk[j]
        assert min(accounts)>=0
        vals=dict(pred=p,witness=u,available_after=[float(a) for a in accounts],reserved=[float(a) for a in risk],lo=lo,hi=hi,contraction=c,objective=obj,gap=gap)
        for key,value in vals.items():out[key].append(value)
        stats.append(info);previous=p;reservation=risk
    return {**{key:np.asarray(a) for key,a in out.items()},'stats':stats}
