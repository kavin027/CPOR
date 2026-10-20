from fractions import Fraction
import numpy as np
from fast_solver import SafeProjection
from engine import safe_bounds,numerical_guard

def excess_bound(b,p,rr,k):
    result=[]
    for bb,pp in zip(b,p):
        bb=Fraction.from_float(float(bb));pp=Fraction.from_float(float(pp))
        result.append(pp*pp-rr*bb*bb if pp>rr*bb else rr*(pp-bb)**2/k)
    return result

def proposal(b,z,y,v,delay=1):
    a=np.ones(b.shape[1]);G=np.full(b.shape[1],1e-12);out=[]
    for t in range(len(b)):
        i=t-delay
        if i>=0:
            g=2*(out[i]-y[i])*(z[i]-b[i])/v
            G+=g*g;a=np.clip(a-.3*g/np.sqrt(G),0,1)
        out.append(b[t]+a*(z[t]-b[t]))
    return np.asarray(out)

def release(x,target,geometry='full',delay=1):
    b,z,y,S,ub,uz,v=[x[k] for k in ['b','z','truth','S','ub','uz','v']]
    solver=SafeProjection(S,v);rr=(1+Fraction('.005'))**2;k=rr-1
    available=[Fraction(0) for _ in v];reserved={}; pred={}; states=[];witness=[];stats=[];factors=[];costs=[];lohist=[];hihist=[]
    for t in range(36,len(y)):
        matured=t-delay
        if matured>=36:
            for j in range(len(v)):
                bb,yy,pp=map(lambda a:Fraction.from_float(float(a)),[b[matured,j],y[matured,j],pred[matured][j]])
                refund=reserved[matured][j]+rr*(bb-yy)**2-(pp-yy)**2
                assert refund>=0;available[j]+=refund
        assert min(available)>=0
        credit=np.array([float(a) for a in available]);lo,hi=safe_bounds(b[t],credit)
        if geometry=='segment':lo=np.maximum(lo,np.minimum(b[t],z[t]));hi=np.minimum(hi,np.maximum(b[t],z[t]))
        if geometry=='scalar':
            d=z[t]-b[t];den=np.sum(d*d/v);a=np.clip(np.sum(d*(target[t]-b[t])/v)/den,0,1) if den else 0.
            cap=np.ones(len(v));positive=d>0;negative=d<0
            cap[positive]=(hi[positive]-b[t,positive])/d[positive];cap[negative]=(lo[negative]-b[t,negative])/d[negative]
            a=min(float(a),float(np.min(cap)))
            u=(1-a)*ub[t]+a*uz[t];p=u@S.T if a else b[t].copy()
            info=dict(solver='analytic_scalar',alpha=a)
        else:p,u,info=solver.solve(b[t],target[t],ub[t],lo,hi)
        p,u,lam=numerical_guard(b[t],p,ub[t],u,S,lo,hi,available,rr,k,True)
        h=excess_bound(b[t],p,rr,k)
        for j in range(len(v)):available[j]-=h[j]
        assert min(available)>=0
        reserved[t]=h;pred[t]=p;witness.append(u);states.append([float(a) for a in available]);costs.append([float(a) for a in h]);stats.append(info);factors.append(lam);lohist.append(lo);hihist.append(hi)
    return dict(pred=np.array(list(pred.values())),witness=np.asarray(witness),available_after=np.asarray(states),reserved=np.asarray(costs),lo=np.asarray(lohist),hi=np.asarray(hihist),contraction=np.asarray(factors),stats=stats)
