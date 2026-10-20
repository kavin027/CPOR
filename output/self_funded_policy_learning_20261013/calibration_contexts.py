"""Numerical polish of the unchanged original calibration covariance QP."""
import numpy as np
from scipy.optimize import minimize

def weights(Q,original):
    try:return original.solve_weights(Q)[0]
    except (AssertionError,np.linalg.LinAlgError):
        out=[]
        for q in Q:
            J=len(q);w=np.ones(J)/J
            sol=minimize(lambda a:float(a@q@a),w,jac=lambda a:2*q@a,method='SLSQP',bounds=[(0,1)]*J,
                constraints=[dict(type='eq',fun=lambda a:a.sum()-1,jac=lambda a:np.ones(J))],options=dict(ftol=1e-15,maxiter=3000))
            w=np.maximum(sol.x,0);w/=w.sum()
            for threshold in [1e-12,1e-10,1e-8,1e-6]:
                ids=np.flatnonzero(w>threshold)
                if not len(ids):continue
                a=np.linalg.solve(q[np.ix_(ids,ids)],np.ones(len(ids)));a/=a.sum()
                if a.min()<0:continue
                candidate=np.zeros(J);candidate[ids]=a
                oldg=2*q@w;newg=2*q@candidate
                if newg@candidate-newg.min()<oldg@w-oldg.min():w=candidate
            g=2*q@w;obj=float(w@q@w);gap=float(g@w-g.min())
            assert gap<=1e-7*(1+obj),(gap,obj)
            out.append(w)
        return np.array(out)

def make_contexts(x,bank,saved,original):
    S,v=x['S'],x['v'];targets=[]
    for t in range(36):
        w=np.full_like(bank[t],1/9) if t==0 else weights(original.cov.moment(bank[:t],x['truth'][:t],v,t),original.cov)
        targets.append(np.sum(bank[t]*w,axis=1))
    cal,_=original.project(S,np.array(targets),v)
    # Saved scoring targets are reused byte-for-byte, never refitted by score.
    e=np.moveaxis(bank,2,1).reshape(-1,len(v));z,_=original.project(S,e,v)
    expert=np.moveaxis(z.reshape(len(bank),9,len(v)),1,2)
    return np.concatenate([cal,saved['target']]),expert
