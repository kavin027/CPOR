"""Matched current squared-risk control, not a proposed new algorithm."""
import time,math
import numpy as np
import cvxpy as cp
import clarabel
from scipy import sparse

ALPHA=.1;GAP=2e-5;TOLS=(1e-8,1e-10,1e-12);MAX_ITER=400;TIME_LIMIT=120.

def objective(records,theta,kappa):
    value=ALPHA/2*np.sum((theta-np.eye(len(theta))[0])**2)
    for r in records:
        e=r['b']+(r['P']-r['b'][:,None])@theta-r['y']
        value+=kappa*float(np.mean(e*e))
    return float(value)

def terms(records,kappa):
    J=records[0]['P'].shape[1];n=len(records[0]['b'])
    H=ALPHA*np.eye(J);c=-ALPHA*np.eye(J)[0]
    for r in records:
        D=r['P']-r['b'][:,None];by=r['b']-r['y']
        H+=2*kappa/n*(D.T@D);c+=2*kappa/n*(D.T@by)
    return (H+H.T)/2,c

def polish(H,c,theta):
    best=theta.copy();best_q=float(.5*best@H@best+c@best)
    for cutoff in (1e-12,1e-10,1e-8,1e-6):
        ids=np.flatnonzero(theta>cutoff)
        if not len(ids):continue
        block=np.block([[H[np.ix_(ids,ids)],np.ones((len(ids),1))],[np.ones((1,len(ids))),np.zeros((1,1))]])
        try:z=np.linalg.solve(block,np.r_[-c[ids],1.])[:-1]
        except np.linalg.LinAlgError:continue
        if np.min(z)<-1e-12:continue
        th=np.zeros(len(theta));th[ids]=np.maximum(z,0);th/=th.sum()
        v=float(.5*th@H@th+c@th)
        if v<=best_q+1e-12*(1+abs(best_q)):best=th;best_q=v
    return best

def fit(records,kappa):
    begin=time.perf_counter();H,c=terms(records,kappa);J=len(c)
    scale=max(1.,objective(records,np.eye(J)[-1],kappa))
    # Native QP over the same simplex; centered physical objective evaluated below.
    A=sparse.vstack([sparse.csc_matrix(np.ones((1,J))),sparse.eye(J),-sparse.eye(J)],format='csc')
    rhs=np.r_[1.,np.ones(J),np.zeros(J)]
    cones=[clarabel.ZeroConeT(1),clarabel.NonnegativeConeT(2*J)]
    attempts=[]
    for tol in TOLS:
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.max_iter=MAX_ITER;settings.time_limit=TIME_LIMIT
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=tol
        settings.iterative_refinement_max_iter=20
        solver=clarabel.DefaultSolver(sparse.csc_matrix(H/scale),c/scale,A,rhs,cones,settings)
        sol=solver.solve()
        try:
            th=np.maximum(np.asarray(sol.x),0);th/=th.sum();th=polish(H,c,th)
            # Compute the supporting gradient from residuals, avoiding a large Hth+c cancellation.
            g=ALPHA*(th-np.eye(J)[0])
            for r in records:
                D=r['P']-r['b'][:,None];e=r['b']+D@th-r['y']
                g+=2*kappa/len(e)*(D.T@e)
            gap=max(0.,math.fsum(float(g[i]*th[i]) for i in range(J))-float(g.min()))
            gap+=64*np.finfo(float).eps*(1+float(abs(g)@th)+float(abs(g).max()))
            obj=objective(records,th,kappa);relative=gap/(1+abs(obj))
            info=dict(status=str(sol.status),objective=obj,gap=gap,relative_gap=relative,records=len(records),
                      normalized_objective=obj/len(records),normalized_gap=gap/len(records),tol=tol,
                      iterations=sol.iterations,seconds=time.perf_counter()-begin,
                      certificate='convex supporting-plane FW gap; same original-objective tolerance',kappa=kappa)
            attempts.append(info)
            if relative<=GAP:return th,{**info,'attempts':attempts,'passed':True}
        except Exception as err:attempts.append(dict(status=str(sol.status),tol=tol,error=repr(err)))
    raise RuntimeError(dict(mode='Square',attempts=attempts,records=len(records)))
