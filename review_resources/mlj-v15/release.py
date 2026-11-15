import numpy as np
import cvxpy as cp
from fractions import Fraction
from engine import safe_bounds, certified
RR=Fraction(40401,40000)
K=RR-1
class SafeProjection:
    """Same weighted QP, centered latent witness, checked box dual certificate."""
    def __init__(self,S,v):
        self.S=np.asarray(S,float);self.s=np.sqrt(v); n,k=S.shape
        norm=np.max(S/self.s[:,None],axis=0);self.cols=np.flatnonzero(norm>0)
        self.cs=1/norm[self.cols];self.M=S[:,self.cols]*self.cs/self.s[:,None]
        self.x=cp.Variable(n); self.w=cp.Variable(len(self.cols))
        self.target=cp.Parameter(n);self.lo=cp.Parameter(n);self.hi=cp.Parameter(n)
        self.lw=cp.Parameter(len(self.cols));self.uw=cp.Parameter(len(self.cols))
        self.eq=self.x==self.M@self.w
        self.problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(self.x-self.target)),
            [self.eq,self.x>=self.lo,self.x<=self.hi,self.w>=self.lw,self.w<=self.uw])
    def solve(self,b,m,ub,lo,hi):
        target=(m-b)/self.s; lower=(lo-b)/self.s;upper=(hi-b)/self.s
        if not np.any(upper>lower):return b.copy(),ub.copy(),dict(objective=float(.5*target@target),gap=0.,iterations=0,attempts=0,solver='Clarabel')
        sub=self.S[:,self.cols]
        cap=np.min(np.divide(hi[:,None],sub,out=np.full_like(sub,np.inf),where=sub>0),axis=0)
        lw=-ub[self.cols]/self.cs;uw=(np.maximum(cap,ub[self.cols])-ub[self.cols])/self.cs
        self.target.value=target;self.lo.value=lower;self.hi.value=upper;self.lw.value=lw;self.uw.value=uw
        attempts=[]
        for tol,divisor in [(1e-11,1.),(1e-13,1.),(1e-14,1.),(1e-13,1000.),(1e-13,1e6)]:
            problem=self.problem if divisor==1 else cp.Problem(cp.Minimize(self.problem.objective.expr/divisor),self.problem.constraints)
            try:
                problem.solve(solver='CLARABEL',tol_gap_abs=tol,tol_gap_rel=tol,tol_feas=tol,max_iter=500,max_threads=1,warm_start=False)
            except cp.error.SolverError as error:
                attempts.append(dict(solver_error=str(error),tol=tol,objective_divisor=divisor));continue
            if self.w.value is None:continue
            w=np.asarray(self.w.value).ravel(); u=ub.copy();u[self.cols]+=self.cs*w
            u=np.maximum(u,0);p=u@self.S.T;x=(p-b)/self.s;actualw=(u[self.cols]-ub[self.cols])/self.cs
            eta=np.asarray(self.eq.dual_value).ravel()*divisor;cw=-self.M.T@eta
            xd=np.clip(target-eta,lower,upper);wd=np.where(cw>=0,lw,uw)
            # Minimum Lagrangian over finite boxes; stable evaluation of primal-dual difference.
            dx=x-xd;dw=actualw-wd
            gap=.5*(dx@dx)+(xd-target+eta)@dx+cw@dw-eta@(x-self.M@actualw)
            objective=.5*np.sum((x-target)**2)
            bounds=max(float(np.max(lower-x)),float(np.max(x-upper)),0.)
            boundw=max(float(np.max(lw-actualw)),float(np.max(actualw-uw)),0.)
            info=dict(objective=float(objective),gap=float(gap),iterations=int(problem.solver_stats.num_iters),attempts=len(attempts)+1,solver='Clarabel',status=problem.status,objective_divisor=divisor,bound_error=bounds,witness_bound_error=boundw,equality=float(np.max(np.abs(x-self.M@actualw))))
            attempts.append(info)
            scale=np.maximum(1,np.maximum(abs(lo),abs(hi)))
            original_error=max(0.,float(np.max((lo-p)/scale)),float(np.max((p-hi)/scale)))
            info['original_relative_box_error']=original_error
            if bounds<=1e-8 and boundw<=1e-8 and original_error<=2e-10 and -1e-7*(1+objective)<=gap<=2e-6*(1+objective):return p,u,info
        from engine import SafeProjection as Original
        p,u,info=Original(self.S,self.s*self.s).solve(b,m,ub,lo,hi)
        info['solver']='SLSQP_checked_fallback';info['clarabel_attempts']=attempts
        return p,u,info

def guard(b,p,ub,u,S,lo,hi,accounts,rr,k):
    delta=p-b;du=u-ub
    if np.all(p>=lo) and np.all(p<=hi) and certified(b,p,accounts,rr,k,True):return p,u,1.
    ratios=np.ones_like(delta);pos=delta>0;neg=delta<0
    ratios[pos]=(hi[pos]-b[pos])/delta[pos];ratios[neg]=(lo[neg]-b[neg])/delta[neg]
    lam=max(0.,min(1.,float(ratios.min())))
    # Bounds already have their frozen interior margin. Search representable convex
    # contractions and independently certify every proposed released float.
    for i in range(40):
        lam=np.nextafter(lam,0.) if i<8 else lam*(1-1e-12) if i<24 else lam*.5
        pp=b+lam*delta;uu=ub+lam*du
        if uu.min()>=0 and np.all(pp>=lo) and np.all(pp<=hi) and certified(b,pp,accounts,rr,k,True):
            np.testing.assert_allclose(pp,uu@S.T,rtol=1e-8,atol=2e-6)
            return pp,uu,lam
    return b.copy(),ub.copy(),0.

def one_action(qp,S,b,ub,target,A):
    lo,hi=safe_bounds(b,np.array([float(a) for a in A]))
    if not np.any(hi>lo):return b.copy(),ub.copy(),dict(gap=0.,objective=float(.5*np.sum((b-target)**2/qp.s**2)),contraction=1.,rounding_move=0.,lo=lo,hi=hi)
    p,u,info=qp.solve(b,target,ub,lo,hi);initial=p.copy();pp=np.clip(p,lo,hi)
    for _ in range(16):
        if certified(b,pp,A,RR,K,True):break
        for j in range(len(b)):
            if not certified(b[j:j+1],pp[j:j+1],A[j:j+1],RR,K,True):pp[j]=np.nextafter(pp[j],b[j])
    if np.allclose(pp,S@u,rtol=1e-8,atol=2e-6) and certified(b,pp,A,RR,K,True):p=pp;c=1.
    else:p,u,c=guard(b,p,ub,u,S,lo,hi,A,RR,K)
    # Update the original solver certificate for the repaired physical point.
    obj=float(.5*np.sum((p-target)**2/qp.s**2))
    gap=info['gap']+obj-info['objective']
    assert gap<=2e-6*(1+obj),(gap,obj)
    return p,u,{**info,'gap':max(0.,float(gap)),'objective':obj,'contraction':c,'rounding_move':float(abs(p-initial).max()),'lo':lo,'hi':hi}