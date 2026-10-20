import numpy as np
import cvxpy as cp

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
        for tol,divisor in [(1e-9,1.),(1e-11,1.),(1e-13,1.),(1e-11,1000.),(1e-11,1e6)]:
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
            if bounds<=1e-8 and boundw<=1e-8 and -1e-7*(1+objective)<=gap<=2e-6*(1+objective):return p,u,info
        from engine import SafeProjection as Original
        p,u,info=Original(self.S,self.s*self.s).solve(b,m,ub,lo,hi)
        info['solver']='SLSQP_checked_fallback';info['clarabel_attempts']=attempts
        return p,u,info
