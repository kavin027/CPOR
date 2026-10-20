import numpy as np
import cvxpy as cp

class Planner:
    def __init__(self,S,v,H,segment=False):
        self.S=S;self.s=np.sqrt(v);self.H=H;self.R=1.005**2;self.k=self.R-1
        n,m=S.shape;norm=np.max(S/self.s[:,None],axis=0);self.cols=np.flatnonzero(norm>0);self.cs=1/norm[self.cols]
        self.M=S[:,self.cols]*self.cs/self.s[:,None]
        self.x=cp.Variable((H,n));self.w=cp.Variable((H,len(self.cols)));self.a=cp.Variable((H-1,n),nonneg=True)
        self.d=cp.Parameter((H,n));self.bs=cp.Parameter((H,n),nonneg=True);self.lw=cp.Parameter((H,len(self.cols)))
        self.rhs=cp.Parameter((H-1,n),nonneg=True);self.lo=cp.Parameter(n);self.hi=cp.Parameter(n)
        con=[self.x==self.w@self.M.T,self.w>=self.lw,self.x>=-self.bs,self.x[0]>=self.lo,self.x[0]<=self.hi]
        if segment:
            self.seglo=cp.Parameter((H,n));self.seghi=cp.Parameter((H,n));con.extend([self.x>=self.seglo,self.x<=self.seghi])
        for h in range(1,H):
            q=self.rhs[h-1]-2*self.R*cp.multiply(self.bs[h],self.a[h-1])
            vec=cp.vstack([cp.reshape(2*np.sqrt(self.R/self.k)*(self.x[h]-self.a[h-1]),(1,n),order='C'),
                cp.reshape(2*np.sqrt(self.R)*self.a[h-1],(1,n),order='C'),2*(self.x[:h]-self.d[:h]),cp.reshape(q-1,(1,n),order='C')])
            con.append(cp.SOC(q+1,vec,axis=0))
        self.problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(self.x-self.d)),con)
        assert self.problem.is_dcp() and self.problem.is_dpp()

    def solve(self,b,ub,mu,credit,lo,hi):
        d=(mu-b)/self.s
        scale=max(1.,float(np.max(abs(d))),float(np.sqrt(np.max(credit/(self.s*self.s)))))
        self.d.value=d/scale;self.bs.value=b/self.s/scale;self.lw.value=-ub[:,self.cols]/self.cs/scale
        rhs=credit/(self.s*self.s)+self.R*np.cumsum(d*d,axis=0)[:-1];self.rhs.value=np.maximum(rhs,0)/(scale*scale)
        self.lo.value=(lo-b[0])/self.s/scale;self.hi.value=(hi-b[0])/self.s/scale
        if hasattr(self,'seglo'):self.seglo.value=np.minimum(d,0)/scale;self.seghi.value=np.maximum(d,0)/scale
        trials=[]
        for tol in [1e-9,1e-11,1e-13]:
            try:self.problem.solve(solver='CLARABEL',max_iter=500,max_threads=1,tol_gap_abs=tol,tol_gap_rel=tol,tol_feas=tol,warm_start=False,equilibrate_max_iter=30,equilibrate_max_scaling=1e12,equilibrate_min_scaling=1e-12,static_regularization_constant=1e-10)
            except cp.error.SolverError as e:trials.append(dict(error=str(e)));continue
            if self.w.value is None:continue
            u=ub.copy();u[:,self.cols]+=scale*self.cs*np.asarray(self.w.value);u=np.maximum(u,0);p=u@self.S.T;x=(p-b)/self.s
            # Check the actual nonlinear reserve function, not just conic auxiliaries.
            lift=np.maximum(0,(x-self.k*b/self.s)/self.R)
            risk=self.R/self.k*(x-lift)**2+self.R*lift**2+2*self.R*b/self.s*lift
            future_error=risk[1:]+np.cumsum((x-d)**2,axis=0)[:-1]-rhs
            budget_error=float(np.max(np.maximum(future_error,0)/(1+rhs)))
            node_scale=np.maximum(1,np.maximum(abs(b),abs(mu)))
            segerr=float(np.max(np.maximum(0,np.maximum(np.minimum(b,mu)-p,p-np.maximum(b,mu)))/node_scale)) if hasattr(self,'seglo') else 0.
            info=dict(status=self.problem.status,iterations=int(self.problem.solver_stats.num_iters),objective=float(.5*np.sum((x-d)**2)),budget_relative_error=budget_error,segment_relative_error=segerr,tolerance=tol,attempts=len(trials)+1,coordinate_scale=scale)
            trials.append(info)
            if budget_error<=2e-7 and segerr<=2e-10:return p,u,info
        raise RuntimeError(str(trials))
