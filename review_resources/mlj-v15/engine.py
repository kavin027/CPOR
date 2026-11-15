import numpy as np
from scipy.optimize import minimize,linprog
from fractions import Fraction
EPS=.005
MARGIN=1e-6
def safe_bounds(b,credit,epsilon=EPS,nonnegative=True):
    rr=(1+epsilon)**2;k=rr-1
    assert np.all(b>=0) and np.all(credit>=0)
    R=np.sqrt(k/rr*credit)
    lo=np.maximum(0,b-R);hi=b+R
    if nonnegative:hi=np.where(credit<=k*rr*b*b,hi,np.sqrt(rr*b*b+credit))
    return b+(1-MARGIN)*(lo-b),b+(1-MARGIN)*(hi-b)

def independent_rows(E):
    if not len(E):return E
    norm=np.linalg.norm(E,axis=1);E=E[norm>0]/norm[norm>0,None]
    if not len(E):return E
    _,s,V=np.linalg.svd(E,full_matrices=False)
    rank=int(np.sum(s>max(E.shape)*np.finfo(float).eps*s[0]*10))
    return V[:rank]

class SafeProjection:
    def __init__(self,S,v):
        self.S=np.asarray(S,float);self.s=np.sqrt(v);self.n,self.k=S.shape
        assert np.all(S>=0) and np.all(self.s>0)
        bottom=[]
        for j in range(self.k):
            unit=np.zeros(self.k);unit[j]=1
            ids=np.flatnonzero(np.all(S==unit,axis=1))
            if len(ids)!=1:break
            bottom.append(ids[0])
        self.bottom=np.array(bottom,dtype=int) if len(bottom)==self.k else None
        if self.bottom is not None:
            agg=np.setdiff1d(np.arange(self.n),self.bottom)
            A=np.eye(self.n)[agg];A[:,self.bottom]-=S[agg]
            self.E=A*self.s
        else:
            norm=np.max(S/self.s[:,None],axis=0)
            self.cols=np.flatnonzero(norm>0);self.cs=1/norm[self.cols]
            self.M=S[:,self.cols]*self.cs/self.s[:,None]

    def solve(self,b,m,ub,lo,hi):
        target=(m-b)/self.s;lower=(lo-b)/self.s;upper=(hi-b)/self.s
        assert np.all(lower<=0) and np.all(upper>=0)
        free=np.flatnonzero(upper>lower);nf=len(free)
        if not nf:return b.copy(),ub.copy(),dict(objective=float(.5*target@target),gap=0.,iterations=0,attempts=0,equality=0.)
        bounds=[(float(lower[j]),float(upper[j])) for j in free]
        if self.bottom is not None:
            E=independent_rows(self.E[:,free]);x0=np.zeros(nf);wc=None
        else:
            Ssub=self.S[:,self.cols]
            cap=np.min(np.divide(hi[:,None],Ssub,out=np.full_like(Ssub,np.inf),where=Ssub>0),axis=0)
            # A finite upper witness bound is implied by nonnegative S and node caps.
            lw=-ub[self.cols]/self.cs;uw=(np.maximum(cap,ub[self.cols])-ub[self.cols])/self.cs
            wc=np.flatnonzero(uw>lw)
            E=independent_rows(np.column_stack([-np.eye(self.n)[:,free],self.M[:,wc]]))
            bounds += [(float(lw[j]),float(uw[j])) for j in wc]
            x0=np.zeros(nf+len(wc))
        const=.5*np.sum(target[np.setdiff1d(np.arange(self.n),free)]**2)
        def fun(x):return .5*np.sum((x[:nf]-target[free])**2)+const
        def jac(x):
            g=np.zeros_like(x);g[:nf]=x[:nf]-target[free];return g
        constraints=[] if not len(E) else [dict(type='eq',fun=lambda x:E@x,jac=lambda x:E)]
        attempts=[]
        for ftol,maxiter,scaled in [(1e-11,300,False),(1e-13,1200,False),(1e-14,3000,True)]:
            divisor=max(1.,float(np.max(np.abs(target)))) if scaled else 1.
            if scaled:x0=np.zeros_like(x0)
            result=minimize(lambda x:fun(x)/divisor,x0,jac=lambda x:jac(x)/divisor,bounds=bounds,constraints=constraints,method='SLSQP',options=dict(ftol=ftol,maxiter=maxiter))
            x=result.x;g=jac(x)
            lp=linprog(g,A_eq=E if len(E) else None,b_eq=np.zeros(len(E)) if len(E) else None,bounds=bounds,method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            # Retry only failed multiplier estimation. Acceptance still uses the
            # analytic QP dual gap and original feasibility tolerances below.
            if not lp.success:
                lp=linprog(g,A_eq=E if len(E) else None,b_eq=np.zeros(len(E)) if len(E) else None,bounds=bounds,method='highs-ds',options=dict(presolve=False))
            gap=float(g@x-lp.fun) if lp.success else float('inf');eq=float(np.max(np.abs(E@x))) if len(E) else 0.
            bound_error=max(0.,max(max(l-xx,xx-h) for xx,(l,h) in zip(x,bounds)))
            # Any equality multiplier supplies a dual bound for this convex QP.
            # Minimize its Lagrangian over the box analytically, avoiding error
            # in the LP primal solution when measuring the optimality gap.
            if lp.success:
                c=-(E.T@lp.eqlin.marginals) if len(E) else np.zeros_like(x)
                blo=np.array([a[0] for a in bounds]);bhi=np.array([a[1] for a in bounds])
                xd=np.where(c>=0,blo,bhi)
                xd[:nf]=np.clip(target[free]-c[:nf],blo[:nf],bhi[:nf])
                xc=np.clip(x,blo,bhi);dx=xc-xd
                laggap=.5*np.sum(dx[:nf]**2)+np.sum((xd[:nf]-target[free]+c[:nf])*dx[:nf])+c[nf:]@dx[nf:]
                dualgap=float(max(0,laggap)+abs(c@xc))
            else:dualgap=float('inf')
            attempts.append(dict(success=bool(result.success),message=result.message,iterations=int(result.nit),gap=gap,dual_gap=dualgap,equality=eq,objective_divisor=divisor))
            # Convex objective: independent feasibility and LP gap certify optimality,
            # even if SLSQP's line-search status reports stagnation at that optimum.
            trial_delta=np.zeros(self.n);trial_delta[free]=x[:nf]*self.s[free]
            if self.bottom is not None:trial_u=(b+trial_delta)[self.bottom]
            else:
                trial_u=ub.copy();trial_u[self.cols[wc]]+=self.cs[wc]*x[nf:]
            reconstruction_ok=trial_u.min()>=-1e-8 and np.allclose(np.maximum(trial_u,0.)@self.S.T,b+trial_delta,rtol=1e-8,atol=2e-6)
            if bound_error<=1e-9 and eq<=2e-8 and dualgap<=2e-6*(1+fun(x)) and reconstruction_ok:break
            x0=x
        else:raise RuntimeError(str(attempts))
        delta=np.zeros(self.n);delta[free]=x[:nf]*self.s[free]
        if self.bottom is not None:u=(b+delta)[self.bottom]
        else:
            u=ub.copy();u[self.cols[wc]]+=self.cs[wc]*x[nf:]
        assert u.min()>=-1e-8;u=np.maximum(u,0.)
        pred=u@self.S.T
        np.testing.assert_allclose(pred,b+delta,rtol=1e-8,atol=2e-6)
        return pred,u,dict(objective=float(fun(x)),gap=dualgap,lp_primal_gap=gap,iterations=sum(a['iterations'] for a in attempts),attempts=len(attempts),equality=eq,solver_success=bool(result.success),bound_error=bound_error)

def certified(b,p,accounts,rr,k,nonnegative):
    for bb,pp,B in zip(b,p,accounts):
        bd=Fraction.from_float(float(bb));pd=Fraction.from_float(float(pp))
        worst=pd*pd-rr*bd*bd if nonnegative and pd>rr*bd else rr*(pd-bd)**2/k
        if worst>B:return False
    return True