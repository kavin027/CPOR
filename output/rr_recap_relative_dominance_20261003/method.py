from pathlib import Path
import sys
R=Path(__file__).resolve().parent;O=R.parent;B=O/'rr_recap_evidence_extension_20261002'
sys.path[:0]=[str(O/'rr_recap_release_learning_20261002'),str(O/'rr_recap_multistep_20261002'),str(B),str(O/'rr_recap_robust_repair_20261002'),str(O/'rr_recap_coherent_state_20261003'),str(O/'rr_recap_pooling_20261003')]
from strict_solver import SafeProjection
from precision_release import release_repaired,certificate
from guard_v8 import guard
from engine import safe_bounds,certified
from controller import excess_bound
from rollout import advance,RR,K
from fractions import Fraction
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog

def strata(S):
    support=S>0;count=support.sum(1);depth=np.zeros(len(S),int)
    for j in np.argsort(count):
        children=np.flatnonzero((count<count[j])&np.all(~support|support[j],axis=1))
        if len(children):depth[j]=1+depth[children].max()
    return [np.flatnonzero(depth==d) for d in np.unique(depth)]

def factors(x,target,t,groups,kind):
    history=np.concatenate([x['z'][:36],target]);T=min(28,t)
    e=(x['truth'][t-T:t]-history[t-T:t])/np.sqrt(x['v'])
    out=[]
    for ids in groups:
        if kind=='NominalFull':A=np.zeros((1,len(ids)))
        else:
            eg=e[:,ids];diag=(eg*eg).mean(0)
            if kind=='DiagonalRDRFull':A=np.diag(np.sqrt((diag+1e-6)/T))
            else:A=np.vstack([np.sqrt(.9)*eg/T,np.diag(np.sqrt((.1*diag+1e-6)/T))])
        out.append(A)
    return out

def envelope(d,e,A):return float((d@d+2*d@e+2*np.linalg.norm(A@d))/len(d))
def grad_envelope(d,e,A):
    z=A@d;norm=np.linalg.norm(z)
    return 2*(d+e+(A.T@z/norm if norm>0 else 0))/len(d)

def outer_certificates(qp,target,reference,active,point,lower,upper,lw,uw,floor,margin_needed):
    n=len(target);G=[];rhs=[];scales=[];e=reference-target
    for ids,A in active:
        d=point[ids]-reference[ids];g=np.zeros(n);g[ids]=grad_envelope(d,e[ids],A)
        G.append(g);rhs.append(float(g@point-envelope(d,e[ids],A)));scales.append(1+float(np.mean(e[ids]**2)))
    G=np.asarray(G);rhs=np.asarray(rhs);scales=np.asarray(scales);upper_margin=0.
    if margin_needed:
        AA=np.vstack([np.c_[qp.M,np.zeros(n)],np.c_[-qp.M,np.zeros(n)],np.c_[G@qp.M,scales]])
        bb=np.r_[upper,-lower,rhs];c=np.r_[np.zeros(len(lw)),-1.];ll=np.r_[lw,0.];uu=np.r_[uw,1.]
        upper_margin=1.
        for iteration in range(32):
            lp=linprog(c,A_ub=AA,b_ub=bb,bounds=list(zip(ll,uu)),method='highs')
            if not lp.success:break
            mult=np.maximum(0.,-lp.ineqlin.marginals);red=c+AA.T@mult;lb=red@np.where(red>=0,ll,uu)-mult@bb;upper_margin=min(upper_margin,max(0.,-float(lb)))
            probe=qp.M@lp.x[:-1];actual=min(-envelope(probe[ids]-reference[ids],e[ids],A)/scale for (ids,A),scale in zip(active,scales))
            if upper_margin-actual<1e-7:break
            extra=[];erhs=[]
            for (ids,A),scale in zip(active,scales):
                dd=probe[ids]-reference[ids];gg=np.zeros(n);gg[ids]=grad_envelope(dd,e[ids],A)
                extra.append(np.r_[gg@qp.M,scale]);erhs.append(float(gg@probe-envelope(dd,e[ids],A)))
            AA=np.vstack([AA,np.array(extra)]);bb=np.r_[bb,erhs]
    xx=cp.Variable(n);ww=cp.Variable(len(lw));eq=xx==qp.M@ww;ineq=G@xx<=rhs-floor*scales
    problem=cp.Problem(cp.Minimize(cp.sum_squares(xx-target)/(2*n)),[eq,ineq,xx>=lower,xx<=upper,ww>=lw,ww<=uw])
    try:problem.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12,max_iter=500,max_threads=1,warm_start=False)
    except cp.error.SolverError:return upper_margin,0.
    if eq.dual_value is None or ineq.dual_value is None:return upper_margin,0.
    eta=np.asarray(eq.dual_value).ravel();mul=np.maximum(0.,np.asarray(ineq.dual_value).ravel());cx=eta+G.T@mul;cw=-qp.M.T@eta
    xd=np.clip(target-n*cx,lower,upper);wd=np.where(cw>=0,lw,uw)
    lb=float(.5*np.sum((xd-target)**2)/n+cx@xd+cw@wd-mul@(rhs-floor*scales))
    return upper_margin,max(0.,lb)

def safe_action(qp,b,m,ub,lo,hi,accounts):
    p,u,info=qp.solve(b,m,ub,lo,hi);pp=np.clip(p,lo,hi)
    for _ in range(16):
        if certified(b,pp,accounts,RR,K,True):break
        for j in range(len(b)):
            if not certified(b[j:j+1],pp[j:j+1],accounts[j:j+1],RR,K,True):pp[j]=np.nextafter(pp[j],b[j])
    if np.allclose(pp,qp.S@u,rtol=1e-8,atol=2e-6) and certified(b,pp,accounts,RR,K,True):return pp,u,info
    p,u,c=guard(b,p,ub,u,qp.S,lo,hi,accounts,RR,K);info['guard_contraction']=c
    return p,u,info

def select(qp,b,m,ub,lo,hi,q,uq,groups,As,accounts):
    n=len(b);s=qp.s;target=(m-b)/s;reference=(q-b)/s;e=reference-target
    lower=(lo-b)/s;upper=(hi-b)/s;active=[];fixed=0
    # Exact analytical reductions for constraints with no interior.
    for ids,A in zip(groups,As):
        if len(ids)==1:
            j=ids[0];radius=np.linalg.norm(A);mag=max(0.,abs(e[j])-radius)
            dl=-2*mag if e[j]>0 else 0.;du=2*mag if e[j]<0 else 0.
            lower[j]=max(lower[j],reference[j]+dl);upper[j]=min(upper[j],reference[j]+du)
            if mag==0:fixed+=1
            else:active.append((ids,A))
        elif np.all(A==0) and np.all(e[ids]==0):
            lower[ids]=reference[ids];upper[ids]=reference[ids];fixed+=1
        elif np.any(A) and float(e[ids]@np.linalg.solve(A.T@A,e[ids]))<=1.:
            lower[ids]=reference[ids];upper[ids]=reference[ids];fixed+=1
        else:active.append((ids,A))
    # q is feasible in exact arithmetic; include its represented coordinate.
    for ids in groups:
        radius=np.linalg.norm(e[ids]);lower[ids]=np.maximum(lower[ids],target[ids]-radius);upper[ids]=np.minimum(upper[ids],target[ids]+radius)
    lower=np.minimum(lower,reference);upper=np.maximum(upper,reference)
    if not np.any(upper>lower):return q.copy(),uq.copy(),dict(gap=0.,objective=float(.5*np.sum(e*e)/n),fallback=False,fixed_groups=fixed),np.zeros(len(groups))
    sub=qp.S[:,qp.cols];cap=np.min(np.divide((b+s*upper)[:,None],sub,out=np.full_like(sub,np.inf),where=sub>0),axis=0)
    lw=-ub[qp.cols]/qp.cs;uw=(np.maximum(cap,0)-ub[qp.cols])/qp.cs
    xx=cp.Variable(n);ww=cp.Variable(len(lw));eq=xx==qp.M@ww
    base=[eq,xx>=lower,xx<=upper,ww>=lw,ww<=uw];risk=[];gscales=[];exprs=[]
    for ids,A in active:
        d=xx[ids]-reference[ids];scale=1+float(np.mean(e[ids]**2));gscales.append(scale)
        expr=(cp.sum_squares(d)+2*e[ids]@d+2*cp.norm(A@d,2))/len(ids)
        exprs.append(expr)
    oscale=1+float(.5*np.sum(e*e)/n)
    beta=cp.Variable();margin_risk=[expr/scale+beta<=0 for expr,scale in zip(exprs,gscales)]
    margin_problem=cp.Problem(cp.Minimize(-beta),base+[beta>=0,beta<=1]+margin_risk)
    best=None;attempts=[]
    for tol in [1e-10,1e-12,1e-13]:
        upper_margin=0.;best_margin=0.
        if not fixed:
            try:margin_problem.solve(solver='CLARABEL',tol_gap_abs=tol,tol_gap_rel=tol,tol_feas=tol,max_iter=500,max_threads=1,warm_start=False)
            except cp.error.SolverError as err:attempts.append(str(err));continue
            if beta.value is None or xx.value is None:continue
            bx=np.asarray(xx.value).ravel();bw=np.asarray(ww.value).ravel();bv=float(beta.value);be=np.asarray(eq.dual_value).ravel()
            bgx=be.copy();bgw=-qp.M.T@be;bgb=-1.;blag=-bv+be@(bx-qp.M@bw)
            for (ids,A),constraint,gscale in zip(active,margin_risk,gscales):
                mul=max(0.,float(constraint.dual_value));dd=bx[ids]-reference[ids]
                blag+=mul*(envelope(dd,e[ids],A)/gscale+bv);bgx[ids]+=mul*grad_envelope(dd,e[ids],A)/gscale;bgb+=mul
            blb=blag+bgx@(np.where(bgx>=0,lower,upper)-bx)+bgw@(np.where(bgw>=0,lw,uw)-bw)+bgb*((0. if bgb>=0 else 1.)-bv)
            upper_margin=max(0.,-float(blb));best_margin=max(0.,min(1.,bv))
        floor=max(0.,best_margin-1e-8)
        risk=[expr/scale+floor<=0 for expr,scale in zip(exprs,gscales)]
        problem=cp.Problem(cp.Minimize(cp.sum_squares(xx-target)/(2*n*oscale)),base+risk)
        try:problem.solve(solver='CLARABEL',tol_gap_abs=tol,tol_gap_rel=tol,tol_feas=tol,max_iter=500,max_threads=1,warm_start=False)
        except cp.error.SolverError as err:attempts.append(str(err));continue
        if ww.value is None:continue
        x0=np.asarray(xx.value).ravel();w0=np.asarray(ww.value).ravel();u=ub.copy();u[qp.cols]+=qp.cs*w0;u=np.maximum(u,0);p=u@qp.S.T
        # Only reference-directed repair preserves all convex dominance constraints.
        lam=1.;p0=p.copy();u0=u.copy();actual=None
        tolerances=np.array([1e-8*(1+np.mean(e[ids]**2)) for ids in groups])
        for step in range(45):
            p=q+lam*(p0-q);u=uq+lam*(u0-uq);d=(p-q)/s
            actual=np.array([envelope(d[ids],e[ids],A) for ids,A in zip(groups,As)])
            if u.min()>=0 and np.allclose(p,u@qp.S.T,rtol=1e-8,atol=2e-6) and certified(b,p,accounts,RR,K,True) and np.all(actual<=tolerances):break
            lam=np.nextafter(lam,0.) if step<8 else lam*.5
        else:p=q.copy();u=uq.copy();lam=0.;actual=np.zeros(len(groups))
        if .5*np.sum(((p-m)/s)**2)>.5*np.sum(((q-m)/s)**2)+1e-10*(1+np.sum(e*e)):
            p=q.copy();u=uq.copy();lam=0.;actual=np.zeros(len(groups))
        # Supporting-plane lower bound on the Lagrangian, minimized over finite boxes.
        eta=np.asarray(eq.dual_value).ravel()*oscale
        gx=(x0-target)/n+eta;gw=-qp.M.T@eta
        lag=float(.5*np.sum((x0-target)**2)/n+eta@(x0-qp.M@w0))
        for (ids,A),constraint,gscale in zip(active,risk,gscales):
            mul=max(0.,float(constraint.dual_value))*oscale/gscale;d0=x0[ids]-reference[ids]
            lag+=mul*(envelope(d0,e[ids],A)+floor*gscale);gx[ids]+=mul*grad_envelope(d0,e[ids],A)
        lowerbound=lag+gx@(np.where(gx>=0,lower,upper)-x0)+gw@(np.where(gw>=0,lw,uw)-w0)
        obj=float(.5*np.sum(((p-m)/s)**2)/n);gap=max(0.,float(obj-lowerbound))
        actual_margin=float(min(-envelope(((p-q)/s)[ids],e[ids],A)/(1+np.mean(e[ids]**2)) for ids,A in zip(groups,As)))
        margin_gap=max(0.,upper_margin-actual_margin)
        if gap>2e-6*(1+obj) or margin_gap>2e-6:
            outer_margin,outer_lb=outer_certificates(qp,target,reference,active,x0,lower,upper,lw,uw,floor,not fixed)
            gap=min(gap,max(0.,obj-outer_lb));upper_margin=min(upper_margin,outer_margin) if not fixed else 0.;margin_gap=max(0.,upper_margin-actual_margin)
        info=dict(objective=obj,gap=gap,relative_gap=gap/(1+obj),solver_status=problem.status,iterations=int(problem.solver_stats.num_iters),repair_contraction=lam,fallback=lam==0,fixed_groups=fixed,attempts=len(attempts)+1,common_margin=actual_margin,stage1_margin=best_margin,margin_gap=margin_gap)
        if best is None or gap/(1+obj)<best[2]['gap']/(1+best[2]['objective']):best=(p,u,info,actual)
        if gap<=2e-6*(1+obj) and margin_gap<=2e-6:return p,u,info,actual
        attempts.append(dict(info))
    if best is not None:best[2]['attempt_records']=attempts;return best
    return q.copy(),uq.copy(),dict(objective=float(.5*np.sum(e*e)/n),gap=float(.5*np.sum(e*e)/n),fallback=True,solver_failure=attempts,fixed_groups=fixed),np.zeros(len(groups))

def release(x,target,kind):
    if kind in ['CovFullFix','CovSegmentFix']:
        return release_repaired(x,target,'Segment' if kind=='CovSegmentFix' else 'Full')
    groups=strata(x['S'])
    if kind=='LevelWeightedFull':
        xx=dict(x);v=x['v'].copy()
        for ids in groups:v[ids]*=len(ids)
        xx['v']=v;return release_repaired(xx,target,'Full')
    qp=SafeProjection(x['S'],x['v']);accounts=[Fraction(0) for _ in x['v']];previous=None;reservation=None;stats=[]
    out={k:[] for k in ['pred','witness','available_after','reserved','lo','hi','reference','reference_witness','risk_envelope','objective','gap']}
    for i,t in enumerate(range(36,len(x['truth']))):
        if i:accounts=advance([a+r for a,r in zip(accounts,reservation)],x['b'][t-1],previous,x['truth'][t-1])
        b,ub,m=x['b'][t],x['ub'][t],target[i];lo,hi=safe_bounds(b,np.array([float(a) for a in accounts]))
        ql=np.maximum(lo,np.minimum(b,m));qh=np.minimum(hi,np.maximum(b,m))
        q,uq,qinfo=safe_action(qp,b,m,ub,ql,qh,accounts)
        As=factors(x,target,t,groups,kind)
        p,u,info,riskvec=select(qp,b,m,ub,lo,hi,q,uq,groups,As,accounts)
        risk=excess_bound(b,p,RR,K)
        for j in range(len(accounts)):accounts[j]-=risk[j]
        assert min(accounts)>=0
        vals=dict(pred=p,witness=u,available_after=[float(a) for a in accounts],reserved=[float(a) for a in risk],lo=lo,hi=hi,reference=q,reference_witness=uq,risk_envelope=riskvec,objective=info['objective'],gap=info['gap'])
        for key,value in vals.items():out[key].append(value)
        info['reference_solver']=qinfo;stats.append(info);previous=p;reservation=risk
    return {**{k:np.asarray(a) for k,a in out.items()},'stats':stats}
