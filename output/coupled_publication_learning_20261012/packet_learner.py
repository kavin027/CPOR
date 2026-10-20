"""Convex mature-publication loss. No future labels or policy-account transfer.

The certificate is a floating-point Lagrangian supporting-plane bound, not an
interval-arithmetic proof. Historical packets and numerical residuals are kept.
"""
import time
import gc
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

R=40401/40000
K=401/40000
INSET=1-1e-6
ALPHA=.1
FIT_REL_GAP=2e-5
_PINV_CACHE={}

def gamma(b,p):
    delta=p-b
    return np.where(delta<=K*b,R/K*delta*delta,delta*delta+2*b*delta-K*b*b)

def upper_bound(b,A,c):
    radius=np.sqrt(K/R*np.maximum(A,0))
    hi=np.where(A<=K*R*b*b,b+radius,np.sqrt(R*b*b+A))
    return b+c*(hi-b)

def fit(packets,S,v,future=True,alpha=ALPHA,inset=INSET):
    """Packets contain only already returned labels and current known context.

    The SOCP lifts all future projections jointly. It returns the global
    parameter-loss lower bound and primal residual, never an oracle label.
    """
    gc.collect()  # CVXPY compilation graphs contain cycles across daily fits.
    begin=time.perf_counter();s=np.sqrt(v);n=len(v)
    J=packets[0]['P'].shape[1] if packets else 10
    prior=np.eye(J)[0]
    if not packets:return prior,dict(objective=0.,lower_bound=0.,gap=0.,relative_gap=0.,seconds=0.,packets=0,active_future=0)
    norm=np.max(S/s[:,None],axis=0);cols=np.flatnonzero(norm>0)
    cs=1/norm[cols];M=S[:,cols]*cs/s[:,None]
    cachekey=(S.shape,S.tobytes(),v.tobytes())
    if cachekey not in _PINV_CACHE:_PINV_CACHE[cachekey]=np.linalg.pinv(M)
    pinv=_PINV_CACHE[cachekey]
    theta=cp.Variable(J);constraints=[theta>=0,cp.sum(theta)==1]
    # Algebraically aggregate the ten-parameter current quadratic, avoiding
    # thousands of redundant cones without changing any packet's loss.
    G=alpha/2*np.eye(J);h=alpha*prior.copy();base=alpha/2*float(prior@prior)
    for packet in packets:
        PP=packet['P']/s[:,None];yy=packet['y']/s
        G+=PP.T@PP/n;h+=2/n*PP.T@yy;base+=float(yy@yy/n)
    objective=cp.quad_form(theta,cp.psd_wrap(G))-h@theta+base
    lifts=[];constant=0.;skipped=0
    for packet in packets:
        P=packet['P']/s[:,None];y=packet['y']/s
        e=P@theta-y
        if not future:continue
        beta=packet['beta']/s;D=packet['D']/s
        if packet.get('reset',False):
            constant+=float(np.mean((beta-D)**2));skipped+=1;continue
        if np.isfinite(packet.get('fixed_future_value',np.nan)):
            constant+=float(packet['fixed_future_value']);skipped+=1;continue
        C=(packet['A']+R*(packet['b']-packet['y'])**2)/v
        # V=0 for every theta if D is affordable at the smallest possible B.
        # This is an exact sufficient test; no score-dependent truncation.
        minB=C-np.max((P-y[:,None])**2,axis=1)
        dz=(D-(1-inset)*beta)/inset
        if np.all(D>=(1-inset)*beta) and np.all(gamma(beta,dz)<=minB):
            skipped+=1;continue
        cap=upper_bound(packet['beta'],C*v,inset)
        sub=S[:,cols]
        capw=np.min(np.divide(cap[:,None],sub,out=np.full_like(sub,np.inf),where=sub>0),axis=0)/cs
        if not np.any(capw>0):
            # All reachable future latent coordinates are zero. This includes
            # zero anchor/zero credit without creating a positive credit floor.
            constant+=float(np.mean(D*D));skipped+=1;continue
        w=cp.Variable(len(cols));q=M@cp.multiply(capw,w);z=(q-(1-inset)*beta)/inset
        constant_rows=np.all(packet['P']==packet['P'][:,0,None],axis=1)
        left=(C<=K*R*beta*beta)&(beta>0)
        lid=np.flatnonzero(~constant_rows&left);zid=np.flatnonzero(~constant_rows&(beta==0));mid=np.flatnonzero(~constant_rows&~left&(beta>0))
        ids=np.r_[lid,zid,mid];fixed=np.array([],dtype=int)
        cids=np.flatnonzero(constant_rows)
        cc=[w>=0,w<=1];constraints.extend(cc)
        low=q>=(1-inset)*beta;constraints.append(low)
        guard=None;eq=None
        qlow=(1-inset)*beta;nodecap=cap/s;clow=None;chigh=None
        if len(cids):
            by=(packet['b']-packet['y'])/s;delta=(packet['P'][:,0]-packet['b'])/s
            BC=packet['A']/v+K*by*by-2*by*delta-delta*delta
            assert np.all(BC[cids]>=-1e-12*(1+C[cids]))
            BC=np.maximum(BC,0)
            rad=np.sqrt(K/R*BC)
            qlow[cids]=beta[cids]+inset*(np.maximum(0,beta[cids]-rad[cids])-beta[cids])
            nodecap[cids]=upper_bound(beta,BC,inset)[cids]
            clow=q[cids]>=qlow[cids];chigh=q[cids]<=nodecap[cids]
            constraints.extend([clow,chigh])
        if len(ids):
            assert np.all(C[ids]>0)
            by=(packet['b']-packet['y'])/s;pd=(packet['P'][:,0]-packet['b'])/s
            B0=packet['A']/v+K*by*by-2*by*pd-pd*pd
            DP=P-P[:,0,None];e0=P[:,0]-y
            maxdelta=np.max(abs(DP),axis=1)
            scale2=B0+2*abs(e0)*maxdelta+maxdelta**2
            assert np.all(scale2[ids]>0)
            scales=np.sqrt(scale2[ids]);chunks=[]
            for rows,factor in [(lid,R/K),(zid,1.)]:
                if len(rows):chunks.append(factor*cp.square((q[rows]-beta[rows])/(inset*np.sqrt(scale2[rows]))))
            if len(mid):
                bb=beta[mid]/np.sqrt(scale2[mid]);dd=(q[mid]-beta[mid])/(inset*np.sqrt(scale2[mid]))
                chunks.append(cp.square(dd)+2*cp.multiply(bb,dd)-K*bb**2+cp.square(cp.pos(K*bb-dd))/K)
            gg=cp.hstack(chunks)
            de=(DP@theta)[ids]/scales
            g=gg+cp.square(de)+2*cp.multiply(e0[ids]/scales,de)-B0[ids]/scale2[ids]
            guard=g<=0;constraints.append(guard)
        else:B0=None;DP=None;e0=None;scale2=None
        qcap=q<=nodecap;constraints.append(qcap)
        if len(fixed):eq=q[fixed]==beta[fixed];constraints.append(eq)
        objective+=cp.sum_squares(q-D)/n
        lifts.append(dict(w=w,cap=capw,nodecap=nodecap,qlow=qlow,P=P,y=y,C=C,beta=beta,D=D,ids=ids,fixed=fixed,cids=cids,clow=clow,chigh=chigh,qcap=qcap,guard=guard,eq=eq,low=low,wlo=cc[0],whi=cc[1],B0=B0,DP=DP,e0=e0,scale2=scale2,beta_u=packet.get('beta_u',None)))
    problem=cp.Problem(cp.Minimize(objective+constant),constraints)
    assert problem.is_dcp()
    attempts=[]
    for solver,tol in [('CLARABEL',1e-7),('CLARABEL',1e-9),('CLARABEL',1e-11),('SCS',1e-9)]:
        try:
            if solver=='CLARABEL':problem.solve(solver=solver,max_iter=1000,max_threads=1,tol_gap_abs=tol,tol_gap_rel=tol,tol_feas=tol,warm_start=False)
            else:problem.solve(solver=solver,eps=tol,max_iters=200000,warm_start=False)
        except cp.error.SolverError as err:
            attempts.append(dict(solver=solver,tol=tol,error=str(err)));continue
        if theta.value is None:continue
        th0=np.asarray(theta.value).ravel();th=np.maximum(th0,0);th/=th.sum()
        if alpha>0 and all(np.array_equal(p['P'],np.tile(p['P'][:,0,None],(1,J))) for p in packets):th=prior.copy()
        obj=alpha/2*np.sum((th-prior)**2)+constant
        grad=alpha*(th-prior);L=obj;max_resid=0.;Gcur=G.copy()
        for packet in packets:
            P=packet['P']/s[:,None];e=P@th-packet['y']/s
            value=float(e@e/n);obj+=value;L+=value;grad+=2/n*P.T@e
        lower_correction=0.;credit_residual=0.;max_future_repair=0.
        for lift in lifts:
            w=lift['cap']*np.clip(np.asarray(lift['w'].value).ravel(),0,1)
            q=M@w;e=lift['P']@th-lift['y'];beta=lift['beta'];D=lift['D']
            # Repair the lifted primal future witness on its own fixed B(theta),
            # leaving theta unchanged. The lower bound uses the same multipliers
            # at this repaired point and pays the explicit equality residual.
            if lift['DP'] is not None:
                de=lift['DP']@th;BB=lift['B0']-2*lift['e0']*de-de*de
            else:BB=lift['C']-e*e
            assert np.min(BB)>=-1e-12*(1+np.max(lift['C']))
            BB=np.maximum(BB,0);rad=np.sqrt(K/R*BB)
            flo=beta+inset*(np.maximum(0,beta-rad)-beta);fhi=upper_bound(beta,BB,inset)
            dd=q-beta;rat=np.ones(n)
            pos=dd>0;neg=dd<0
            rat[pos]=(fhi[pos]-beta[pos])/dd[pos];rat[neg]=(flo[neg]-beta[neg])/dd[neg]
            contraction=max(0.,min(1.,float(rat.min())))
            if contraction<1:
                contraction=np.nextafter(contraction,0.)
                oldq=q.copy();q=beta+contraction*dd
                if lift['beta_u'] is not None:
                    # The lift stores the packet index below, not a different
                    # actor's account or future action.
                    anchor=lift['beta_u'][cols]/cs
                else:anchor=np.maximum(0,pinv@beta)
                w=anchor+contraction*(w-anchor)
                max_future_repair=max(max_future_repair,float(abs(q-oldq).max()))
            z=(q-(1-inset)*beta)/inset
            value=float(np.mean((q-D)**2));obj+=value;L+=value
            lam=np.zeros(n);eta=np.zeros(n)
            if lift['guard'] is not None:
                lam[lift['ids']]=np.maximum(0,np.asarray(lift['guard'].dual_value).ravel())/lift['scale2'][lift['ids']]
            if lift['eq'] is not None:eta[lift['fixed']]=np.asarray(lift['eq'].dual_value).ravel()
            nu=np.maximum(0,np.asarray(lift['low'].dual_value).ravel())
            gres=gamma(beta,z)+e*e-lift['C']
            if lift['DP'] is not None:
                de=lift['DP']@th
                gres=gamma(beta,z)+de*de+2*lift['e0']*de-lift['B0']
            L+=float(lam@gres+nu@((1-inset)*beta-q)+eta@(q-beta))
            if lift['DP'] is not None:
                grad+=2*lift['DP'].T@(lam*e)
                Gcur+=lift['DP'].T@(lam[:,None]*lift['DP'])
            delta=z-beta
            gz=np.where(delta<=K*beta,2*R/K*delta,2*z)
            gq=2/n*(q-D)+lam*gz/inset-nu+eta
            gqfull=gq.copy()
            gqfull+=np.asarray(lift['qcap'].dual_value).ravel()
            if lift['clow'] is not None:gqfull[lift['cids']]+=np.asarray(lift['chigh'].dual_value).ravel()-np.asarray(lift['clow'].dual_value).ravel()
            # Gamma has curvature >=2 globally. Its valid quadratic minorant
            # yields a tighter lower bound than a linear plane over a loose
            # non-identifiable latent box. Relax coherence only in the lower
            # bound calculation, never in the fitted or deployed prediction.
            # Restore a free equality multiplier for q=Mw. Splitting q,w in
            # the lower-bound minimization permits exploiting its normal price
            # without assuming a numerical pseudoinverse gives an exact null.
            eqprice=M@(pinv@gqfull)-gqfull
            boxred=np.asarray(lift['wlo'].dual_value).ravel()-np.asarray(lift['whi'].dual_value).ravel()
            desired=np.divide(boxred,lift['cap'],out=np.zeros_like(boxred),where=lift['cap']>0)
            choices=[eqprice,eqprice-pinv.T@desired,-gqfull]
            bounds=[]
            for price in choices:
                wwgrad=-M.T@price;linear=-nu+eta+price
                # Analytic minimum of the exact separable Lagrangian.
                dl=(2/n*(D-beta)-linear)/(2/n+2*lam*R/(K*inset**2))
                dr=(2/n*(D-beta)-linear-2*lam*beta/inset)/(2/n+2*lam/inset**2)
                qstar=beta+np.where(dl<=inset*K*beta,dl,dr)
                qstar=np.clip(qstar,lift['qlow'],lift['nodecap']);qstar[lift['fixed']]=beta[lift['fixed']]
                dq=qstar-q
                bounds.append(float(price@(q-M@w)+dq@(qstar+q-2*D)/n+lam@(gamma(beta,(qstar-(1-inset)*beta)/inset)-gamma(beta,z))+linear@dq+wwgrad@(np.where(wwgrad>=0,0,lift['cap'])-w)))
            exact_lower=max(bounds)
            original_wgrad=M.T@gq
            linear_lower=float(original_wgrad@(np.where(original_wgrad>=0,0,lift['cap'])-w))
            lower_correction+=exact_lower if np.any(q!=M@w) else max(exact_lower,linear_lower)
            max_resid=max(max_resid,float(np.max(gres[lift['ids']])) if len(lift['ids']) else 0.,float(np.max(lift['qlow']-q)),float(np.max(q-lift['nodecap'])),float(np.max(abs(q[lift['fixed']]-beta[lift['fixed']]))) if len(lift['fixed']) else 0.)
            credit_residual=max(credit_residual,float(np.max(np.maximum(gres,0)/(1+lift['C']))))
        if alpha>0:
            u=np.sort(th-grad/alpha)[::-1];cums=np.cumsum(u)-1
            rho=np.flatnonzero(u>cums/np.arange(1,J+1))[-1]
            ts=np.maximum(th-grad/alpha-cums[rho]/(rho+1),0)
            dt=ts-th;parameter_lower=float(grad@dt+alpha/2*(dt@dt))
        else:parameter_lower=float(grad.min()-grad@th)
        # The exact current quadratic and quadratic refund part also provide
        # a valid parameter minorant. Certify its small simplex QP by its own
        # first-order lower bound, rather than trusting SLSQP convergence.
        result=minimize(lambda a:float(grad@(a-th)+(a-th)@Gcur@(a-th)),th,
            jac=lambda a:grad+2*Gcur@(a-th),bounds=[(0,1)]*J,
            constraints=[dict(type='eq',fun=lambda a:a.sum()-1,jac=lambda a:np.ones(J))],
            method='SLSQP',options=dict(ftol=1e-14,maxiter=1000))
        a=np.maximum(result.x,0);a/=a.sum();dt=a-th;gg=grad+2*Gcur@dt
        qp_lower=float(grad@dt+dt@Gcur@dt+gg.min()-gg@a)
        parameter_lower=max(parameter_lower,qp_lower)
        lb=float(L+parameter_lower+lower_correction)
        gap=float(obj-lb)
        info=dict(objective=float(obj),lower_bound=lb,gap=gap,relative_gap=gap/(1+obj),max_future_primal_repair=max_future_repair,parameter_lower_correction=parameter_lower,future_lower_correction=lower_correction,lagrangian_residual=L-obj,max_normalized_constraint_residual=max_resid,relative_credit_residual=credit_residual,theta_rounding=float(abs(th-th0).max()),seconds=time.perf_counter()-begin,packets=len(packets),active_future=len(lifts),skipped_constant_future=skipped,status=problem.status,iterations=problem.solver_stats.num_iters,solver=solver,tol=tol)
        attempts.append(info.copy())
        if -1e-7*(1+obj)<=gap<=FIT_REL_GAP*(1+obj) and max_resid<=1e-6 and credit_residual<=1e-8:
            info['attempts']=attempts;return th,info
    raise RuntimeError({'parameter_fit_not_certified':attempts})
