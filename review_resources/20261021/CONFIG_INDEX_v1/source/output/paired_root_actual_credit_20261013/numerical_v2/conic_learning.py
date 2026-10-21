"""Exact root-increment epigraph; original objective and account unchanged."""
import gc,time
import numpy as np
import cvxpy as cp
import clarabel
from scipy import sparse

R=40401/40000; K=401/40000; ALPHA=.1; GAP=2e-5
TOLS=(1e-8,1e-10,1e-12); MAX_ITER=400; TIME_LIMIT=120.

def gamma(b,p):
    d=p-b
    return np.where(d<=K*b,R/K*d*d,d*d+2*b*d-K*b*b)

def increment(E,e,f=0.):
    sq=e*e+np.asarray(f)**2
    den=np.sqrt(E+sq)+np.sqrt(E)
    return np.divide(sq,den,out=np.zeros_like(sq),where=den>0)

def simplex(z):
    u=np.sort(z)[::-1]; ss=np.cumsum(u)-1
    ids=np.flatnonzero(u>ss/np.arange(1,len(u)+1))
    tau=ss[ids[-1]]/(ids[-1]+1)
    out=np.maximum(z-tau,0);return out/out.sum()

def weights(records,mode):
    N=len(records); single=np.ones(N) if mode=='C0' else np.zeros(N);pairs=[]
    if mode!='C0':
        start=0
        while start<N:
            end=start+1
            while end<N and records[end]['epoch']==records[start]['epoch']:end+=1
            if end-start==1:single[start]=1.
            else:
                single[start]=single[end-1]=.5
                pairs.extend((j,j+1,.5) for j in range(start,end-1))
            start=end
    return single,pairs

def project_dual(z,dims):
    z=np.array(z,copy=True);off=dims.zero
    z[off:off+dims.nonneg]=np.maximum(z[off:off+dims.nonneg],0);off+=dims.nonneg
    for dim in dims.soc:
        t=z[off];v=z[off+1:off+dim];a=np.linalg.norm(v)
        if a<=t:pass
        elif a<=-t:z[off:off+dim]=0
        else:
            tnew=(a+t)/2;z[off]=tnew;z[off+1:off+dim]=tnew/a*v
        if dim>1:z[off]=max(z[off],np.linalg.norm(z[off+1:off+dim]))
        off+=dim
    assert off==len(z)
    return z

def original_bounds(b,B):
    rad=np.sqrt(K/R*np.maximum(B,0));lo=np.maximum(0,b-rad)
    hi=np.where(B<=K*R*b*b,b+rad,np.sqrt(R*b*b+np.maximum(B,0)))
    return lo,hi

def _repair_q(q,u,b,ub,B,S):
    magnitude=1+np.max(abs(B))
    assert np.min(B)>=-1e-11*magnitude,('negative_counterfactual_cash',float(B.min()))
    B=np.maximum(B,0);zero=B==0
    if np.any(zero):
        assert np.max(abs(q[zero]-b[zero]))<=2e-6*(1+np.max(abs(b)))
        q=q.copy();q[zero]=b[zero]
    lo,hi=original_bounds(b,B);d=q-b;ratio=np.ones(len(b))
    pos=d>0;neg=d<0
    ratio[pos]=(hi[pos]-b[pos])/d[pos];ratio[neg]=(lo[neg]-b[neg])/d[neg]
    rho=max(0.,min(1.,float(ratio.min())))
    if rho<1:rho=np.nextafter(rho,0.)
    q=b+rho*d;u=ub+rho*(u-ub)
    q[zero]=b[zero]
    if np.max(gamma(b,q)-B)>1e-10*magnitude:
        raise AssertionError(('future_primal_reserve',float(np.max(gamma(b,q)-B))))
    np.testing.assert_allclose(q,S@u,rtol=1e-8,atol=2e-6)
    return q,u,rho

def build(records,S,mode,alpha=ALPHA,fixed_theta=None,pair_value=False):
    assert mode in ['C0','C1','C2'] and records
    n=S.shape[0];J=records[0]['P'].shape[1];prior=np.eye(J)[0]
    single,pairs=weights(records,mode);constraints=[];bounds={};variables={}
    if pair_value:
        assert len(records)==2 and records[0]['epoch']==records[1]['epoch'] and mode!='C0' and fixed_theta is not None
        single=np.zeros(2);pairs=[(0,1,1.)]
    def variable(shape,lo,hi,name):
        x=cp.Variable(shape,name=name);l=np.broadcast_to(lo,shape).copy();h=np.broadcast_to(hi,shape).copy()
        constraints.extend([x>=l,x<=h]);bounds[x.id]=(l.ravel(order='F'),h.ravel(order='F'));variables[x.id]=x
        return x
    theta=variable((J,),0. if fixed_theta is None else fixed_theta,1. if fixed_theta is None else fixed_theta,'theta');constraints.append(cp.sum(theta)==1)
    value=alpha/2*cp.sum_squares(theta)-alpha*theta[0];constant=alpha/2
    active=np.flatnonzero(single)
    if len(active):
        D=np.concatenate([records[j]['P']-records[j]['b'][:,None] for j in active])
        by=np.concatenate([records[j]['b']-records[j]['y'] for j in active])
        E=np.concatenate([records[j]['E'] for j in active]);e=D@theta+by
        mx=np.max(abs(by[:,None]+D),axis=1);hc=np.sqrt(E+mx*mx)
        w=np.repeat(single[active]/n,n);changing=np.flatnonzero(np.any(D!=0,axis=1));fixed=np.flatnonzero(np.all(D==0,axis=1))
        if len(fixed):constant+=float(w[fixed]@increment(E[fixed],by[fixed]))
        if len(changing):
            scale=increment(E[changing],mx[changing]);vs=scale+2*np.sqrt(E[changing])
            radius=np.sqrt(scale*vs)
            assert np.min(radius)>0
            h=variable((len(changing),),0.,2.,'single_increment')
            other=cp.multiply(scale/vs,h)+2*np.sqrt(E[changing])/vs
            constraints.append(cp.SOC(h+other,cp.vstack([2*cp.multiply(1/radius,e[changing]),h-other]),axis=0))
            value+=(w[changing]*scale)@h
    future=None
    if pairs:
        starts=[j for j,k,w in pairs];ends=[k for j,k,w in pairs];m=len(pairs)
        D=np.concatenate([records[j]['P']-records[j]['b'][:,None] for j in starts])
        by=np.concatenate([records[j]['b']-records[j]['y'] for j in starts])
        if pair_value and 'publication_override' in records[0]:
            D=np.zeros_like(D);by=records[0]['publication_override']-records[0]['y']
        E=np.concatenate([records[j]['E'] for j in starts]);e=D@theta+by
        bn=np.column_stack([records[k]['b'] for k in ends]);un=np.column_stack([records[k]['ub'] for k in ends])
        yn=np.concatenate([records[k]['y'] for k in ends])
        coupled=mode=='C1'
        if pair_value and 'fixed_future_B' in records[0]:
            cash=np.column_stack([records[j]['fixed_future_B'] for j in starts]);coupled=False
        elif mode=='C1':cash=np.column_stack([records[j]['A']+R*(records[j]['b']-records[j]['y'])**2 for j in starts])
        else:cash=np.column_stack([records[k]['A'] for k in ends])
        assert np.min(cash)>=0
        upper=bn+np.sqrt(cash)
        caps=np.empty((S.shape[1],m))
        for col in range(S.shape[1]):
            rows=S[:,col]>0
            caps[col]=np.min(upper[rows]/S[rows,col,None],axis=0) if np.any(rows) else 0.
        free=caps>0;safe=np.where(free,caps,1.)
        alo=-un/safe;ahi=np.where(free,1-un/safe,0.)
        # Zero columns are fixed; nonnegative latent bounds are compact and do
        # not change any feasible observed q under its node-wise upper bound.
        assert np.min(ahi-alo)>=-1e-8
        t=variable(un.shape,alo,np.maximum(ahi,alo),'future_latent_delta')
        delta=cp.reshape(S@cp.multiply(caps,t)+(S@un-bn),(n*m,),order='F')
        q=delta+bn.ravel(order='F');C=cash.ravel(order='F');beta=bn.ravel(order='F')
        zero=np.flatnonzero(C==0);positive=C>0
        if len(zero):
            constraints.append(delta[zero]==0)
            if coupled:
                assert np.max(abs(by[zero]))==0 and np.max(abs(D[zero]))==0
        left=np.flatnonzero(positive&(C<=K*R*beta*beta)&(beta>0))
        bz=np.flatnonzero(positive&(beta==0))
        general=np.flatnonzero(positive&~((C<=K*R*beta*beta)&(beta>0))&(beta>0))
        for ids,factor in [(left,R/K),(bz,1.)]:
            if len(ids):
                components=[np.sqrt(factor)*delta[ids]/np.sqrt(C[ids])]
                if coupled:components.append(e[ids]/np.sqrt(C[ids]))
                constraints.append(cp.SOC(np.ones(len(ids)),cp.vstack(components),axis=0))
        if len(general):
            z=variable((len(general),),0.,1.,'gamma_positive_part')
            ds=delta[general]/np.sqrt(C[general]);r=1-2*cp.multiply(beta[general]/np.sqrt(C[general]),z)
            components=[2*ds,2*(ds-z)/np.sqrt(K)]
            if coupled:components.append(2*e[general]/np.sqrt(C[general]))
            components.append(r-1)
            constraints.append(cp.SOC(r+1,cp.vstack(components),axis=0))
        mx=np.max(abs(by[:,None]+D),axis=1);fm=abs(beta-yn)+np.sqrt(C)
        scale=increment(E,mx,fm);zero_norm=scale==0
        hp=variable((len(E),),0.,np.where(zero_norm,0.,2.),'pair_increment')
        nz=np.flatnonzero(~zero_norm);vs=scale[nz]+2*np.sqrt(E[nz]);radius=np.sqrt(scale[nz]*vs)
        if len(nz):
            other=cp.multiply(scale[nz]/vs,hp[nz])+2*np.sqrt(E[nz])/vs
            constraints.append(cp.SOC(hp[nz]+other,cp.vstack([2*cp.multiply(1/radius,e[nz]),2*cp.multiply(1/radius,q[nz]-yn[nz]),hp[nz]-other]),axis=0))
        w=np.repeat(np.array([w/n for j,k,w in pairs]),n);value+=(w*scale)@hp
        future=dict(starts=starts,ends=ends,t=t,caps=caps,un=un,bn=bn,cash=cash,by=by,D=D,E=E,yn=yn)
    # Scale without modifying alpha or the original risk. Anchor provides an
    # exactly feasible reference even at zero cash.
    anchor=np.eye(J)[-1];anchor_value=alpha/2*np.sum((anchor-prior)**2)
    for j in active:
        r=records[j];anchor_value+=single[j]*float(increment(r['E'],r['b']-r['y']).mean())
    for j,k,w in pairs:
        r=records[j];s=records[k];anchor_value+=w*float(increment(r['E'],r['b']-r['y'],s['b']-s['y']).mean())
    divisor=max(1.,anchor_value)
    problem=cp.Problem(cp.Minimize(value/divisor),constraints);assert problem.is_dcp()
    data,_,_=problem.get_problem_data(cp.CLARABEL)
    dims=data['dims'];assert not dims.exp and not dims.psd and not dims.p3d
    lo=np.empty(len(data['c']));hi=np.empty_like(lo);covered=np.zeros(len(lo),bool)
    for vid,col in data['param_prob'].var_id_to_col.items():
        l,h=bounds[vid];lo[col:col+len(l)]=l;hi[col:col+len(h)]=h;covered[col:col+len(l)]=True
    assert covered.all()
    tc=data['param_prob'].var_id_to_col[theta.id]
    return dict(problem=problem,data=data,bounds=(lo,hi),theta=theta,theta_col=tc,variables=variables,constant=constant,divisor=divisor,future=future,single=single,pairs=pairs,prior=prior,alpha=alpha,fixed_theta=fixed_theta,pair_value=pair_value)

def dual_lower(model,z):
    d=model['data'];z=project_dual(z,d['dims']);g=d['c']+d['A'].T@z
    lo,hi=model['bounds'];mins=np.minimum(g*lo,g*hi)
    col=model['theta_col'];J=len(model['prior']);ids=np.arange(col,col+J)
    a=model['alpha']/model['divisor'];th=model['fixed_theta'] if model['fixed_theta'] is not None else simplex(-g[ids]/a)
    mins[ids]=0
    LB=float(mins.sum()+a/2*(th@th)+g[ids]@th-d['b']@z)
    arithmetic=64*np.finfo(float).eps*(1+abs(d['b'])@abs(z)+np.sum(abs(mins)))
    return max(0.,(LB-arithmetic)*model['divisor']+model['constant']),float(np.max(abs(g)))

def fit(records,S,mode,alpha=ALPHA,save_future=False,fixed_theta=None,pair_value=False):
    started=time.perf_counter();model=build(records,S,mode,alpha,fixed_theta,pair_value);d=model['data'];dims=d['dims']
    cones=[]
    if dims.zero:cones.append(clarabel.ZeroConeT(dims.zero))
    if dims.nonneg:cones.append(clarabel.NonnegativeConeT(dims.nonneg))
    cones.extend(clarabel.SecondOrderConeT(n) for n in dims.soc)
    attempts=[];best=None
    for tol in TOLS:
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=MAX_ITER;settings.time_limit=TIME_LIMIT
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=tol
        settings.iterative_refinement_max_iter=20
        solver=clarabel.DefaultSolver(d.get('P',sparse.csc_matrix((len(d['c']),len(d['c'])))),d['c'],d['A'],d['b'],cones,settings)
        sol=solver.solve()
        try:
            x=np.asarray(sol.x);tc=model['theta_col'];J=len(model['prior']);th=np.maximum(x[tc:tc+J],0);th/=th.sum()
            if fixed_theta is not None:th=np.array(fixed_theta,copy=True)
            qvalues=[];uvalues=[];rho=[]
            future=model['future']
            if future is not None:
                var=future['t'];col=d['param_prob'].var_id_to_col[var.id]
                t=x[col:col+var.size].reshape(var.shape,order='F')
                u=np.clip(future['un']+future['caps']*t,0,future['caps'])
                q=S@u
                for k,(j,nxt,w) in enumerate(model['pairs']):
                    r=records[j];s=records[nxt];delta=(r['P']-r['b'][:,None])@th;by=r['b']-r['y']
                    B=r['fixed_future_B'] if pair_value and 'fixed_future_B' in r else (r['A']+K*by*by-2*by*delta-delta*delta if mode=='C1' else s['A'])
                    pp,uu,rr=_repair_q(q[:,k],u[:,k],s['b'],s['ub'],B,S)
                    qvalues.append(pp);uvalues.append(uu);rho.append(rr)
            objective=alpha/2*np.sum((th-model['prior'])**2)
            for j,w in enumerate(model['single']):
                if w:
                    r=records[j];e=r['b']+(r['P']-r['b'][:,None])@th-r['y'];objective+=w*float(increment(r['E'],e).mean())
            for k,(j,nxt,w) in enumerate(model['pairs']):
                r=records[j];s=records[nxt];p=r.get('publication_override',r['b']+(r['P']-r['b'][:,None])@th);e=p-r['y'];objective+=w*float(increment(r['E'],e,qvalues[k]-s['y']).mean())
            lb,residual=dual_lower(model,sol.z);gap=max(0.,float(objective-lb));relative=gap/(1+abs(objective))
            info=dict(status=str(sol.status),objective=float(objective),lower_bound=lb,gap=gap,relative_gap=relative,normalized_objective=float(objective/len(records)),normalized_gap=gap/len(records),records=len(records),pairs=len(model['pairs']),soc_variables=len(d['c']),soc_constraints=d['A'].shape[0],iterations=sol.iterations,tol=tol,min_future_contraction=min(rho,default=1.),lagrangian_linear_max=residual,seconds=time.perf_counter()-started)
            attempts.append(info)
            if best is None or relative<best[1]['relative_gap']:best=(th,info,qvalues,uvalues)
            if relative<=GAP:
                info={**info,'attempts':attempts,'passed':True}
                if save_future:return th,info,dict(q=np.asarray(qvalues),u=np.asarray(uvalues),pairs=model['pairs'])
                del solver,model;gc.collect();return th,info,None
        except Exception as err:attempts.append(dict(status=str(sol.status),tol=tol,error=repr(err)))
    error=RuntimeError(dict(mode=mode,records=len(records),attempts=attempts,best=None if best is None else best[1]))
    del model;gc.collect();raise error
