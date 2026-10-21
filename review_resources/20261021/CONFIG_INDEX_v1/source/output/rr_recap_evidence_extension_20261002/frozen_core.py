from pathlib import Path
import json,hashlib,shutil,time
import numpy as np
import pandas as pd
from scipy.optimize import nnls,lsq_linear,linprog
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
SEEDS=[42,1,2,3,4]
EXPERTS=['raw','bias3','bias7','bias14','bias28','persistence','mean7','seasonal7','online_ridge']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(dataset,seed):
    dest=ROOT/'inputs'/dataset/str(seed);dest.mkdir(parents=True,exist_ok=True)
    sources={'data.csv':PARENT/f'procast_comparison_20261001/data/{dataset}.csv',
        'S.npy':PARENT/f'procast_comparison_20261001/data/{dataset}_S.npy',
        'test.csv':PARENT/f'procast_comparison_20261001/forecasts/{dataset}/{seed}/test.csv',
        'calibration.csv':PARENT/f'procast_independent_20261001/calibration/{dataset}/{seed}/predictions.csv'}
    hashes={}
    for name,src in sources.items():
        target=dest/name
        if src.exists():
            if not target.exists():shutil.copy2(src,target)
            assert sha(src)==sha(target)
        assert target.exists()
        hashes[name]={'source':str(src),'sha256':sha(target)}
    (dest/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    return dest

def read_inputs(dataset,seed,validation=False):
    folder=prepare(dataset,seed)
    wide=pd.read_csv(folder/'data.csv',parse_dates=['ds']).set_index('ds')
    start=len(wide)-108
    frames=[pd.read_csv(folder/'calibration.csv',parse_dates=['ds','origin'])]
    if not validation:frames.append(pd.read_csv(folder/'test.csv',parse_dates=['ds','origin']))
    else: wide=wide.iloc[:-72].copy() # Test labels excluded before any modeling.
    frame=pd.concat(frames,ignore_index=True)
    assert frame.origin.lt(frame.ds).all()
    raw=frame.pivot(index='ds',columns='unique_id',values='PatchTST').reindex(index=wide.index[start:],columns=wide.columns).to_numpy()
    assert raw.shape[0]==(36 if validation else 108) and np.isfinite(raw).all()
    variance=wide.iloc[:start].var(ddof=0).to_numpy()
    positive=variance[variance>1e-12]
    floor=max(1e-12,1e-6*np.median(positive)) if len(positive) else 1e-12
    return wide,raw,np.load(folder/'S.npy'),np.maximum(variance,floor),start

def project(S,targets,v=None):
    weight=np.ones(S.shape[0]) if v is None else 1/np.sqrt(v)
    design=S*weight[:,None];ws=[]
    for row in np.atleast_2d(targets):
        try:u,_=nnls(design,row*weight,maxiter=100*S.shape[1])
        except RuntimeError:
            result=lsq_linear(design,row*weight,bounds=(0,np.inf),tol=1e-12,max_iter=2000)
            assert result.success,result.message
            u=result.x
        assert u.min()>=0
        ws.append(u)
    ws=np.asarray(ws)
    return ws@S.T,ws

def expert_bank(wide,raw,v,start):
    y=wide.to_numpy(); m=raw.shape[1];features=[];bank=[]
    for t,f in enumerate(raw):
        i=start+t; weekday=wide.index[i].dayofweek
        feat=np.column_stack([f,y[i-1],y[i-7:i].mean(0),np.tile([(t%12+1)/12,
            np.sin(2*np.pi*weekday/7),np.cos(2*np.pi*weekday/7)],(m,1))])
        features.append(feat)
        estimates=[f]
        for window in [3,7,14,28]:
            left=max(0,t-window)
            bias=np.mean(raw[left:t]-y[start+left:i],axis=0) if t else np.zeros(m)
            estimates.append(f-bias)
        estimates.extend([y[i-1],y[i-7:i].mean(0),y[i-7]])
        ridge=f.copy()
        if t>=12:
            left=max(0,t-36); X=np.asarray(features[left:t]);error=raw[left:t]-y[start+left:i]
            for j in range(m):
                scaler=StandardScaler().fit(X[:,j]);model=Ridge(alpha=10.).fit(scaler.transform(X[:,j]),error[:,j]/np.sqrt(v[j]))
                ridge[j]-=model.predict(scaler.transform(feat[j][None,:]))[0]*np.sqrt(v[j])
        estimates.append(ridge)
        bank.append(np.stack(estimates,axis=1))
    return np.asarray(bank)

def candidate(bank,y,S,v,window,power):
    weights=[];targets=[]
    for t,now in enumerate(bank):
        if t:
            past=bank[max(0,t-window):t]-y[max(0,t-window):t,:,None]
            mse=np.mean(past*past,axis=0)
            smoothed=mse+.1*mse.mean(axis=1,keepdims=True)+np.maximum(1e-12,1e-8*v)[:,None]
            logw=-power*np.log(smoothed);logw-=logw.max(axis=1,keepdims=True)
            w=np.exp(logw);w/=w.sum(axis=1,keepdims=True)
        else:w=np.full_like(now,1/now.shape[1])
        weights.append(w);targets.append(np.sum(now*w,axis=1))
    z,uz=project(S,np.asarray(targets),v)
    return z,uz,np.asarray(weights)

def caps(e,d,epsilon=.005):
    A=(d*d).sum(0);B=(e*d).sum(0);E=(e*e).sum(0)
    scale=np.maximum.reduce([A,np.abs(B),E,np.full_like(A,np.finfo(float).tiny)])
    a=A/scale;bb=B/scale;c=((1+epsilon)**2-1)*E/scale
    roots=np.ones(len(A))
    for i in range(len(A)):
        if a[i]>0:
            disc=np.sqrt(bb[i]**2+a[i]*c[i])
            roots[i]=(c[i]/(disc+bb[i]) if disc+bb[i]>0 else 0.) if bb[i]>=0 else (-bb[i]+disc)/a[i]
        elif bb[i]>0:roots[i]=c[i]/(2*bb[i])
    return np.clip(roots,0,1),(a,bb,c)

def coherent_ray(S,b,direction,ub,v):
    if np.max(np.abs(direction))<1e-12:return b.copy(),ub.copy(),1.
    # Optimize latent witness and common radial attenuation; b is always feasible.
    latent_scale=max(1.,float(np.max(np.abs(b))))
    row_scale=np.maximum.reduce([np.abs(b),np.abs(direction),np.sqrt(v),np.ones(len(b))])
    design=np.column_stack([S*latent_scale,-direction])/row_scale[:,None]
    rhs=b/row_scale
    objective=np.zeros(S.shape[1]+1);objective[-1]=-1
    result=linprog(objective,A_eq=design,b_eq=rhs,bounds=[(0,None)]*S.shape[1]+[(0,1)],method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:
        result=linprog(objective,A_eq=design,b_eq=rhs,bounds=[(0,None)]*S.shape[1]+[(0,1)],method='highs-ds',
            options={'presolve':False,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:
        result=linprog(objective,A_eq=design,b_eq=rhs,bounds=[(0,None)]*S.shape[1]+[(0,1)],method='highs-ipm',
            options={'presolve':False,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:raise RuntimeError(result.message)
    u=result.x[:-1]*latent_scale;lam=float(np.clip(result.x[-1],0,1));out=b+lam*direction
    np.testing.assert_allclose(out,u@S.T,rtol=1e-8,atol=2e-6)
    assert u.min()>=-1e-9
    return out,u,lam

def release(b,z,y,S,ub,uz,v,mode='component_exact',initial=1.,eta=.3):
    n,m=y.shape;scalar=mode.startswith('scalar');dim=1 if scalar else m
    state=np.full(dim,initial);G=np.full(dim,1e-12);d=z-b
    outputs=[];proposals=[];ratios=[];witnesses=[];rays=[];limits=[];excesses=[]
    for t in range(n):
        if t:
            error=b[t-1]+proposals[-1]*d[t-1]-y[t-1]
            g=2*error*d[t-1]/v
            if scalar:g=np.array([g.mean()])
            G+=g*g;state=np.clip(state-eta*g/np.sqrt(G),0,1)
            sl=slice(max(0,t-72),t);lim,poly=caps(b[sl]-y[sl],d[sl])
        else:lim=np.ones(m);poly=None
        proposals.append(np.full(m,state[0]) if scalar else state.copy())
        if mode=='scalar_heuristic':
            if t:
                e=b[sl]-y[sl]
                risk=np.max(np.sqrt(np.sum((z[sl]-y[sl])**2,axis=0)/np.maximum(np.sum(e*e,axis=0),1e-12))-1)
                cap=np.clip(1-max(risk,0)/.005,0,1)
            else:cap=1.
            q=np.full(m,min(state[0],cap))
        elif mode=='scalar_exact':q=np.full(m,min(state[0],lim.min()))
        elif mode=='component_uncapped':q=state.copy()
        else:q=np.minimum(state,lim)
        if scalar:
            lam=1.;out=b[t]+q*d[t];u=ub[t]+q[0]*(uz[t]-ub[t])
        else:out,u,lam=coherent_ray(S,b[t],q*d[t],ub[t],v)
        final=lam*q
        if t and mode in ['component_exact','scalar_exact']:
            a,bb,c=poly;excess=float(np.max(a*final*final+2*bb*final-c));excesses.append(excess)
            assert excess<=2e-12
        outputs.append(out);witnesses.append(u);ratios.append(final);rays.append(lam);limits.append(lim)
    return dict(pred=np.asarray(outputs),q=np.asarray(ratios),proposal=np.asarray(proposals),
        witness=np.asarray(witnesses),ray=np.asarray(rays),limits=np.asarray(limits),
        max_historical_excess=max(excesses,default=0.))
