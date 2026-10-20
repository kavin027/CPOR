import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from frozen_core import project

def build(wide,raw,S,start,delay=1):
    # start is calibration-start index; raw contains 36 calibration + 72 test dates.
    yall=wide.to_numpy();y=yall[start:start+len(raw)];v=wide.iloc[:start].var(ddof=0).to_numpy()
    positive=v[v>1e-12];floor=max(1e-12,1e-6*np.median(positive)) if len(positive) else 1e-12;v=np.maximum(v,floor)
    features=[];bank=[];simple=[];mixed=[]
    for t,f in enumerate(raw):
        i=start+t;end=i-delay+1;count=max(0,t-delay+1);week=wide.index[i].dayofweek
        lag=yall[end-1];mean=yall[end-7:end].mean(0);season=yall[i-7*int(np.ceil(delay/7))]
        feat=np.column_stack([f,lag,mean,np.tile([(t%12+1)/12,np.sin(2*np.pi*week/7),np.cos(2*np.pi*week/7)],(len(f),1))]);features.append(feat)
        values=[f]
        for window in [3,7,14,28]:
            bias=(raw[max(0,count-window):count]-y[max(0,count-window):count]).mean(0) if count else np.zeros(len(f))
            values.append(f-bias)
        values.extend([lag,mean,season]);ridge=f.copy()
        if count>=12:
            left=max(0,count-36);X=np.asarray(features[left:count]);error=raw[left:count]-y[left:count]
            for j in range(len(f)):
                scaler=StandardScaler().fit(X[:,j]);model=Ridge(alpha=10.).fit(scaler.transform(X[:,j]),error[:,j]/np.sqrt(v[j]));ridge[j]-=model.predict(scaler.transform(feat[j][None]))[0]*np.sqrt(v[j])
        values.append(ridge);now=np.stack(values,axis=1);bank.append(now);simple.append((lag+mean+season)/3)
        if count:
            past=np.asarray(bank[max(0,count-28):count])-y[max(0,count-28):count,:,None];mse=np.mean(past*past,axis=0)
            smooth=mse+.1*mse.mean(axis=1,keepdims=True)+np.maximum(1e-12,1e-8*v)[:,None]
            logw=-2*np.log(smooth);logw-=logw.max(1,keepdims=True);w=np.exp(logw);w/=w.sum(1,keepdims=True)
        else:w=np.full_like(now,1/now.shape[1])
        mixed.append((now*w).sum(1))
    b,ub=project(S,raw);z,uz=project(S,np.asarray(mixed),v);simp,us=project(S,np.asarray(simple),v)
    return dict(raw=raw,truth=y,S=S,v=v,b=b,ub=ub,z=z,uz=uz,simple=simp,usimple=us)
