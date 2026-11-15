"""Independently score saved publications and check the scalar supremum and every prefix."""
from pathlib import Path
import argparse,csv,hashlib,gzip,os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from run_groups import HERE,DEFAULT_ARCHIVE,DEFAULT_INDEX,RULES
import pandas as pd

def ints(a,e):
    x,q=np.frexp(a)
    mant=(x*2**53).astype(np.int64)
    # e was chosen so the shift is nonnegative for all nonzero entries.
    shifts=e+q-53
    assert ((shifts>=0)|(a==0)).all()
    return np.left_shift(mant.astype(object),np.maximum(shifts,0)).astype(object)
def audit(B,Y,P):
    smallest=min((np.abs(a[a!=0]).min() for a in (B,Y,P) if np.any(a)),default=1.)
    e=max(0,54-int(np.frexp(smallest)[1]))
    b,y,p=(ints(a,e) for a in (B,Y,P))
    # Let A denote 40000 times available allowance in squared integer units.
    # 401*40000*Gamma = 401*(40000*p²-40401*b²) + positive_part(40401*b-40000*p)².
    delta=40401*(b-y)**2-40000*(p-y)**2
    post=delta.cumsum(axis=0)
    prior=np.vstack([np.zeros(p.shape[1],dtype=object),post[:-1]])
    charge=401*(40000*p*p-40401*b*b)+np.maximum(0,40401*b-40000*p)**2
    env=(charge>401*prior);pre=(post<0)
    return env.sum(0),pre.sum(0)
def one(job):
    ix,archive,folder=job
    dataset=ix['dataset']; case=ix['case']
    c=archive/ix['context_archive']; md=(archive/ix['PIRFull_archive']).parent
    stream=[];timings=[];rms={};envs={};prefix={}
    with np.load(c) as f:B,Y=f['b'][36:],f['truth'][36:]
    if dataset=='AEMO':names=['System','NSW1','QLD1','SA1','TAS1','VIC1'];scopes=['all;system']+['all;region']*5
    else:
        node_meta=pd.read_csv(archive/'node_map.csv').query('dataset==@dataset').sort_values('node')
        names=node_meta.name;scopes=node_meta.scopes
    for arm in ('Reference','Direct','Full','Common','Fixed2',*RULES):
        stats=dict(zeros=0,targets=0,retries=0,guarded=0,anchors_on_failure=0,max_gap=0.,max_guard=0.)
        if arm=='Reference': P=B
        elif arm in RULES:
            with np.load(folder/'runs'/case/(arm+'.npz')) as f:
                P=f['pred'];dec=f['decision'].astype(str)
                stats.update(zeros=sum(dec=='zero'),targets=sum(dec=='target'),retries=np.char.startswith(dec,'retry_').sum(),guarded=np.char.endswith(dec,'guard').sum(),max_gap=f['gap'].max(),max_guard=f['move'].max())
                timings.extend(zip([case]*len(P),[arm]*len(P),np.arange(len(P)),f['ms']))
        else:
            filename=dict(Direct='PIRCoherent',Full='PIRFull',Common='PIRCommon',Fixed2='PIRTwoGroup')[arm]
            P=np.load(md/(filename+'.npz'))['pred']
        e,p=audit(B,Y,P)
        assert np.all(P>=0)
        if arm!='Direct':assert not np.any(e) and not np.any(p)
        rms[arm]=np.sqrt(np.mean((P-Y)**2,axis=0))
        envs[arm]=e;prefix[arm]=p
        if arm not in RULES: stats={k:float('nan') for k in stats}
        stream.append(dict(case=case,rule=arm,node_dates=P.size,envelope_events=e.sum(),prefix_events=p.sum(),node_sequences_ever=(p>0).sum(),**stats))
    rows=[]
    for i,(name,scope) in enumerate(zip(names,scopes)):
        rows.append(dict(case=case,dataset=dataset,backbone=ix['backbone'],run=int(ix['run']),part=ix['part'],protocol=ix['protocol'],node=i,name=name,scopes=scope,**{arm:v[i] for arm,v in rms.items()}))
    return rows,stream,timings

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE);p.add_argument('--index',type=Path,default=DEFAULT_INDEX);p.add_argument('--out',type=Path,default=HERE);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    cases=list(csv.DictReader(a.index.open(encoding='utf-8-sig')))
    with ProcessPoolExecutor(a.workers) as pool:
        results=list(pool.map(one,[(r,a.archive,a.out) for r in cases]))
    records=a.out/'records';records.mkdir(exist_ok=True)
    for j,n in enumerate(('node_rmse.csv.gz','arithmetic_audit.csv','publication_ms.csv.gz')):
        data=pd.DataFrame([r for item in results for r in item[j]],columns=(None if j<2 else ['case','rule','issue','ms']))
        data.to_csv(records/n,index=False)
    print('PASS: independent dyadic expression, prefixes, node-level scores and issued streams',len(cases))
if __name__=='__main__':main()

