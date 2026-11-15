import csv, sys, time, hashlib, json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from publishers import ar, S, SafeProjection, publish

HERE=Path(__file__).resolve().parent
DATA=HERE/'inputs'
EXPECTED=HERE/'expected'
NODES=['SYSTEM','NSW1','QLD1','SA1','TAS1','VIC1']
ARMS=['Full','Independent']
LEVELS=['overall','regions']+NODES
ZERO=[Fraction(0)]*6

class State:
    def __init__(self):
        self.settled=ZERO.copy()
        self.pending=[]
        self.prefix=ZERO.copy()
        self.violations=np.zeros(6,int)
        self.bad=0
    def available(self,issue):
        wait=[]
        for ready,ref,p,charge,y in self.pending:
            if ready<=issue:
                self.settled=ar.balance(self.settled,ref,p,y)
            else:
                wait.append((ready,ref,p,charge,y))
        self.pending=wait
        free=self.settled.copy()
        for _,_,_,charge,_ in wait:
            free=[v-c for v,c in zip(free,charge)]
        if min(free)<0:raise ValueError('negative available credit')
        return free
    def add(self,ready,ref,p,free,y):
        charge=ar.excess(ref,p)
        self.bad+=sum(c>v for c,v in zip(charge,free))
        self.pending.append((ready,ref,p,charge,y))
        self.prefix=ar.balance(self.prefix,ref,p,y)
        self.violations+=[v<0 for v in self.prefix]

def scores(p,y):
    n=np.sqrt(np.mean((p-y)**2,axis=0))
    return dict(zip(LEVELS,[n.mean(),n[1:].mean(),*n]))
def one(path,idx,build,clock,labels,legacy,times,rows):
    bb,run=path.stem.split('_');run=int(run)
    with np.load(path) as a:
        y,b,ub,z,u,v,ready,issue=[a[k] for k in ('Y','B','UB','Z','U','v','ready_ns','issue_ns')]
        archived={arm:a[arm] for arm in ['Reference','Full','Direct','AdaptiveTwo','Common']}
    generated=clock.last_component_issued.to_numpy('datetime64[ns]').astype('int64')
    dates=clock.ds.to_numpy('datetime64[ns]').astype('int64')
    last=[np.flatnonzero(generated<=i)[-1] for i in issue]
    np.testing.assert_array_equal(issue,dates[24:]-1800*10**9)
    np.testing.assert_array_equal(ready,generated[24:])
    np.testing.assert_array_equal(last,np.arange(24,120)-3)
    np.testing.assert_allclose(ub,labels[NODES[1:]].to_numpy()[last],atol=1e-10,rtol=1e-14)
    np.testing.assert_allclose(y,labels[NODES].to_numpy()[24:],atol=1e-10,rtol=1e-14)
    np.testing.assert_array_equal(ub@S.T,b)
    np.testing.assert_array_equal(archived['Reference'],b)
    np.testing.assert_array_equal(archived['Direct'],z)
    solver=SafeProjection(S,v)
    # Non-scored synthetic free credit triggers both publishers before the timed loop.
    for arm in ARMS:publish(arm,solver,v,b[0],ub[0],z[0],u[0],[Fraction(100)]*6)
    states={arm:State() for arm in ARMS}
    values={arm+suffix:[] for arm in ARMS for suffix in ('','_w','_free','_decision','_gap')}
    for j in range(96):
        bq,yq=ar.exact(b[j]),ar.exact(y[j])
        order=ARMS if (idx+j)%2==0 else ARMS[::-1]
        for position,arm in enumerate(order):
            state=states[arm]
            free=state.available(issue[j])
            tic=time.perf_counter_ns()
            p,w,pq,decision,gap=publish(arm,solver,v,b[j],ub[j],z[j],u[j],free)
            ms=(time.perf_counter_ns()-tic)/1e6
            state.add(ready[j],bq,pq,free,yq)
            times.append(dict(bb=bb,run=run,t=j+1,rule=arm,order=position+1,ms=ms))
            for suffix,val in [('',p),('_w',w),('_free',list(map(str,free))),('_decision',decision),('_gap',gap)]:
                values[arm+suffix].append(val)
    values={k:np.array(x) for k,x in values.items()}
    np.testing.assert_allclose(values['Full'],archived['Full'],atol=2e-6,rtol=1e-9)
    identical=np.array_equal(values['Full'],archived['Full'])
    fout=EXPECTED/path.name
    if build:
        np.savez_compressed(fout,**values)
    else:
        with np.load(fout) as frozen:
            for arm in ARMS:
                np.testing.assert_allclose(values[arm],frozen[arm],atol=2e-6,rtol=1e-9)
    old=legacy[(legacy.bb==bb)&(legacy.run==run)].set_index('rule')
    for arm,p in {**archived,**{k:values[k] for k in ARMS}}.items():
        result=scores(p,y)
        if arm in old.index:
            np.testing.assert_allclose(list(result.values()),old.loc[arm,LEVELS].to_numpy(dtype=float),atol=2e-6,rtol=1e-9)
        if arm in ARMS:
            state=states[arm];env=state.bad;prefix=state.violations.sum();ever=np.count_nonzero(state.violations)
            assert env==prefix==ever==0
        else:
            env,prefix,ever=map(int,old.loc[arm,['envelope','prefix','ever']])
        rows.append(dict(bb=bb,run=run,rule=arm,**result,checks=576,envelope=env,prefix=prefix,ever=ever))
    return identical

def tables(scores,times):
    flat=scores.set_index(['bb','run','rule'])
    pairs=[]
    for (bb,run),part in scores.groupby(['bb','run']):
        rp=part.set_index('rule')
        for comp in ['Independent','Direct','Common','AdaptiveTwo','Reference']:
            g=100*(1-rp.loc['Full',LEVELS]/rp.loc[comp,LEVELS])
            pairs.append(dict(bb=bb,run=run,comparator=comp,**g.to_dict()))
    return pd.DataFrame(pairs)

def main(build):
    hashes_path=HERE/'INPUT_HASHES.json'
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.iterdir()) if p.is_file()}
    if build:hashes_path.write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    else:assert hashes==json.loads(hashes_path.read_text(encoding='utf-8'))
    clock=pd.read_csv(DATA/'clock.csv',parse_dates=['ds','last_component_issued'])
    labels=pd.read_csv(DATA/'observed.csv',parse_dates=['ds']); assert labels.ds.equals(clock.ds)
    assert len(clock)==120 and np.diff(clock.ds.to_numpy()).min()==np.diff(clock.ds.to_numpy()).max()==np.timedelta64(30,'m')
    old=pd.read_csv(DATA/'stream_scores.csv')
    times=[];rows=[];identical=[]
    for idx,p in enumerate(sorted(DATA.glob('*.npz'))):
        identical.append(one(p,idx,build,clock,labels,old,times,rows))
    scores_df=pd.DataFrame(rows);time_df=pd.DataFrame(times)
    pairs=tables(scores_df,time_df)
    assert len(scores_df)==54 and len(time_df)==1728 and len(identical)==9
    grouped=scores_df.groupby('rule')[LEVELS].mean()
    pernode=pairs.groupby('comparator')[LEVELS].mean()
    timings=time_df.groupby('rule').ms.agg(['mean','median',lambda x:x.quantile(.95)]);timings.columns=['mean','median','p95']
    if build:
        for name,df in [('scores',scores_df),('timings',time_df),('comparisons',pairs)]:
            df.to_csv(HERE/(name+'.csv'),index=False)
        grouped.to_csv(HERE/'mean_rmse_MW.csv');pernode.to_csv(HERE/'mean_paired_percent.csv');timings.to_csv(HERE/'timing_summary_ms.csv')
    else:
        ref=pd.read_csv(HERE/'scores.csv')
        np.testing.assert_allclose(scores_df[LEVELS].to_numpy(),ref[LEVELS].to_numpy(),rtol=1e-9,atol=2e-6)
        np.testing.assert_array_equal(scores_df[['checks','envelope','prefix','ever']],ref[['checks','envelope','prefix','ever']])
    print(grouped.to_string(),pernode.to_string(),timings.to_string(),sep='\n')
    print('PASS: independently replayed Full and Independent, 5184 exact envelope and prefix checks per arm; archived Full bitwise equal on',sum(identical),'of 9 streams.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        main('--build' in sys.argv[1:])

