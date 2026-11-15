from fractions import Fraction as F
from pathlib import Path
import sys,time,hashlib,json,platform
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
W=Path(__file__).resolve().parent;O=W.parents[1]
sys.path.insert(0,str(O/'cpor_aemo_november_holdout_20261112'))
from score import legacy
sys.path.insert(0,str(O/'cpor_mlj_v15_20261115/kit'))
sys.modules.pop('release',None)
import aemo_release as a
sys.path.pop(0)
REG=['NSW1','QLD1','SA1','TAS1','VIC1']; nodes=['SYSTEM',*REG]
read=lambda f:pd.read_csv(f,parse_dates=['ds']).set_index('ds')
H=read(W/'observed.csv')
C=pd.to_datetime(read(W/'clock.csv').last_component_issued)
assert H.index.equals(C.index)
Y=H.to_numpy()[24:]; dates=H.index[24:]
S=a.S
history=read(O/'cpor_aemo_underlying_october_20261107/observed.csv')
a.v=history.loc[:'2026-09-30 06:00'].to_numpy().var(0);a.iv=1/a.v
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(W/'REGISTRATION.txt')=='6d8fe62eadcc6e2e84298afff144fb4453e0f58e1f0d88d934a00919f5e3c8a6'
last=[];issues=dates-pd.Timedelta(minutes=30)
for issue in issues:last.append(np.flatnonzero((C<=issue).to_numpy())[-1])
UB=H[REG].to_numpy()[last];B=UB@S.T
assert np.array_equal(np.arange(24,24+len(Y))-last,np.full(len(Y),3))
arms=['Reference','Direct','Common','AdaptiveTwo','Full']
def run():
    stats=[];times=[];by_node=[];rates=[];folder=W/'records';folder.mkdir(exist_ok=True)
    # Warm both nontrivial kernels before measuring with three arbitrary positive credits.
    q=a.SafeProjection(S,a.v)
    wb=np.ones(5)*100;wu=wb+np.array([15,16,-2,4,-5]);warmb=wb@S.T;warmz=wu@S.T
    for arm in ['Full','AdaptiveTwo']:
        a.issue(arm,q,warmb,wb,warmz,wu,a.exact(warmb),[F(1)]*6)
    for idx,receipt in enumerate(sorted((W/'predictions').glob('*/*/receipt.json'))):
        bb,seed=receipt.parts[-3:-1];pfile=receipt.with_name('forecast.npz');assert sha(pfile)==json.loads(receipt.read_text())['sha256']
        inp=np.load(pfile);Z,U=legacy.cpu.old.project(S,inp['pir'],a.v)
        pub={ar:[] for ar in arms}; zero=lambda:[F(0) for _ in nodes]
        state={ar:zero() for ar in arms};pending={ar:[] for ar in arms};actual={ar:zero() for ar in arms}
        bad={ar:[0,0,np.zeros(6,bool)] for ar in arms}
        for t,(b,ub,z,u,obs) in enumerate(zip(B,UB,Z,U,Y)):
            bq=a.exact(b);yq=a.exact(obs);order=arms[(idx+t)%len(arms):]+arms[:(idx+t)%len(arms)]
            for ar in order:
                queue=pending[ar];matured=[row for row in queue if C.iloc[24+row[0]]<=issues[t]]
                for row in matured:
                    j,bj,pj,_=row
                    state[ar]=a.balance(state[ar],bj,pj,a.exact(Y[j]))
                queue[:]=[row for row in queue if C.iloc[24+row[0]]>issues[t]]
                free=[v-sum(row[3][i] for row in queue) for i,v in enumerate(state[ar])]
                stamp=time.perf_counter_ns()
                if ar=='Reference':p=b
                elif ar=='Direct':p=z
                elif ar=='Common':
                    p,f=legacy.ns['ray'](b,z,bq,free);ww=(1-f)*ub+f*u
                    assert np.min(ww)>=0 and np.allclose(S@ww,p,atol=2e-6,rtol=1e-8) and a.feasible(bq,a.exact(p),free)
                else:p=a.issue(ar,q,b,ub,z,u,bq,free)
                ms=(time.perf_counter_ns()-stamp)/1e6
                times.append(dict(bb=bb,run=int(seed),t=t,rule=ar,ms=ms))
                pq=a.exact(p);charge=a.excess(bq,pq)
                queue.append((t,bq,pq,charge))
                bad[ar][0]+=sum(g>f for g,f in zip(charge,free))
                actual[ar]=a.balance(actual[ar],bq,pq,yq)
                negative=np.array([v<0 for v in actual[ar]])
                bad[ar][1]+=int(negative.sum());bad[ar][2]|=negative
                pub[ar].append(p.copy())
        snap=dict(Y=Y,Z=Z,U=U,B=B,UB=UB,S=S,v=a.v,ready_ns=C.iloc[24:].to_numpy(dtype='datetime64[ns]').astype('int64'),issue_ns=issues.asi8)
        for ar in arms:
            arr=np.array(pub[ar]);snap[ar]=arr
            rm=np.sqrt(np.mean((arr-Y)**2,axis=0))
            af=np.sqrt(np.mean((B-Y)**2,axis=0))
            for i,n in enumerate(nodes):by_node.append(dict(bb=bb,run=int(seed),rule=ar,node=n,rmse=rm[i],ratio_to_reference=rm[i]/af[i]))
            stats.append(dict(bb=bb,run=int(seed),rule=ar,overall=rm.mean(),regions=rm[1:].mean(),**dict(zip(nodes,rm)),checks=Y.size,envelope=bad[ar][0],prefix=bad[ar][1],ever=bad[ar][2].sum()))
            if ar!='Direct':assert bad[ar][0]==bad[ar][1]==0
        np.savez_compressed(folder/f'{bb}_{seed}.npz',**snap)
        print('SCORED',bb,seed,flush=True)
    assert len(stats)==45
    d=pd.DataFrame(stats);d.to_csv(W/'stream_scores.csv',index=False)
    nodes_df=pd.DataFrame(by_node);nodes_df.to_csv(W/'node_scores.csv',index=False)
    td=pd.DataFrame(times);td.to_csv(W/'times.csv',index=False)
    measures=['overall','regions',*nodes]
    result=d.groupby('rule')[measures].mean().join(d.groupby('rule')[['checks','envelope','prefix','ever']].sum()).join(td.groupby('rule').ms.agg(mean_ms='mean',p95_ms=lambda s:s.quantile(.95)))
    result.to_csv(W/'summary.csv');print(result.to_string(),flush=True)
    full=d[d.rule=='Full'].set_index(['bb','run'])[measures].sort_index()
    rows=[]
    for comp in ['Reference','Direct','Common','AdaptiveTwo']:
        bench=d[d.rule==comp].set_index(['bb','run'])[measures].sort_index()
        gain=(1-full/bench)*100
        x=gain.reset_index();x['comparator']=comp;rates.extend(x.to_dict('records'))
        for k in measures:rows.append(dict(comparator=comp,level=k,gain=gain[k].mean(),min=gain[k].min(),max=gain[k].max(),wins=(gain[k]>0).sum(),runs=9))
    pd.DataFrame(rates).to_csv(W/'paired_streams.csv',index=False)
    pd.DataFrame(rows).to_csv(W/'gains.csv',index=False)
    print(pd.DataFrame(rows).query("level in ['overall','regions','SYSTEM']").to_string(index=False))
if __name__=='__main__':
    with threadpool_limits(1):run()

