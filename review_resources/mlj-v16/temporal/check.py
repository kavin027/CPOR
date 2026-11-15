"""Replay every forecast from the preserved inputs and an independent first-report availability table."""
import sys,csv,json,hashlib,zipfile,io
from pathlib import Path
import numpy as np,pandas as pd
from fractions import Fraction as F
from threadpoolctl import threadpool_limits
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'kernel'))
import aemo_release as a
from common_ray import ray
N=['SYSTEM','NSW1','QLD1','SA1','TAS1','VIC1']
def main():
    assert hashlib.sha256((W/'REGISTRATION.txt').read_bytes()).hexdigest()=='6d8fe62eadcc6e2e84298afff144fb4453e0f58e1f0d88d934a00919f5e3c8a6'
    rows=pd.read_csv(W/'source_records.csv',parse_dates=['ds','issued'])
    first=rows.sort_values(['issued','source']).drop_duplicates(['feed','ds','region'])
    assert first.shape[0]==120*2*5
    regions=sum(first[first.feed==feed].pivot(index='ds',columns='region',values='MW').loc[:,N[1:]] for feed in ['op','pv'])
    labels=regions.assign(SYSTEM=regions.sum(axis=1))[N]
    ready=first.groupby('ds').issued.max().loc[regions.index].to_numpy('datetime64[ns]')
    dates=regions.index.to_numpy('datetime64[ns]'); issue=dates[24:]-np.timedelta64(30,'m')
    assert len(issue)==96 and np.all(dates[1:]-dates[:-1]==np.timedelta64(30,'m'))
    results=pd.read_csv(W/'stream_scores.csv')
    maxdiff=0;eq=0;npaths=0;afters=[];nodelist=[]
    for f in sorted((W/'records').glob('*.npz')):
        bb,s=f.stem.split('_');d=np.load(f);Y=d['Y'];B=d['B'];UB=d['UB'];last=np.array([np.flatnonzero(ready<=t)[-1] for t in issue])
        np.testing.assert_array_equal(d['ready_ns'],ready[24:].astype('int64'));np.testing.assert_array_equal(d['issue_ns'],issue.astype('int64'))
        np.testing.assert_allclose(Y,labels.to_numpy()[24:],rtol=1e-14,atol=1e-10)
        np.testing.assert_allclose(UB,regions.to_numpy()[last],rtol=1e-14,atol=1e-10)
        np.testing.assert_array_equal(UB@a.S.T,B)
        a.v=d['v'];a.iv=1/a.v;q=a.SafeProjection(a.S,a.v)
        errors={}
        for ar in ['Reference','Common','AdaptiveTwo','Full','Direct']:
            issued=[];qp=[];vpre=np.zeros(6,int);bad=0;prefix=0
            zeros=[F(0)]*6;actual=zeros.copy()
            for j,(b,ub,z,u,y) in enumerate(zip(B,UB,d['Z'],d['U'],Y)):
                # Rebuild available balance from scratch at every issue, including any pending reports.
                free=zeros.copy()
                for k,(ref,release,charge) in enumerate(qp):
                    if ready[24+k]<=issue[j]:
                        free=a.balance(free,ref,release,a.exact(Y[k]))
                    else:free=[v-g for v,g in zip(free,charge)]
                bq=a.exact(b)
                if ar=='Full' or ar=='AdaptiveTwo':p=a.issue(ar,q,b,ub,z,u,bq,free)
                elif ar=='Common':p,_=ray(b,z,bq,free)
                elif ar=='Reference':p=b
                else:p=z
                published=a.exact(p);charge=a.excess(bq,published);qp.append((bq,published,charge))
                bad+=sum(g>v for g,v in zip(charge,free));actual=a.balance(actual,bq,published,a.exact(y))
                vpre+=np.array([v<0 for v in actual]);issued.append(p)
            issued=np.array(issued);np.testing.assert_allclose(issued,d[ar],rtol=1e-9,atol=2e-6)
            maxdiff=max(maxdiff,float(abs(issued-d[ar]).max()));npaths+=1;eq+=np.array_equal(issued,d[ar])
            row=results[(results.bb==bb)&(results.run==int(s))&(results.rule==ar)].iloc[0]
            assert (bad,int(vpre.sum()),int((vpre>0).sum()))==(row.envelope,row.prefix,row.ever)
            err=np.sqrt(np.mean((issued-Y)**2,axis=0));errors[ar]=(issued-Y)**2
            np.testing.assert_allclose(row[N].astype(float),err,rtol=1e-9,atol=2e-6)
        for cut in [0,4]:
            rm={k:np.sqrt(v[cut:].mean(0)) for k,v in errors.items()}
            for comp in ['AdaptiveTwo','Reference','Common','Direct']:
                gain=100*(1-rm['Full'].mean()/rm[comp].mean());afters.append(dict(cut=cut,bb=bb,run=s,comparator=comp,overall_gain=gain))
            if cut==0:
                for i,node in enumerate(N):
                    nodelist.append(dict(bb=bb,run=s,node=node,rmse_ratio=rm['Full'][i]/rm['Reference'][i],gain=100*(1-rm['Full'][i]/rm['AdaptiveTwo'][i])))
    assert len(results)==45 and npaths==45
    cuts=pd.DataFrame(afters);print(cuts.groupby(['cut','comparator']).overall_gain.mean(),flush=True)
    n=pd.DataFrame(nodelist)
    print('Node-run wins', (n.gain>0).sum(),'/',len(n),'median gain',n.gain.median(),'worst',n.loc[n.gain.idxmin()].to_dict(),'max ratio',n.rmse_ratio.max(),flush=True)
    print('PASS: regenerated',npaths,'paths; identical',eq,'largest delta MW',maxdiff,flush=True)
if __name__=='__main__':
    with threadpool_limits(1):main()
