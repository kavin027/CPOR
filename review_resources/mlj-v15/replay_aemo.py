"""Regenerate releases from frozen forecasts, including delayed settlement and reservations."""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import aemo_release as a

def main():
    folder=Path(__file__).resolve().parent/'aemo'
    totals={}
    streams=list(csv.DictReader((folder/'streams.csv').open()))
    assert len(streams)==36
    for f in sorted(folder.glob('*.npz')):
        block,_,_,bb,seed,*_=f.stem.split('_')
        labels=pd.read_csv(folder/f'block{block}_labels.csv',index_col=0).to_numpy()
        with np.load(f) as z:
            a.v=z['v'];a.iv=1/a.v
            qp=a.SafeProjection(a.S,a.v)
            Y,B,UB,Z,U=[z[k] for k in ('Y','B','UB','Z','U')]
            np.testing.assert_array_equal(Y,labels[24:])
            np.testing.assert_allclose(B,labels[21:-3],rtol=1e-12,atol=1e-8)
            for arm in ('Full','AdaptiveTwo'):
                stored=z[arm];issued=[];pending=[];state=[a.F(0)]*6;realized=state.copy()
                for t,(b,ub,target,u) in enumerate(zip(B,UB,Z,U)):
                    if t>=3:
                        j,bj,pj,_=pending.pop(0)
                        assert j==t-3
                        state=a.balance(state,bj,pj,a.exact(Y[j]))
                    free=[v-sum(x[3][i] for x in pending) for i,v in enumerate(state)]
                    bq=a.exact(b);p=a.issue(arm,qp,b,ub,target,u,bq,free)
                    pq=a.exact(p);charge=a.excess(bq,pq)
                    assert all(g<=v for g,v in zip(charge,free))
                    pending.append((t,bq,pq,charge))
                    realized=a.balance(realized,bq,pq,a.exact(Y[t]))
                    assert min(realized)>=0
                    issued.append(p.copy())
                # Report binary equality on the frozen toolchain. On another
                # platform reject material release differences; never relax exact protection.
                issued=np.asarray(issued)
                np.testing.assert_allclose(issued,stored,rtol=1e-9,atol=2e-6)
                err=np.sqrt(np.mean((issued-Y)**2,axis=0))
                row=next(r for r in streams if (r['block'],r['bb'],r['seed'],r['rule'])==(block,bb,seed,arm))
                np.testing.assert_allclose([err.mean(),err[1:].mean(),err[0]], [float(row[k]) for k in ('overall','regions','system')],rtol=1e-9,atol=2e-6)
                key=(block,arm)
                count,same,delta=totals.get(key,(0,0,0.))
                totals[key]=count+issued.size,same+int(np.array_equal(issued,stored)),max(delta,float(np.abs(issued-stored).max()))
    assert sorted(a[0] for a in totals.values())==[5184,5184,7776,7776]
    print('PASS AEMO: (node-dates, bitwise-identical streams out of 9, maximum release difference in MW)',totals,flush=True)

if __name__=='__main__':
    with threadpool_limits(1): main()


