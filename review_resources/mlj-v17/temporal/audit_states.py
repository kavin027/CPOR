"""Audit archived first reports and reconstruct each saved free allowance with independent arithmetic."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
NI=['SYSTEM','NSW1','QLD1','SA1','TAS1','VIC1']
R=F(40401,40000)
def q(row):return [F.from_float(float(x)) for x in row]
def charge(b,p):
    # Compute the constrained maximizing observation and evaluate excess squared error there.
    y=max(F(0),(R*b-p)/(R-1))
    return (p-y)**2-R*(b-y)**2
def main():
    known=json.loads((P/'INPUT_HASHES.json').read_text(encoding='utf-8'))
    for name,h in known.items():
        assert hashlib.sha256((P/'inputs'/name).read_bytes()).hexdigest()==h,name
    src=pd.read_csv(P/'source_records.csv',parse_dates=['ds','issued'])
    first=src.sort_values(['issued','source']).drop_duplicates(['feed','ds','region'])
    assert first.shape[0]==120*5*2
    vv=sum(first[first.feed==f].pivot(index='ds',columns='region',values='MW').loc[:,NI[1:]] for f in ['op','pv'])
    yy=vv.assign(SYSTEM=vv.sum(1))[NI]
    ready=first.groupby('ds').issued.max().loc[vv.index].to_numpy('datetime64[ns]').astype('int64')
    assert np.all(np.diff(ready)>0)
    for ip in sorted((P/'inputs').glob('*.npz')):
        with np.load(ip) as inp,np.load(P/'expected'/ip.name) as out:
            np.testing.assert_allclose(yy.to_numpy()[24:],inp['Y'],rtol=1e-14,atol=1e-10)
            np.testing.assert_array_equal(ready[24:],inp['ready_ns'])
            for rule in ('Full','Independent'):
                p=out[rule];w=out[rule+'_w'];free=out[rule+'_free']
                np.testing.assert_allclose(p,w@inp['S'].T,rtol=1e-8,atol=2e-6)
                assert min(w.ravel())>=0
                for i in range(6):
                    costs=[];credits=[];prefix=F(0)
                    for t,(bf,pf,yf) in enumerate(zip(inp['B'][:,i],p[:,i],inp['Y'][:,i])):
                        b,pr,y=q([bf,pf,yf]); n=np.searchsorted(inp['ready_ns'][:t],inp['issue_ns'][t],side='right')
                        available=sum(credits[:n],F(0))-sum(costs[n:t],F(0))
                        assert available==F(str(free[t,i]))
                        c=charge(b,pr);assert c<=available
                        delta=R*(b-y)**2-(pr-y)**2;prefix+=delta;assert prefix>=0
                        costs.append(c);credits.append(delta)
    print('PASS: source-file clocks, all exact published snapshots, 10368 prefix/envelope/credit and tolerance-coherence checks.')
if __name__=='__main__':main()

