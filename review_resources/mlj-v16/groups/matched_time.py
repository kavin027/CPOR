"""Serial same-machine latency audit, one fixed DLinear/1 continuous case per dataset."""
import platform, sys,time,csv,hashlib
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_groups import DEFAULT_ARCHIVE,DEFAULT_INDEX,HERE,ROOT,Ledger,issue,check_hash

def main():
    # Use the exact release and fallback from the publicly frozen v15 review kit.
    dep=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'output/cpor_mlj_v15_20261115/kit'
    sys.path.insert(0,str(dep))
    from release import SafeProjection,one_action
    selected=[r for r in csv.DictReader(DEFAULT_INDEX.open(encoding='utf-8-sig')) if (r['backbone'],r['run'],r['part'])==('DLinear','1','continuous')]
    assert len(selected)==7
    out=[]
    for j,row in enumerate(selected):
        c=DEFAULT_ARCHIVE/row['context_archive'];tgt=DEFAULT_ARCHIVE/row['PIRCoherent_archive']
        with np.load(c) as x,np.load(tgt) as z:
            S=x['S'];v=x['v'];B=x['b'][36:];UB=x['ub'][36:];Y=x['truth'][36:];Z=z['pred'];U=z['witness']
        expected={'Full':np.load(DEFAULT_ARCHIVE/row['PIRFull_archive'])['pred'],
                  'Independent':np.load(HERE/'runs'/row['case']/'Independent.npz')['pred']}
        ledger={a:Ledger(B,Y,Z) for a in expected}
        qp=SafeProjection(S,v)
        # Compile and initialize both numerical paths without using future scoring labels.
        warmB=x0=B[0];warmu=UB[0];warmz=Z[0]
        one_action(qp,S,warmB,warmu,warmz,[Fraction(1)]*len(v))
        la=Ledger(B[:1],Y[:1],Z[:1]);la.A[:]=40000*2**(2*la.k)
        issue('Independent',S,v,warmB,warmu,warmz,U[0],0,la,{})
        for t,(b,ub,z,u) in enumerate(zip(B,UB,Z,U)):
            for arm in (('Full','Independent') if (j+t)%2 else ('Independent','Full')):
                l=ledger[arm]
                if arm=='Full':
                    credit=[Fraction(a,40000*2**(2*l.k)) for a in l.A]
                start=time.perf_counter_ns()
                if arm=='Full':p,w,_=one_action(qp,S,b,ub,z,credit);pi=l.toint(p,t)
                else:p,pi,*_=issue('Independent',S,v,b,ub,z,u,t,l,{})
                assert not np.any(l.flags(t,pi))
                ms=(time.perf_counter_ns()-start)/1e6
                assert np.allclose(p,expected[arm][t],atol=2e-6,rtol=1e-8)
                l.step(t,pi)
                out.append([row['dataset'],arm,t,ms,np.abs(p-expected[arm][t]).max()])
    df=pd.DataFrame(out,columns=['dataset','rule','issue','ms','max_abs_reproduction_difference'])
    df.to_csv(HERE/'records/matched_time.csv.gz',index=False)
    print(df.groupby('rule').agg(mean_ms=('ms','mean'),p95_ms=('ms',lambda x:x.quantile(.95)),max_replay_diff=('max_abs_reproduction_difference','max'),issues=('issue','size')))
if __name__=='__main__':
    with threadpool_limits(1):main()

