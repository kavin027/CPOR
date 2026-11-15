"""Replay all six flexible rules on a recorded demand path with no archive checkout."""
import sys,hashlib,json
from fractions import Fraction
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_groups import HERE,RULES,Ledger,exact,split_fixed,issue
from audit_outputs import audit

def check():
    demo=HERE/'demo';src=json.loads((demo/'source.json').read_text())
    p=demo/(src['case']+'.npz')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==src['sha256']
    with np.load(p) as f:
        S,v=f['S'],f['v'];B,UB,Z,U,Y=[f[k] for k in ['B','UB','Z','U','Y']]
        fixed={k:split_fixed(S,k) for k in (4,8)}
        for arm in RULES:
            # Only the current anchor and candidate are visible at decision time.
            # 256 fractional bits suffice for all stored values in this case.
            l=Ledger.__new__(Ledger);l.k=256;l.A=np.zeros(len(v),dtype=object)
            issued=[]
            for t,(b,ub,z,u) in enumerate(zip(B,UB,Z,U)):
                l.B=[exact(b,256)];l.Z=[exact(z,256)]
                pred,pi,*_=issue(arm,S,v,b,ub,z,u,0,l,fixed)
                issued.append(pred.copy())
                # The observation is delivered and then charged to this arm.
                yi=exact(Y[t],256)
                l.A+=40401*(l.B[0]-yi)**2-40000*(pi-yi)**2
                assert min(l.A)>=0
            err=np.max(np.abs(np.array(issued)-f[arm]))
            np.testing.assert_allclose(issued,f[arm],atol=2e-6,rtol=1e-8)
            env,pre=audit(B,Y,np.asarray(issued))
            assert not env.any() and not pre.any()
            print(arm,'releases',len(issued),'checks',Y.size,'max_difference_MW',err,'RMSE',np.sqrt(np.mean((issued-Y)**2,axis=0)).mean())
if __name__=='__main__':
    with threadpool_limits(1):check()

