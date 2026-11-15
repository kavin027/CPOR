"""Exact free-credit interface to the frozen Full and independent-coordinate publishers."""
import inspect,sys
from fractions import Fraction
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'kernel'))
import aemo_release as ar
from release import one_action, SafeProjection
import run_groups as gr
S=ar.S
# Return the witness already computed by the original function; retain every original numerical decision.
source=inspect.getsource(gr.issue)
for old,new in [("return b,led.B[t],0.,0.,'zero'","return b,led.B[t],0.,0.,'zero',ub"),
                ("return z,led.Z[t],0.,0.,'target'","return z,led.Z[t],0.,0.,'target',u"),
                ("('guard' if movement or corrections else 'accepted')","('guard' if movement or corrections else 'accepted'),witness")]:
    if source.count(old)!=1:raise RuntimeError('Input rule signature changed')
    source=source.replace(old,new)
namespace=gr.__dict__.copy()
exec(compile(source,'<independent returning its latent witness>','exec'),namespace)
issue_ind=namespace['issue']

class FreeLedger:
    """Exact first-report ledger supplied as rational free credit by caller."""
    def __init__(self,bq,zq,free):
        self.B=[bq];self.Z=[zq];self.A=free
    def toint(self,p,t):return ar.exact(p)
    def flags(self,t,p):
        return np.fromiter((x<0 or h>v for x,h,v in zip(p,ar.excess(self.B[t],p),self.A)),bool)
    def allowance(self):return np.array(self.A,dtype=float)

def publish(arm,solver,v,b,ub,z,u,free):
    """Use the supplied arbitrary feasible state. Return stored p, witness, exact p, decision and diagnostic gap."""
    bq=ar.exact(b);zq=ar.exact(z)
    if arm=='Full':
        p,w,info=one_action(solver,S,b,ub,z,free)
        pq=ar.exact(p);decision='issued';gap=info['gap']
    elif arm=='Independent':
        p,pq,gap,_,decision,w=issue_ind('Independent',S,v,b,ub,z,u,0,FreeLedger(bq,zq,free),{})
    else:raise ValueError(arm)
    if not np.allclose(p,S@w,atol=2e-6,rtol=1e-8) or np.min(w)<0 or np.min(p)<0:
        raise ArithmeticError('unacceptable representation')
    if not ar.feasible(bq,pq,free):raise ArithmeticError('failed exact envelope')
    return p,w,pq,decision,gap
