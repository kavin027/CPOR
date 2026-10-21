import numpy as np
from engine import certified

def guard(b,p,ub,u,S,lo,hi,accounts,rr,k):
    delta=p-b;du=u-ub
    if np.all(p>=lo) and np.all(p<=hi) and certified(b,p,accounts,rr,k,True):return p,u,1.
    ratios=np.ones_like(delta);pos=delta>0;neg=delta<0
    ratios[pos]=(hi[pos]-b[pos])/delta[pos];ratios[neg]=(lo[neg]-b[neg])/delta[neg]
    lam=max(0.,min(1.,float(ratios.min())))
    # Bounds already have their frozen interior margin. Search representable convex
    # contractions and independently certify every proposed released float.
    for i in range(40):
        lam=np.nextafter(lam,0.) if i<8 else lam*(1-1e-12) if i<24 else lam*.5
        pp=b+lam*delta;uu=ub+lam*du
        if uu.min()>=0 and np.all(pp>=lo) and np.all(pp<=hi) and certified(b,pp,accounts,rr,k,True):
            np.testing.assert_allclose(pp,uu@S.T,rtol=1e-8,atol=2e-6)
            return pp,uu,lam
    return b.copy(),ub.copy(),0.
