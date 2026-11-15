"""Classify a release against the image of a fixed component box."""
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
EPS=np.finfo(float).eps
def separate(S,lo,hi,p,delta,h):
    # Evaluate a proposed separating direction by exact arithmetic.
    hs=list(map(F,h))
    lhs=sum(hj*F(pi)-abs(hj)*F(di) for hj,pi,di in zip(hs,p,delta))
    if not any(hs):return False
    support=F(0)
    for j in range(S.shape[1]):
        c=sum(hs[i]*F(S[i,j]) for i in np.flatnonzero(S[:,j]))
        support+=c*F(hi[j] if c>=0 else lo[j])
    return lhs>support
def classify(S,lo,hi,p,wf):
    m=len(lo)
    delta=2*(2e-6+1e-8*np.abs(p))
    # All summands are nonnegative. This guard dominates m IEEE roundoffs,
    # including the final subtraction and comparisons.
    low=S@lo; high=S@hi
    pad=8*(m+4)*EPS*(1+high+np.abs(p))
    below=p <low-delta-pad;above=p>high+delta+pad
    physical_low=wf<lo-2*(2e-6+1e-8*np.abs(wf))
    physical_high=wf>hi+2*(2e-6+1e-8*np.abs(wf))
    witness=physical_low|physical_high
    if (below|above).any():return 'out_coordinate',below,above,witness
    def acceptable(w):
        return np.all(np.abs(S@w-p)+pad<=delta)
    if acceptable(np.clip(wf,lo,hi)):return 'in_clipped',below,above,witness
    # Could a DIFFERENT latent witness recover p? Minimize max_i |S w-p|/delta_i
    # using a scaled LP; only an independently checked feasible point or
    # separator determines the recorded class.
    scale=1e6*delta
    M=S*(hi-lo)/scale[:,None];r=(p-low)/scale
    a=np.vstack([np.column_stack([M,-np.ones(len(p))]),np.column_stack([-M,-np.ones(len(p))])])
    sol=linprog(np.r_[np.zeros(m),1.],A_ub=a,b_ub=np.r_[r,-r],bounds=[(0,1)]*m+[(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
    if sol.success:
        w=np.clip(lo+np.clip(sol.x[:-1],0,1)*(hi-lo),lo,hi)
        if acceptable(w):return 'in_lp',below,above,witness
        h=(sol.ineqlin.marginals[:len(p)]-sol.ineqlin.marginals[len(p):])/scale
        if separate(S,lo,hi,p,delta,h) or separate(S,lo,hi,p,delta,-h):return 'out_joint',below,above,witness
    return 'unresolved',below,above,witness
