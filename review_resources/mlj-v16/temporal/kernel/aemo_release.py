from fractions import Fraction as F
import numpy as np
from release import one_action, SafeProjection
from engine import certified
from grouped import grouped
R=F(40401,40000)
S=np.vstack([np.ones(5),np.eye(5)])
masks=[np.array([i for i in range(5) if bits & (1<<i)]) for bits in range(1,31,2)]
def exact(a):return [F.from_float(float(x)) for x in a]
def feasible(b,p,a):return all(g<=v for g,v in zip(excess(b,p),a))
ns=dict(exact=exact,feasible=feasible)
kernel=dict(ml=one_action,certified=lambda b,p,a,r,k:certified(b,p,a,r,k,True))
def issue(arm,qp,b,ub,z,u,bq,free):
    if arm=='Full':p,w,_=kernel['ml'](qp,S,b,ub,z,free)
    elif not any(free):p,w=b,ub
    elif ns['feasible'](bq,ns['exact'](z),free):p,w=z,u
    else:
        opts=[]
        for idx,mask in enumerate(masks):
            pp,ww,*_=grouped(S[:,mask],mask,b,ub,z,u,free,bq,iv,ns,float(R))
            opts.append((np.sum((pp-z)**2/v),idx,pp,ww))
        _,_,p,w=min(opts,key=lambda x:x[:2])
    assert kernel['certified'](b,p,free,R,R-1) and min(w)>=0 and np.allclose(S@w,p,rtol=1e-8,atol=2e-6)
    return p

def excess(b,p):
    return [x*x-R*y*y if x>R*y else R*(x-y)**2/(R-1) for y,x in zip(b,p)]

def balance(a,b,p,y):
    return [c+R*(bb-yy)**2-(pp-yy)**2 for c,bb,pp,yy in zip(a,b,p,y)]