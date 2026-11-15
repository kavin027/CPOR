"""Validate feasible paths with dyadic integers and dual bounds with outward arithmetic."""
import math
import numpy as np
from fractions import Fraction as F
from math import isqrt
dn=lambda a:np.nextafter(a,-np.inf)
up=lambda a:np.nextafter(a,np.inf)
def sum_lo(a,axis=0):
    v=np.moveaxis(np.asarray(a),axis,0)
    while len(v)>1:
        n=len(v)//2
        z=dn(v[:2*n:2]+v[1:2*n:2])
        v=np.concatenate([z,v[-1:]],axis=0) if len(v)%2 else z
    return v[0]
def iadd(x,y):return dn(x[0]+y[0]),up(x[1]+y[1])
def imul(x,y):
    prods=np.broadcast_arrays(x[0]*y[0],x[1]*y[0],x[0]*y[1],x[1]*y[1])
    return dn(np.minimum.reduce(prods)),up(np.maximum.reduce(prods))
def point(x):return x,x
def dyadic(x):
    x=np.asarray(x)
    unique,inv=np.unique(x.ravel(),return_inverse=True)
    pairs=[float(z).as_integer_ratio() for z in unique]
    e=max(q.bit_length()-1 for _,q in pairs)
    v=np.array([p<<(e-q.bit_length()+1) for p,q in pairs],dtype=object)
    return v[inv].reshape(x.shape),e
def exact_inputs(model,ub,u,mask):
    s,es=dyadic(model.S[:,mask]);ut,eu=dyadic(u[:,mask]);bt,ebt=dyadic(ub[:,mask])
    ec=max(eu,ebt); du=ut*(1<<(ec-eu))-bt*(1<<(ec-ebt))
    q0=np.zeros(model.B.shape,dtype=object)
    for i,row in enumerate(s):
        for j in np.flatnonzero(row):q0[:,i]+=row[j]*du[:,j]
    eb=ec+es; b,bb=dyadic(model.B); y,ey=dyadic(model.Y);z,ez=dyadic(model.Z)
    e=max(eb,bb,ey,ez);B=b*(1<<(e-bb));Y=y*(1<<(e-ey));Z=z*(1<<(e-ez));q0=q0*(1<<(e-eb))
    D=np.stack([q0,Z-B-q0],axis=-1)
    def ib(v):
        # D is exactly representable as integers times a power of two.
        f=np.ldexp(v.astype(float),-e);return dn(f),up(f)
    return B,Y,D,e,ib(D),ib(B-Y)
def primal(model,data,x,replay=False):
    B,Y,D,e,*_=data
    for shrink in ([1] if replay else [1,1-1e-10,1-1e-8,1-1e-6,1-1e-4]):
        xx=np.clip(x*shrink,0.,1.)
        # Keep zero-reference coordinates inside the nonnegative cone when
        # the second stored direction differs by a binary rounding unit.
        if not replay: xx=(1-2.0**-40)*xx+2.0**-40*xx.mean(1,keepdims=True)
        v,k=dyadic(xx);den=1<<(e+k)
        d=(D*v[:,None,:]).sum(2);b=B*(1<<k);r=(B-Y)*(1<<k)
        # Accumulator is 40000 times allowance in the shared dyadic squared units.
        A=np.zeros(model.n,dtype=object); good=True; er=(r+d)**2
        for t in range(model.T):
            di=d[t];bi=b[t];large=40000*di>401*bi
            if np.any(di+bi<0) or np.any(np.where(large,401*(40000*di*di+80000*bi*di-401*bi*bi),40000*40401*di*di)>401*A):
                problem=(t,np.flatnonzero(di+bi<0).tolist(),np.flatnonzero(np.where(large,401*(40000*di*di+80000*bi*di-401*bi*bi),40000*40401*di*di)>401*A).tolist());good=False;break
            A+=40401*r[t]**2-40000*er[t]
        if good:
            Q=1<<80
            def ceil_rm(val):
                top=val*Q**2;bottom=model.T*den**2;s=isqrt(top//bottom)
                return s+int(s*s*bottom<top)
            hi=F(sum(ceil_rm(v) for v in er.sum(0)),model.n*Q)
            return up(float(hi)),xx
    raise ArithmeticError(('dyadic primal infeasible',problem))
def dual(model,data,meta,anchors,mult):
    _,_,_,_,D,r=data; n,T=model.n,model.T
    g=np.zeros((T,2));c=0.
    ix=np.flatnonzero(mult>0);objmass=-sum_lo(-mult[[i for i in ix if meta[i][1]<0]]) if any(meta[i][1]<0 for i in ix) else 1.
    for i in ix:
        ai,t,j,scale=meta[i];x=anchors[ai]
        if t<0:
            # Arbitrary norm-bounded dual direction for every node. Individual
            # directions and normalizations are verified on the stored floats.
            err=model.r+(model.D*x[:,None,:]).sum(2)
            norm=up(np.sqrt(up(T*(-sum_lo(-up(err**2),0)))))
            v=np.divide(err,norm,out=np.zeros_like(err),where=norm>0)
            maxnorm=up(np.sqrt(up(T*(-sum_lo(-up(v*v),0)))))
            v/=up(np.maximum(1,maxnorm))*(1+1e-12)
            assert np.all(up(T*(-sum_lo(-up(v*v),0)))<=1.)
            p=point(v);lo=imul(p,r)[0]
            cc=dn(sum_lo(sum_lo(lo))/n)
            gg=dn(sum_lo(imul(point(v[:,:,None]),D)[0],1)/n)
            weight=max(0.,dn(mult[i]/max(1.,objmass)))*(1-1e-13)
        else:
            dr=(D[0][:t+1,j],D[1][:t+1,j])
            rr=(r[0][:t,j],r[1][:t,j]); dx=imul(dr,point(x[:t+1]))
            dd=iadd((dx[0][:,0],dx[1][:,0]),(dx[0][:,1],dx[1][:,1]))
            eta=(dn(.010025),up(.010025))
            k=min(float(model.D[t,j]@x[t])/.010025,model.B[t,j])
            re=iadd(rr,(dd[0][:t],dd[1][:t]));yt=iadd((dd[0][-1:],dd[1][-1:]),point(k))
            h=(np.concatenate([re[0],yt[0]])[:,None],np.concatenate([re[1],yt[1]])[:,None])
            gg=dn(2*imul(h,dr)[0])
            d2=up(np.maximum(np.abs(dd[0]),np.abs(dd[1]))**2)
            r2=up(np.maximum(np.abs(rr[0]),np.abs(rr[1]))**2)
            cc=dn(dn(sum_lo(-d2)+sum_lo(-up(eta[1]*r2)))-up(eta[1]*up(k*k)))
            weight=mult[i]*model.online/scale
        g[:len(gg)]=dn(g[:len(gg)]+dn(weight*gg)); c=dn(c+dn(weight*cc))
    if ix.size: 
        weights=[max(0.,dn(mult[i]/max(1.,objmass)))*(1-1e-13) for i in ix if meta[i][1]<0]
        assert sum(map(F,weights))<=1
    return dn(c+sum_lo(np.minimum(g[1:],0.).ravel()))

