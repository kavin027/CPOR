from release import *
from engine import certified

def release_repaired(x,target,kind):
    qp=SafeProjection(x['S'],x['v']);accounts=[Fraction(0) for _ in x['v']];previous=None;reservation=None;stats=[]
    out={key:[] for key in ['pred','witness','available_after','reserved','lo','hi','contraction','objective','gap','first_order_bound','rounding_move']}
    for i,t in enumerate(range(36,len(x['truth']))):
        if i:accounts=advance([a+r for a,r in zip(accounts,reservation)],x['b'][t-1],previous,x['truth'][t-1])
        b,ub=x['b'][t],x['ub'][t];lo,hi=safe_bounds(b,np.array([float(a) for a in accounts]));ql,qh=lo,hi
        if kind.endswith('Segment'):ql=np.maximum(lo,np.minimum(b,target[i]));qh=np.minimum(hi,np.maximum(b,target[i]))
        p,u,info=qp.solve(b,target[i],ub,ql,qh);initial=p.copy();pp=np.clip(p,ql,qh)
        # Repair only observable rounding at the boundary. The original witness,
        # box, reconstruction tolerance and exact reservation check are retained.
        for _ in range(16):
            if certified(b,pp,accounts,RR,K,True):break
            for j in range(len(b)):
                if not certified(b[j:j+1],pp[j:j+1],accounts[j:j+1],RR,K,True):pp[j]=np.nextafter(pp[j],b[j])
        if np.allclose(pp,x['S']@u,rtol=1e-8,atol=2e-6) and certified(b,pp,accounts,RR,K,True):p=pp;c=1.
        else:p,u,c=guard(b,p,ub,u,x['S'],ql,qh,accounts,RR,K)
        obj,gap=certificate(qp,b,p,target[i],ub,ql,qh);fw=gap
        if info.get('solver')=='Clarabel' and qp.eq.dual_value is not None:
            eta=np.asarray(qp.eq.dual_value).ravel()*info.get('objective_divisor',1.)
            xx=(p-b)/qp.s;tar=(target[i]-b)/qp.s;lower=(ql-b)/qp.s;upper=(qh-b)/qp.s
            lw=qp.lw.value;uw=qp.uw.value;cw=-qp.M.T@eta;xd=np.clip(tar-eta,lower,upper);wd=np.where(cw>=0,lw,uw)
            dual=.5*np.sum((xd-tar)**2)+eta@xd+cw@wd;dg=float(obj-dual)
            assert dg>=-1e-7*(1+obj),(dg,obj)
            gap=min(gap,max(0.,dg))
        risk=excess_bound(b,p,RR,K)
        for j in range(len(accounts)):accounts[j]-=risk[j]
        assert min(accounts)>=0
        grad=(p-target[i])/qp.s;endpoint=np.where(grad>=0,ql,qh)
        first=max(0.,float(np.sum(grad*(p-endpoint)/qp.s)))
        vals=dict(pred=p,witness=u,available_after=[float(a) for a in accounts],reserved=[float(a) for a in risk],lo=lo,hi=hi,contraction=c,objective=obj,gap=gap,first_order_bound=first,rounding_move=float(abs(p-initial).max()))
        for key,value in vals.items():out[key].append(value)
        stats.append(info);previous=p;reservation=risk
    return {**{key:np.asarray(a) for key,a in out.items()},'stats':stats}
