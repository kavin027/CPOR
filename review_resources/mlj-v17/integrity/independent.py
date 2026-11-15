"""Independent, integer-arithmetic replay of archived trajectory certificates.

python independent.py --kit PATH --audit PATH [--out summary-dir]
No optimization software or imported certificate-verification implementation.
"""
import argparse, csv, hashlib, json
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path
import numpy as np
from geometry import check as geometry

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def unpack(a,key):
    v=a[key]; width=v.shape[-1]
    return np.array([int.from_bytes(i,"little",signed=True) for i in v.reshape(-1,width)],dtype=object).reshape(v.shape[:-1])

def dyadic(a):
    v,idx=np.unique(np.asarray(a),return_inverse=True)
    pairs=[float(x).as_integer_ratio() for x in v]
    e=max(q.bit_length()-1 for _,q in pairs)
    z=np.array([p*(1<<(e-q.bit_length()+1)) for p,q in pairs],object)
    return z[idx].reshape(np.shape(a)),e

def score(ri,e):
    t,n=ri.shape; denominator=t*(1<<(2*e))
    lb=0; ub=0
    for s in np.sum(ri*ri,axis=0):
        num=int(s)<<180; k=isqrt(num//denominator)
        lb+=k; ub+=k+(k*k*denominator<num)
    return Q(lb,n<<90),Q(ub,n<<90)

def feasible(r,b,d,e,seq):
    v,k=dyadic(seq)
    assert np.all((v>=0)&(v<=(1<<k))) and np.all(v[0]==0)
    s=(d*v[:,None,:]).sum(2); r=r*(1<<k); b=b*(1<<k)
    credit=np.zeros(r.shape[1],dtype=object)
    for j,(ds,rr,bb) in enumerate(zip(s,r,b)):
        # sup_{k<=b}(d^2+2kd-eta*k^2), evaluated at min(d/eta, b).
        envelope=np.where(40000*ds<=401*bb,40401*40000*ds*ds,401*(40000*ds*ds+80000*bb*ds-401*bb*bb))
        assert np.all(ds+bb>=0) and np.all(envelope<=401*credit)
        credit+=40401*rr*rr-40000*(rr+ds)**2
    return score(r+s,e+k)[1]

class AffineSum:
    def __init__(self,t,n):
        self.g=np.zeros((t,2),dtype=object); self.c=0; self.k=0; self.n=n
    def scale(self,exp):
        if exp>self.k:
            self.g*=1<<(exp-self.k); self.c*=1<<(exp-self.k); self.k=exp
        return 1<<(self.k-exp)
    def add(self,c,ce,g,ge):
        self.scale(max(ce,ge))
        self.c+=int(c)*self.scale(ce)
        self.g[:len(g)]+=g*self.scale(ge)
    def bound(self):
        result=self.c+int(np.minimum(self.g[1:],0).sum())
        return Q(result,40000*self.n*(1<<self.k))

def cuts(rec,r,b,d,e,de,online,endpoint,ed):
    t,n=r.shape; accum=AffineSum(t,n)
    mass=sum(Q(float(mu)) for (_,i,_,_),mu in zip(rec['meta'],rec['dual']) if i<0 and mu>0)
    support={}; pen=Q(0)
    edmax=max(ed,de,e); disp=abs(endpoint)*(1<<(edmax-ed));disp[0]=0
    margin=abs(d).sum(2)*(1<<(edmax-de)); residual=abs(r)*(1<<(edmax-e))
    change=disp*(2*(residual+margin)+disp)
    history=np.zeros_like(change);history[1:]=np.cumsum(change[:-1],axis=0)
    cap=401*history+2*40401*disp*(margin+disp)
    penalty=Q(int(disp.max()),1<<edmax)
    for (ai,when,node,scale),mu in zip(rec['meta'],rec['dual']):
        if mu<=0: continue
        ai,when,node=int(ai),int(when),int(node)
        if ai not in support: support[ai]=dyadic(rec['anchors'][ai])
        s,se=support[ai]
        if when<0:
            # A different dual witness from the archived outward-arithmetic checker.
            err=rec['r_float']+(rec['D_float']*rec['anchors'][ai][:,None,:]).sum(2)
            norm=np.sqrt(t*np.sum(err*err,axis=0))
            z=np.divide(err,norm,out=np.zeros_like(err),where=(norm>0))*(1-1e-12)
            v,ve=dyadic(z)
            assert np.all(t*(v*v).sum(0)<=1<<(2*ve))
            weight=Q(float(mu))/max(1,mass)
            w=(weight*(1<<62)).numerator//(weight*(1<<62)).denominator
            c=40000*w*(v*r).sum()
            g=40000*w*(v[:,:,None]*d).sum(1)
            accum.add(c,ve+e+62,g,ve+de+62)
        else:
            j=node;i=when; dj=d[:i+1,j]
            sq=(dj*s[:i+1]).sum(1)
            fl=min(float(rec['D_float'][i,j]@rec['anchors'][ai,i])/.010025,float(Q(int(b[i,j]),1<<e)))
            k,den=fl.as_integer_ratio(); ke=den.bit_length()-1
            h=max(ke,se+de,e)
            k*=1<<(h-ke);assert k <= int(b[i,j])*(1<<(h-e))
            q=sq*(1<<(h-se-de)); rr=r[:i,j]*(1<<(h-e))
            a=(q*q).sum(); old=(rr*rr).sum()
            cc=-40000*int(a)-401*(int(old)+k*k)
            q[:i]+=rr;q[i]+=k
            weight=float(mu*online/float(scale)); w,wden=weight.as_integer_ratio(); we=wden.bit_length()-1
            accum.add(w*n*cc,2*h+we,2*40000*n*w*q[:,None]*dj,h+de+we)
            pen+=Q(w*int(cap[i,j]),401*wden*(1<<(2*edmax)))
    return accum.bound(),penalty+pen

def verify(row,old,kit,audit):
    case=row['case'];path=kit/row['download']; side=audit/'differences'/(case+'.npz')
    for p,expected in ((path,row['sha256']),(side,old['sidecar_sha256'])):
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    with np.load(path) as raw,np.load(side) as diff:
        a={k:raw[k] for k in ('cap','residual','direction','exponent','full_error','full_exponent','anchors','r_float','D_float','dual','meta','path')}
        b,r,d=[unpack(a,k) for k in ('cap','residual','direction')]; e=int(a['exponent'])
        ex=int(diff['exponent']); eb,ez=[unpack(diff,k) for k in ('eb','ez')]
        ee=max(e,ex);dl=d*(1<<(ee-e));dl[:,:,1]+=(ez-eb)*(1<<(ee-ex))
        f0,f1=score(unpack(a,'full_error'),int(a['full_exponent']))
        up=feasible(r,b,d,e,a['path'])
        L,_=cuts(a,r,b,d,e,e,float(row['online']),np.zeros_like(eb),ex)
        ll,pen=cuts(a,r,b,dl,e,ee,float(row['online']),eb,ex)
        pl=ll-pen
        assert L<=up
        assert (L>f1)==(float(old['signed_separation_lower'])>0)
        assert (ll>f1)==(float(old['literal_separation_lower'])>0)
        assert (pl>f1)==(float(old['physical_separation_lower'])>0)
        assert (up<f0)==(float(old['reverse_witness_margin_lower'])>0)
        assert L>f1 or up<f0
        assert abs(float(pen)-float(old['physical_transfer_penalty_upper'])) <=1e-20
        # Compare exact rational inequality to float binary upper bound in v16.
        assert pen<=Q(float(old['physical_transfer_penalty_upper']))
        def val(z):return float(Q(int(np.abs(z).max()),1<<ex))
        return dict(case=case,protocol=row['protocol'],closed=float(L-f1),literal=float(ll-f1),physical=float(pl-f1),
                    reverse=float(f0-up),penalty=float(pen),ref_error=val(eb),target_error=val(ez),direction_error=val(ez-eb),
                    relative_physical_pct=float(100*(pl-f1)/f1),identified=int(old['identified_subset']))

def write_csv(path,records):
    with path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('kit','audit','out'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args(); geometry()
    if not __debug__:raise RuntimeError('Disable Python -O')
    old={r['case']:r for r in rows(a.audit/'certificate_margins.csv')}
    records=[]
    for r in rows(a.kit/'inputs.csv'):
        records.append(verify(r,old[r['case']],a.kit,a.audit))
        if len(records)%25==0:print('Verified',len(records),'trajectories',flush=True)
    assert len({r['case'] for r in records})==len(records)==189
    summary=[]
    for protocol in ('window','continuous'):
        for subset in (False,True):
            selected=[r for r in records if r['protocol']==protocol and (not subset or r['identified'])]
            positive=[r for r in selected if r['closed']>0]; rev=[r for r in selected if r['reverse']>0]
            counts=dict(protocol=protocol,subset='identified' if subset else 'all',paths=len(selected),strict=len(positive),reverse_cases=len(rev))
            extrema={}
            for col in ('closed','literal','physical','relative_physical_pct','reverse','penalty','ref_error','target_error','direction_error'):
                pool,op=(rev,min) if col=='reverse' else (positive,min) if col in ('closed','literal','physical','relative_physical_pct') else (selected,max)
                best=op(pool,key=lambda r:r[col])
                extrema[col]=best[col]; extrema[col+'_case']=best['case']
            summary.append({**counts,**extrema})
    a.out.mkdir(parents=True,exist_ok=True)
    write_csv(a.out/'exact_margins.csv',records);write_csv(a.out/'summary.csv',summary)
    print(json.dumps(summary,indent=2));print('PASS independently verified 189 path and transfer certificates')

if __name__=='__main__':main()

