"""Rebuild CPOR fixed-partition certificates from archived trajectories (NumPy, Python 3.10+)."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
RR=F(40401,40000)

def exact(v):
    return F(v.item() if isinstance(v,np.generic) else v)

def exact_envelope(b,p):
    return p*p-RR*b*b if p>RR*b else RR*(p-b)**2/(RR-1)

def partition(S):
    target_size = (S.shape[1] + 1) // 2
    selected = set()
    supports = [set(np.flatnonzero(row > 0)) for row in S]
    for s in sorted(supports, key=len, reverse=True):
        if selected.isdisjoint(s) and len(selected) + len(s) <= target_size:
            selected.update(s)
    for j in range(S.shape[1]):
        if len(selected) == target_size:
            break
        selected.add(j)
    return np.array(sorted(selected), dtype=int)

def clip(vertices, a, b, cap):
    """Intersect a convex polygon (possibly a segment) with one half-plane."""
    out = []
    start = vertices[-1]
    ds = start[0]*a + start[1]*b - cap
    for end in vertices:
        de = end[0]*a + end[1]*b - cap
        if (de > 0) != (ds > 0):
            f = ds / (ds-de)
            out.append((start[0]+f*(end[0]-start[0]), start[1]+f*(end[1]-start[1])))
        if de <= 0:
            out.append(end)
        start, ds = end, de
    return out

Q=2**72


def floor_sqrt(x):
    return F(isqrt((x.numerator*Q*Q)//x.denominator),Q)


def ceil_sqrt(x):
    h=floor_sqrt(x)
    return h if h*h==x else h+F(1,Q)


def array(a):
    return np.vectorize(exact,otypes=[object])(a)


def check(folder):
    ARCHIVE=folder.parents[4]
    mode="carry" if folder.name=="continuous" else "single"
    with np.load(folder/"context.npz") as f:
        S=array(f["S"]);mask=partition(f["S"])
        B,U,Y=[array(f[key][36:]) for key in ("b","ub","truth")]
    with np.load(folder/mode/"PIRCoherent.npz") as f:
        Z,V=array(f["pred"]),array(f["witness"])
    P=array(np.load(folder/mode/"PIRFull.npz")["pred"])
    old={key:array(np.load(folder/mode/(name+".npz"))["pred"]) for key,name in
         [("common","PIRCommon"),("two","PIRTwoGroup")]}
    n=len(S);T=len(Y); zero=lambda:[F(0) for _ in range(n)]
    dataset=folder.parts[-4]
    if dataset=="AEMO":
        system=[0];regions=list(range(1,n))
    else:
        with (ARCHIVE/"node_map.csv").open(encoding="utf-8-sig") as meta:
            nodes=[row for row in csv.DictReader(meta) if row["dataset"]==dataset]
        system=[int(row["node"]) for row in nodes if "system" in row["scopes"].split(";")]
        regions=[int(row["node"]) for row in nodes if "region" in row["scopes"].split(";")]
    selected=[[j for j in mask if S[i,j]] for i in range(n)]
    A=zero();AP=zero(); loss={j:zero() for j in old}
    supports=[np.flatnonzero(row) for row in S]
    max_ratio=F(0)
    for t,(b,z,y) in enumerate(zip(B,Z,Y)):
        L=[];H=[]
        for i in range(n):
            radius=ceil_sqrt(A[i]*(RR-1)/RR)
            L.append(max(F(0),b[i]-radius))
            H.append(b[i]+radius if A[i]<=RR*(RR-1)*b[i]**2 else ceil_sqrt(A[i]+RR*b[i]**2))
        # Allow bounded output roundoff on both sides of each group's affine map.
        pad=[(1+max(abs(bi),abs(zi),abs(hi)))/10**7 for bi,zi,hi in zip(b,z,H)]
        # The stored second direction closes on z-b. Against underlying
        # reconstruction the affine difference is (1-lambda2)*e_b + lambda2*e_z;
        # with the same stored intercept b it is lambda2*(e_z-e_b).
        # Their max absolute values occur at the endpoints, for all lambda1.
        for i,js in enumerate(supports):
            eb=sum((S[i,j]*U[t,j] for j in js),F(0))-b[i]
            ez=sum((S[i,j]*V[t,j] for j in js),F(0))-z[i]
            corner=max(abs(eb),abs(ez),abs(ez-eb))
            if corner>pad[i]:
                raise ArithmeticError((folder,t,i,corner,pad[i]))
            max_ratio=max(max_ratio,corner/pad[i])
        d=z-b; du=V[t]-U[t]
        d0=np.array([sum((du[j] if S[i,j]==1 else S[i,j]*du[j] for j in selected[i]),F(0))
                     for i in range(n)],dtype=object)
        d1=d-d0
        points=[(F(0),F(0)),(F(1),F(0)),(F(1),F(1)),(F(0),F(1))]
        cap=F(1)
        for i in range(n):
            for s,rhs in [(1,H[i]-b[i]+pad[i]),(-1,b[i]-L[i]+pad[i])]:
                h0,h1=s*d0[i],s*d1[i]
                if max(0,h0)+max(0,h1)>rhs:
                    points=clip(points,h0,h1,rhs)
                if s*d[i]>0: cap=min(cap,rhs/(s*d[i]))
        assert points and 0<=cap<=1
        for i in range(n):
            endpoints=[b[i]+d0[i]*x+d1[i]*v for x,v in points]
            common_end=b[i]+cap*d[i]
            for j,(lb,ub) in dict(common=(min(b[i],common_end),max(b[i],common_end)),
                                    two=(min(endpoints),max(endpoints))).items():
                lb-=pad[i];ub+=pad[i]
                assert lb<=old[j][t,i]<=ub
                loss[j][i]+=max(F(0),lb-y[i],y[i]-ub)**2
            A[i]+=RR*(b[i]-y[i])**2-max(F(0),L[i]-y[i],y[i]-H[i])**2
            assert exact_envelope(b[i],P[t,i])<=AP[i]
            AP[i]+=RR*(b[i]-y[i])**2-(P[t,i]-y[i])**2
            assert min(A[i],AP[i])>=0
    observed=[sum((P[:,i]-Y[:,i])**2,F(0)) for i in range(n)]
    rows=[]
    for j in old:
        fm=sum(ceil_sqrt(v/T) for v in observed)/n
        lb=sum(floor_sqrt(v/T) for v in loss[j])/n
        rows.append(dict(dataset=dataset,backbone=folder.parts[-3],run=folder.parts[-2],part=folder.name,rule=j,
                         full_upper=float(fm),relaxation_lower=float(lb),strict_mean=int(fm<lb),
                         strict_system=sum(observed[i]<loss[j][i] for i in system),
                         systems=len(system),regions=len(regions),
                         strict_regions=sum(observed[i]<loss[j][i] for i in regions),
                         strict_nodes=sum(x<y for x,y in zip(observed,loss[j])),nodes=n))
    # Round mean squared errors on a Q^-2 grid. This preserves the
    # original directed Q^-1 square roots exactly (integer-square thresholds).
    def down(value):
        return value.numerator//value.denominator
    def up(value):
        return -down(-value)
    ids=(dataset,folder.parts[-3],folder.parts[-2],folder.name)
    moments=[dict(zip(("dataset","backbone","run","part"),ids),
                  node=i,full=up(observed[i]*Q**2/T),
                  common=down(loss["common"][i]*Q**2/T),two=down(loss["two"][i]*Q**2/T))
             for i in range(n)]
    audit=dict(zip(("dataset","backbone","run","part"),ids),node_dates=T*n,
               max_error_to_pad_num=max_ratio.numerator,max_error_to_pad_den=max_ratio.denominator)
    return rows,moments,audit



def write_csv(dest,records):
    with dest.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=records[0]); writer.writeheader(); writer.writerows(records)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True,help="directory containing streams/ and node_map.csv")
    parser.add_argument("--out",type=Path,required=True,help="new output directory")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    folders=sorted(p for part in ("early","late","continuous")
                   for p in (args.archive.resolve()/"streams").glob("*/[A-Za-z]*/[0-9]*/"+part))
    if len(folders)!=189 or args.workers<1: raise ValueError("Expected 189 paths and at least one worker")
    args.out.mkdir(exist_ok=False,parents=True)
    rows=[]; moments=[]; audits=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(res,terms,audit) in enumerate(pool.map(check,folders),1):
            rows.extend(res);moments.extend(terms);audits.append(audit)
            if i%10==0:print(f"{i}/189 verified",flush=True)
    write_csv(args.out/"exact_grid.csv",rows)
    write_csv(args.out/"node_moments.csv",moments)
    write_csv(args.out/"corner_audit.csv",audits)
    print("Completed 189 full archived-input regenerations.",flush=True)

if __name__=="__main__":main()
