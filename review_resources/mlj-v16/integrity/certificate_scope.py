"""Audit stored endpoint closure, physical directions and all 189 path margins.
Usage: python certificate_scope.py --kit ../v15 --legacy ../legacy [--source ARCHIVE --build]
"""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np

def outward(q,up):
    f=float(q)
    if (F(f)<q if up else F(f)>q):f=np.nextafter(f,np.inf if up else -np.inf)
    return f

def pack(a):
    k=max(1,(max(abs(int(i)).bit_length() for i in a.flat)+8)//8)
    s=b''.join(int(i).to_bytes(k,'little',signed=True) for i in a.flat)
    return np.frombuffer(s,dtype=np.uint8).reshape(*a.shape,k)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def archive_delta(arc,row):
    x,y=[arc/row[key+'_archive'] for key in ('context','PIRCoherent')]
    assert digest(x)==row['context_sha256'] and digest(y)==row['PIRCoherent_sha256']
    with np.load(x) as a,np.load(y) as z:
        S,es=val.dyadic(a['S']); U,eu=val.dyadic(a['ub'][36:])
        V,ev=val.dyadic(z['witness']);B,eb=val.dyadic(a['b'][36:]);Z,ez=val.dyadic(z['pred'])
    e=max(es+eu,es+ev,eb,ez)
    def err(u,ex,q,eq):
        ans=-q*(1<<(e-eq))
        for i,w in enumerate(S):
            ix=np.flatnonzero(w)
            ans[:,i]+=(u[:,ix]@w[ix])*(1<<(e-es-ex))
        return ans
    f=err(U,eu,B,eb);g=err(V,ev,Z,ez)
    return f,g,e

def read_rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write_rows(p,rows):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    if not __debug__:raise RuntimeError('Run with assertions enabled (omit -O).')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit',required=True,type=Path);parser.add_argument('--legacy',required=True,type=Path)
    parser.add_argument('--source',type=Path);parser.add_argument('--build',action='store_true')
    args=parser.parse_args();sys.path.insert(0,str(args.kit.resolve()))
    global val
    import validate as val
    from verify import unpack,score
    root=Path(__file__).resolve().parent;side=root/'differences';side.mkdir(exist_ok=True)
    affected={(r['dataset'],r['backbone'],r['run']) for r in read_rows(args.legacy/'AFFECTED_STREAMS.csv')};assert len(affected)==8
    out=[]; combos={}
    for row in read_rows(args.kit/'inputs.csv'):
        name=row['case'];casefile=side/(name+'.npz')
        if args.source:
            eb,ez,exp=archive_delta(args.source,row)
            if args.build:np.savez_compressed(casefile,eb=pack(eb),ez=pack(ez),exponent=exp)
        with np.load(casefile) as h:
            fb,fz,ec=unpack(h,'eb'),unpack(h,'ez'),int(h['exponent'])
        if args.source:
            assert ec==exp and np.array_equal(fb,eb) and np.array_equal(fz,ez)
        path=args.kit/row['download'];assert digest(path)==row['sha256']
        with np.load(path) as rec:
            e=int(rec['exponent']);b,r,d=[unpack(rec,k) for k in ('cap','residual','direction')]
            m=SimpleNamespace(T=r.shape[0],n=r.shape[1],B=np.ldexp(b.astype(float),-e),r=rec['r_float'],D=rec['D_float'],online=float(row['online']))
            def bounds(a,e):
                x=np.ldexp(a.astype(float),-e);return val.dn(x),val.up(x)
            def lower(D,ex):
                meta=[(int(i),int(t),int(j),float(s)) for i,t,j,s in rec['meta']]
                return F(float(val.dual(m,(None,None,None,None,bounds(D,ex),bounds(r,e)),meta,rec['anchors'],rec['dual'])))
            L=lower(d,e)
            U,_=val.primal(m,(b,b-r,d,e,None,None),rec['path'],replay=True)
            full=score(unpack(rec,'full_error'),int(rec['full_exponent']))
            E=max(ec,e);dnew=d*(1<<(E-e));dnew[:,:,1]+=(fz-fb)*(1<<(E-ec))
            Lphys=lower(dnew,E)
            # This second lower bound substitutes the literal group directions into
            # every historical square, release envelope, and terminal objective.
            # The scalar intercept / contract reference and outcomes stay fixed.
            # The purely reconstructed forecast differs from the anchored literal
            # path by e_b; at an anchor fallback choose x=0 and displacement 0.
            # At the first release the exact contract forces p=b, so both maps
            # use x_1=0 without restricting the set of achievable scores.
            delta=np.abs(fb)*(1<<(E-ec));delta[0]=0
            mag=np.abs(dnew).sum(2);res=np.abs(r)*(1<<(E-e))
            past=delta*(2*(res+mag)+delta)
            prefix=np.zeros_like(past)
            prefix[1:]=np.cumsum(past[:-1],axis=0)
            caps=401*prefix+2*40401*delta*(mag+delta)
            denom=401*(1<<(2*E))
            penalty=F(int(delta.max()),1<<E)
            for (_,it,node,scale),mu in zip(rec['meta'],rec['dual']):
                if it>=0 and mu>0:
                    lam=F(float(mu*m.online/float(scale)))
                    penalty+=lam*F(int(caps[int(it),int(node)]),denom)
            Ltransfer=Lphys-penalty
            key=tuple(row[k] for k in ('dataset','backbone','run'))
            good=int(key not in affected);combos[key]=good
            U=F(float(U)); strict=full[1]<L; assert strict==(row['full_strict']=='True')
            assert strict or U<full[0]
            delta=lambda a:outward(F(int(np.abs(a).max()),2**ec),True)
            out.append(dict(case=name,dataset=key[0],backbone=key[1],run=key[2],part=row['part'],protocol=row['protocol'],identified_subset=good,
                   stored_lower=outward(L,False),stored_upper=outward(U,True),
                   full_lower=outward(full[0],False),full_upper=outward(full[1],True),
                   optimum_width_upper=outward(U-L,True),full_score_bracket_width_upper=outward(full[1]-full[0],True),
                   signed_separation_lower=outward(L-full[1],False),
                   reverse_witness_margin_lower=outward(full[0]-U,False),
                   literal_direction_lower=outward(Lphys,False),
                   literal_separation_lower=outward(Lphys-full[1],False),
                   physical_transfer_penalty_upper=outward(penalty,True),physical_release_lower=outward(Ltransfer,False),
                   physical_separation_lower=outward(Ltransfer-full[1],False),
                   reference_reconstruction_max=delta(fb),target_reconstruction_max=delta(fz),
                   direction_difference_max=delta(fz-fb),sidecar_sha256=digest(casefile)))
    assert len(out)==189 and len(combos)==63 and sum(combos.values())==55
    groups=[dict(dataset=k[0],backbone=k[1],run=k[2],include_primary=v) for k,v in sorted(combos.items())]
    if args.build:
        write_rows(root/'certificate_margins.csv',out);write_rows(root/'identified_combinations.csv',groups)
    else:
        expected=read_rows(root/'certificate_margins.csv')
        assert len(expected)==len(out)
        for saved,replay in zip(expected,out):
            for k,value in replay.items():
                if isinstance(value,float):
                    # BLAS may propose slightly different support directions.
                    # Tolerance here applies only to agreement with printed CSV.
                    pairs=(float(saved[k]),value)
                    assert np.isclose(*pairs,rtol=1e-10,atol=1e-11)
                    if k in ('signed_separation_lower','reverse_witness_margin_lower','literal_separation_lower','physical_separation_lower'):
                        assert (pairs[0]>0)==(pairs[1]>0)
                else:assert saved[k]==str(value)

        assert read_rows(root/'identified_combinations.csv')==[{k:str(v) for k,v in r.items()} for r in groups]
    for typ in ('window','continuous'):
        for filt in (False,True):
            a=[r for r in out if r['protocol']==typ and (not filt or r['identified_subset'])]
            print(typ,('identified' if filt else 'all'),len(a),sum(r['signed_separation_lower']>0 for r in a),sum(r['literal_separation_lower']>0 for r in a),sum(r['physical_separation_lower']>0 for r in a),sum(r['reverse_witness_margin_lower']>0 for r in a))
    for k in ('direction_difference_max','reference_reconstruction_max','target_reconstruction_max','full_score_bracket_width_upper'):
        print(k,max(r[k] for r in out))
    print('smallest strict and reverse margins:',min(r['signed_separation_lower'] for r in out if r['signed_separation_lower']>0),min(r['reverse_witness_margin_lower'] for r in out if r['reverse_witness_margin_lower']>0))
    print('PASS 189 incoming certificates, literal-group lower cuts, disjoint scope and numerical margins')
if __name__=='__main__':main()

