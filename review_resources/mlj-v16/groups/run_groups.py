"""Evaluate the publication rules defined in the preregistered historical-grid protocol."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, csv, json, hashlib, os, time, warnings
import numpy as np
import scipy.sparse as sp
import clarabel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEFAULT_ARCHIVE=ROOT/'output/cpor_unified_protocol_duet_20261108'
DEFAULT_INDEX=ROOT/'output/cpor_mlj_v15_20261115/kit/inputs.csv'
RULES=('Independent','Adaptive2','Adaptive4','Adaptive8','Fixed4','Fixed8')
R=40401/40000
ETA=401/40000

def split_fixed(S,k):
    # Reuse the original first bipartition, then split sorted indices into halves.
    m=S.shape[1]; selected=set(); size=(m+1)//2
    for inds in sorted([set(np.flatnonzero(row>0)) for row in S],key=len,reverse=True):
        if selected.isdisjoint(inds) and len(selected)+len(inds)<=size:selected.update(inds)
    for j in range(m):
        if len(selected)==size:break
        selected.add(j)
    sets=[np.array(sorted(selected),int),np.array(sorted(set(range(m))-selected),int)]
    while len(sets)<k:
        sets=[part for g in sets for part in np.array_split(g,2)]
    return [g for g in sets if len(g)]

def adaptive_partition(D,lo,hi,k):
    # Fraction affordable if that latent correction were made alone.
    widths=np.where(D>0,hi[:,None],-lo[:,None])
    fraction=np.divide(widths,np.abs(D),out=np.full(D.shape,np.inf),where=D!=0)
    score=np.minimum(1.,fraction.min(axis=0))
    return np.array_split(np.argsort(score,kind='stable'),min(k,D.shape[1]))

def exact(x,scale):
    res=[]
    for z in x:
        a,b=float(z).as_integer_ratio();k=scale-b.bit_length()+1
        if a and k<0: raise ArithmeticError('insufficient dyadic exponent')
        res.append(a<<max(0,k))
    return np.array(res,dtype=object)

class Ledger:
    # One integer unit for observations and predictions, and its square for credit.
    def __init__(self,B,Y,Z):
        vals=np.concatenate([np.abs(a[a!=0]) for a in (B,Y,Z)])
        exp=min(0,int(np.frexp(vals.min())[1]))
        self.k=max(110,65-exp)
        self.B=[exact(b,self.k) for b in B]
        self.Y=[exact(y,self.k) for y in Y]
        self.Z=[exact(z,self.k) for z in Z]
        self.Rerr=[40401*(b-y)**2 for b,y in zip(self.B,self.Y)]
        self.A=np.zeros(B.shape[1],object)
    def toint(self,p,t):
        try:return exact(p,self.k)
        except ArithmeticError:
            k=max(self.k,max(float(x).as_integer_ratio()[1].bit_length()-1 for x in p))
            fac=1<<(k-self.k);self.k=k
            for x in self.B+self.Y+self.Z:x*=fac
            self.A*=fac*fac
            for r in self.Rerr:r*=fac*fac
            return exact(p,k)
    def allowance(self):
        # Scale arbitrary-size integers without first casting a huge integer to float.
        factor=40000*(1<<(2*self.k))
        return np.array([x/factor for x in self.A])
    def flags(self,t,pi):
        b=self.B[t];d=pi-b
        return (pi<0)|np.where(40000*d>401*b,40000*pi*pi-40401*b*b>self.A,40000*40401*d*d>401*self.A)
    def step(self,t,pi):
        assert not np.any(self.flags(t,pi))
        self.A+=self.Rerr[t]-40000*(pi-self.Y[t])**2
        assert min(self.A)>=0

def qp(D,lo,hi,std,tol):
    # Scale the weighted distance by a common factor. Bounds use per-row equilibration.
    n,k=D.shape
    M=D/std[:,None]
    target=M.sum(1)
    weight=max(np.max(abs(M)),np.max(abs(target)),1e-12)
    M=M/weight; target=target/weight
    cs=np.max(abs(D),axis=1);cs[cs==0]=1.
    plus=np.maximum(D,0.).sum(1)>hi
    minus=np.minimum(D,0.).sum(1)<lo
    G=sp.csc_matrix(np.vstack([D[plus]/cs[plus,None],-D[minus]/cs[minus,None]]))
    cap=np.r_[hi[plus]/cs[plus],-lo[minus]/cs[minus]]
    C=sp.vstack([G,sp.eye(k),-sp.eye(k)],format='csc')
    rhs=np.r_[cap,np.ones(k),np.zeros(k)]
    if k<=16:
        P=sp.triu(sp.csc_matrix(M.T@M),format='csc')
        lin=-M.T@target; mat=C;bvec=rhs
        cones=[clarabel.NonnegativeConeT(len(rhs))]
    else:
        # Sparse residual formulation avoids a dense aggregate-induced Hessian.
        P=sp.diags(np.r_[np.zeros(k),np.ones(n)],format='csc');lin=np.r_[np.zeros(k),-target]
        mat=sp.vstack([sp.hstack([sp.csc_matrix(M),-sp.eye(n)],format='csc'),sp.hstack([C,sp.csc_matrix((len(rhs),n))])],format='csc')
        bvec=np.r_[np.zeros(n),rhs];cones=[clarabel.ZeroConeT(n),clarabel.NonnegativeConeT(len(rhs))]
    opt=clarabel.DefaultSettings();opt.verbose=False;opt.max_threads=1;opt.max_iter=250
    opt.tol_gap_abs=opt.tol_gap_rel=opt.tol_feas=tol
    res=clarabel.DefaultSolver(P,lin,mat,bvec,cones,opt).solve()
    x=np.clip(np.asarray(res.x[:k]),0.,1.)
    if not np.all(np.isfinite(x)):raise ArithmeticError(str(res.status))
    # Independent primal--dual gap: any signed precision of nonnegative
    # multipliers yields a valid mathematical affine lower bound for this box QP.
    mu=np.maximum(0.,np.asarray(res.z)[0 if k<=16 else n:][:len(cap)])
    residual=M@x-target
    grad=M.T@residual+np.asarray(G.T@mu).ravel()
    gap=float(grad@x-np.minimum(grad,0).sum()+mu@(cap-G@x))
    obj=.5*float(residual@residual)
    return x,max(0.,gap)/(1+obj),str(res.status)

def issue(rule,S,v,b,ub,z,u,t,led,fixed):
    if not any(led.A):return b,led.B[t],0.,0.,'zero'
    if not np.any(led.flags(t,led.Z[t])):return z,led.Z[t],0.,0.,'target'
    credit=led.allowance();width=np.sqrt(credit*ETA/R)
    lo=np.maximum(-b,-width)
    hi=np.where(credit<=R*ETA*b*b,width,(credit+ETA*b*b)/(np.sqrt(credit+R*b*b)+b+1e-300))
    du=u-ub;single=S*du
    if rule=='Independent':
        groups=None;D=single.copy()
    else:
        k=int(rule[-1])
        groups=adaptive_partition(single,lo,hi,k) if rule.startswith('Adaptive') else fixed[k]
        D=np.column_stack([single[:,g].sum(axis=1) for g in groups])
    D[:,-1]=(z-b)-D[:,:-1].sum(axis=1)
    for it,tol in enumerate((1e-9,1e-11,1e-12)):
        x,gap,status=qp(D,lo,hi,np.sqrt(v),tol)
        p0=b+D@x
        if groups is None:latent=x
        else:
            latent=np.empty(len(u))
            for g,val in zip(groups,x):latent[g]=val
        witness=(1-latent)*ub+latent*u
        p=np.minimum(np.maximum(p0,np.maximum(0.,b+lo)),b+hi)
        corrections=0
        for _ in range(12):
            pi=led.toint(p,t);bad=led.flags(t,pi)
            if not np.any(bad):break
            p[bad]=np.nextafter(p[bad],b[bad]);corrections+=1
        else:continue
        # Test the actual release under the original nodewise consistency
        # tolerance. Also require its weighted objective to match the optimum.
        if not np.allclose(p,S@witness,rtol=1e-8,atol=2e-6):continue
        if np.any(witness<0):continue
        weight=max(np.abs(D/np.sqrt(v)[:,None]).max(),np.abs(D.sum(1)/np.sqrt(v)).max(),1e-12)
        obj=lambda a: .5*np.sum(((a-z)/np.sqrt(v)/weight)**2)
        gap=(gap*(1+obj(p0))+obj(p)-obj(p0))/(1+obj(p))
        if gap>2e-6 or not np.isfinite(gap):continue
        movement=float(np.max(abs(p-p0)))
        return p,pi,max(0.,gap),movement,('retry_' if it else '')+('guard' if movement or corrections else 'accepted')
    # A feasible reference cannot certify an arbitrary solver failure as an optimum.
    raise ArithmeticError((rule,t,status,gap,'guard or optimality test failed'))

def check_hash(p,h):
    if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError(p)
def run_task(job):
    row,rule,archive,out=job
    path=Path(out)/'runs'/row['case'];path.mkdir(parents=True,exist_ok=True)
    dest=path/(rule+'.npz')
    if dest.exists():return
    files={k:Path(archive)/row[k+'_archive'] for k in ('context','PIRCoherent','PIRTwoGroup','PIRFull')}
    for k,p in files.items():check_hash(p,row[k+'_sha256'])
    with np.load(files['context']) as x,np.load(files['PIRCoherent']) as z:
        S=x['S'].astype(float);Y=x['truth'][36:].astype(float);B=x['b'][36:];UB=x['ub'][36:];v=x['v'];Z=z['pred'];U=z['witness']
    fixed={k:split_fixed(S,k) for k in (4,8)}
    led=Ledger(B,Y,Z);issued=[];times=[];gaps=[];moves=[];decisions=[]
    for t,(b,ub,z,u) in enumerate(zip(B,UB,Z,U)):
        start=time.perf_counter_ns()
        p,pi,gap,move,how=issue(rule,S,v,b,ub,z,u,t,led,fixed)
        times.append((time.perf_counter_ns()-start)/1e6)
        led.step(t,pi);issued.append(p);gaps.append(gap);moves.append(move);decisions.append(how)
    tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,pred=issued,ms=times,gap=gaps,move=moves,decision=decisions)
    tmp.replace(dest)

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE)
    p.add_argument('--index',type=Path,default=DEFAULT_INDEX);p.add_argument('--workers',type=int,default=10)
    p.add_argument('--rules',nargs='*',default=RULES);p.add_argument('--case',default='')
    p.add_argument('--out',type=Path,default=HERE);a=p.parse_args()
    rows=list(csv.DictReader(a.index.open(encoding='utf-8-sig')))
    rows=[r for r in rows if a.case in r['case']]
    jobs=[(r,arm,str(a.archive),str(a.out)) for r in rows for arm in a.rules]
    # Start the largest cases first; run each process with one BLAS thread.
    from concurrent.futures import ThreadPoolExecutor
    jobs.sort(key=lambda j:-(int(j[0]['nodes'])*int(j[0]['issues'])))
    ok=bad=0
    with ProcessPoolExecutor(a.workers) as pool:
        f={pool.submit(run_task,j):j for j in jobs}
        for future in as_completed(f):
            j=f[future]
            try:future.result();ok+=1
            except Exception as e:print('FAIL',j[0]['case'],j[1],repr(e),flush=True);bad+=1
            if (ok+bad)%50==0:print('completed',ok,'failed',bad,'of',len(jobs),flush=True)
    if bad:raise SystemExit(f'{bad} incomplete')
    print('PASS',ok,flush=True)
if __name__=='__main__':main()

