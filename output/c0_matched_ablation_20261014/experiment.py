from pathlib import Path
import sys,json,time,hashlib,copy,importlib.util
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;O=HERE.parent;V=O/'paired_root_actual_credit_20261013/numerical_v2'
sys.path.insert(0,str(V))
import conic_learning as root_kernel
spec=importlib.util.spec_from_file_location('frozen_c0_run',V/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import square_learning as square
CELLS=base.CELLS
ARMS=['SquareFull','RootDirectional','SquareDirectional']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def contexts(cell):
    src=base.original.paths(cell);x=dict(np.load(src['endpoint']))
    D,experts=base.make_contexts(x,np.load(src['bank'])['bank'],dict(np.load(src['target'])),base.original)
    rel='/'.join(cell)+'/42';snap=dict(np.load(V/'results'/rel/'C0_snapshots.npz'))
    records=[{k:np.array(a[t],copy=True) for k,a in snap.items()} for t in range(36)]
    # Labels mature strictly before the first scoring decision.
    assert all(r['label_event']<73 for r in records)
    total=np.sum((snap['published'][:36]-snap['y'][:36])**2,axis=0)
    denominator=float(np.mean(total));numerator=float(np.mean(np.sqrt(total)))
    kappa=numerator/denominator if denominator>0 else 1/float(np.sqrt(np.mean(x['v'])))
    return x,D,experts,records,kappa

def one_directional(qp,S,b,ub,target,A):
    lo,hi=base.original.safe_bounds(b,np.array([float(a) for a in A]))
    lo=np.maximum(lo,np.minimum(b,target));hi=np.minimum(hi,np.maximum(b,target))
    if not np.any(hi>lo):return b.copy(),ub.copy()
    p,u,info=qp.solve(b,target,ub,lo,hi);initial=p.copy();pp=np.clip(p,lo,hi)
    for _ in range(16):
        if base.original.certified(b,pp,A,base.original.RR,base.original.K,True):break
        for j in range(len(b)):
            if not base.original.certified(b[j:j+1],pp[j:j+1],A[j:j+1],base.original.RR,base.original.K,True):pp[j]=np.nextafter(pp[j],b[j])
    if np.allclose(pp,S@u,rtol=1e-8,atol=2e-6) and base.original.certified(b,pp,A,base.original.RR,base.original.K,True):p=pp
    else:p,u,_=base.original.guard(b,p,ub,u,S,lo,hi,A,base.original.RR,base.original.K)
    obj=float(.5*np.sum((p-target)**2/qp.s**2));gap=info['gap']+obj-info['objective']
    assert gap<=2e-6*(1+obj),(gap,obj)
    assert np.all(p>=np.minimum(b,target)-2e-6) and np.all(p<=np.maximum(b,target)+2e-6)
    return p,u

def make_bank(qp,S,b,ub,D,experts,A,directional):
    if not directional:return base.make_bank(qp,S,b,ub,D,experts,A,False)
    P=[];U=[]
    for target in [D,*experts.T]:
        p,u=one_directional(qp,S,b,ub,target,A)
        p=b+(1-base.INWARD)*(p-b);u=ub+(1-base.INWARD)*(u-ub)
        lo,hi=base.original.safe_bounds(b,np.array([float(a) for a in A]))
        p,u,_=base.original.guard(b,p,ub,u,S,lo,hi,A,base.original.RR,base.original.K)
        assert base.original.certified(b,p,A,base.original.RR,base.original.K,True)
        P.append(p);U.append(u)
    P.append(b.copy());U.append(ub.copy())
    return np.array(P).T,np.array(U).T

class Actor(base.Actor):
    def __init__(self,S,v,arm,records,kappa):
        super().__init__(S,v,arm);self.records=copy.deepcopy(records);self.kappa=kappa
    def issue(self,index,b,ub,D,experts,previous_label=None):
        if self.pending is not None:self.close(previous_label)
        P,U=make_bank(self.qp,self.S,b,ub,D,experts,self.A,self.arm.endswith('Directional'))
        assert all(r['label_event']<2*index+1 for r in self.records)
        failed=False
        try:
            if self.arm.startswith('Root'):theta,info,_=root_kernel.fit(self.records,self.S,'C0')
            else:theta,info=square.fit(self.records,self.kappa)
        except Exception as err:
            theta=np.eye(11)[-1];info=dict(passed=False,error=repr(err),fallback='registered_anchor');failed=True
        initial=b+(P-b[:,None])@theta;u=ub+(U-ub[:,None])@theta
        lo,hi=base.original.safe_bounds(b,np.array([float(a) for a in self.A]))
        p,u,rho=base.original.guard(b,initial,ub,u,self.S,lo,hi,self.A,base.original.RR,base.original.K)
        applied=rho*theta+(1-rho)*np.eye(11)[-1]
        risk=base.original.excess_bound(b,p,base.original.RR,base.original.K)
        assert min(a-r for a,r in zip(self.A,risk))>=0
        self.pending=dict(index=index,epoch=1,scored=True,issue_event=2*index+1,P=P,U=U,b=b.copy(),ub=ub.copy(),
            E=self.E.copy(),A=np.array([float(a) for a in self.A]),A_num=np.array([str(a.numerator) for a in self.A]),
            A_den=np.array([str(a.denominator) for a in self.A]),theta=applied,published=p.copy())
        return dict(pred=p,witness=u,theta=applied,fit_theta=theta,actual_credit=self.pending['A'],energy_before=self.E.copy(),
                    lo=lo,hi=hi,reserved=np.array([float(a) for a in risk]),available_after=np.array([float(a-r) for a,r in zip(self.A,risk)]),
                    contraction=rho,rounding_move=float(abs(p-initial).max()),solver_failed=failed,fit=info)

def registration():
    admission=json.loads((HERE/'ADMISSION.json').read_text());assert admission['passed']
    old=json.loads((V/'PREREGISTRATION_PAIRED_ROOT.json').read_text())
    for p,h in old['source_sha256'].items():assert sha(p)==h,p
    for p,h in admission['source_sha256'].items():assert sha(p)==h,p
    path=HERE/'REGISTRATION.json'
    if path.exists():
        existing=json.loads(path.read_text())
        for p,h in existing['source_sha256'].items():assert sha(p)==h,p
        assert existing['admission_sha256']==sha(HERE/'ADMISSION.json')
        assert existing['original_registration_sha256']==sha(V/'PREREGISTRATION_PAIRED_ROOT.json')
        return path
    for p,h in old['input_sha256'].items():assert sha(p)==h,p
    source={str(p.resolve()):sha(p) for p in [Path(__file__),HERE/'square_learning.py',HERE/'PROTOCOL.md',HERE/'admission.py',HERE/'run_all.py']}
    inputs={}
    for cell in CELLS:
        for p in base.original.paths(cell).values():inputs[str(p.resolve())]=sha(p)
        rel='/'.join(cell)+'/42'
        for name in ['C0.npz','C0.json','C0_snapshots.npz']:
            p=V/'results'/rel/name;inputs[str(p.resolve())]=sha(p)
    reg=dict(cells=CELLS,arms=ARMS,reused_arm='RootFull=C0',source_sha256=source,input_sha256=inputs,
             original_registration_sha256=sha(V/'PREREGISTRATION_PAIRED_ROOT.json'),admission_sha256=sha(HERE/'ADMISSION.json'),
             calibration='common original C0 first36 snapshots; first score resets A,E to0',
             protocol_sha256=sha(HERE/'PROTOCOL.md'),primary='original10 dataset/backbone groups; average windows; mean node RMSE',
             source_change_policy='fail rather than mix modified sources',development_only=True)
    reg=json.loads(json.dumps(reg));p=HERE/'REGISTRATION.json'
    if p.exists():assert json.loads(p.read_text())==reg
    else:p.write_text(json.dumps(reg,indent=2),encoding='utf-8')
    return p

def run_cell(ci,arm):
    reg=registration();cell=CELLS[ci];rel='/'.join(cell)+'/42';dest=HERE/'results'/rel;dest.mkdir(parents=True,exist_ok=True)
    expected=json.loads(reg.read_text())['input_sha256']
    for p in [*base.original.paths(cell).values(),*[V/'results'/rel/z for z in ['C0.npz','C0.json','C0_snapshots.npz']]]:
        assert sha(p)==expected[str(p.resolve())],p
    path=dest/(arm+'.npz')
    if path.with_suffix('.json').exists():
        info=json.loads(path.with_suffix('.json').read_text());assert info['registration_sha256']==sha(reg) and info['output_sha256']==sha(path);return
    x,D,experts,records,kappa=contexts(cell);a=Actor(x['S'],x['v'],arm,records,kappa)
    rows=[];logs=[];started=time.perf_counter()
    for t in range(36,108):
        row=a.issue(t,x['b'][t],x['ub'][t],D[t],experts[t],None if t==36 else x['truth'][t-1])
        logs.append(dict(t=t,fit=row.pop('fit')));rows.append(row)
        if (t-36)%12==0:
            np.savez_compressed(dest/(arm+'_partial.npz'),**{k:np.array([r[k] for r in rows]) for k in row})
            (dest/(arm+'_partial.json')).write_text(json.dumps(logs),encoding='utf-8')
            print('PROGRESS',ci,arm,t-35,'/72',round(time.perf_counter()-started,1),flush=True)
    out={k:np.array([r[k] for r in rows]) for k in rows[0]};checks=base.original.audit(x,out,1)
    a.close(x['truth'][-1]);a.pending=None
    np.savez_compressed(path,**out);stream=dest/(arm+'_snapshots.npz')
    np.savez_compressed(stream,**{k:np.array([r[k] for r in a.records]) for k in a.records[0]})
    info=dict(passed=True,cell=cell,arm=arm,checks=checks,kappa=kappa,seconds=time.perf_counter()-started,
              failed_publications=int(out['solver_failed'].sum()),registration_sha256=sha(reg),output_sha256=sha(path),snapshot_sha256=sha(stream),
              endpoint_sha256=sha(base.original.paths(cell)['endpoint']),logs=logs)
    path.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('DONE',ci,arm,round(info['seconds'],1),'fallback',info['failed_publications'],flush=True)

if __name__=='__main__':
    with threadpool_limits(1):run_cell(int(sys.argv[1]),sys.argv[2])
