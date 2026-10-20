"""Frozen 18-cell causal development experiment; no baseline retraining."""
from pathlib import Path
import sys,json,hashlib,datetime,time,importlib.util,copy
HERE=Path(__file__).resolve().parent;O=HERE.parent
sys.path.insert(0,str(O/'rr_recap_relative_dominance_20261003'))
from method import SafeProjection,guard,safe_bounds,certified,excess_bound,advance,RR,K,Fraction
from frozen_core import project
from test_controller import audit
spec=importlib.util.spec_from_file_location('covariance_target_original',O/'rr_recap_error_dependence_20261003/method.py')
cov=importlib.util.module_from_spec(spec);spec.loader.exec_module(cov)
import numpy as np
from threadpoolctl import threadpool_limits
from packet_learner import fit,ALPHA,INSET,FIT_REL_GAP

B=O/'rr_recap_evidence_extension_20261002';M=B/'major_revision_20261002'
D=O/'rr_recap_error_dependence_20261003';C=O/'rr_recap_relative_dominance_20261003'
CELLS=[(ds,bb,w) for ds in ['RH','ECommerce','Traffic','Wiki2'] for bb in ['PatchTST','DLinear'] for w in ['early','late']]+[('Labour',bb,'late') for bb in ['PatchTST','DLinear']]
ARMS=['CoupledRiskFTRL','CurrentLossFTRL']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def paths(cell):
    ds,bb,w=cell;rel=f'{ds}/{bb}/{w}/42'
    return dict(endpoint=(M if ds=='Labour' else B)/f'results/{rel}/endpoints.npz',bank=D/f'results/{rel}/bank.npz',target=D/f'results/{rel}/CovFull.npz')

def one_action(qp,S,b,ub,target,A):
    lo,hi=safe_bounds(b,np.array([float(a) for a in A]))
    if not np.any(hi>lo):return b.copy(),ub.copy(),dict(gap=0.,objective=float(.5*np.sum((b-target)**2/qp.s**2)),contraction=1.,rounding_move=0.,lo=lo,hi=hi)
    p,u,info=qp.solve(b,target,ub,lo,hi);initial=p.copy();pp=np.clip(p,lo,hi)
    for _ in range(16):
        if certified(b,pp,A,RR,K,True):break
        for j in range(len(b)):
            if not certified(b[j:j+1],pp[j:j+1],A[j:j+1],RR,K,True):pp[j]=np.nextafter(pp[j],b[j])
    if np.allclose(pp,S@u,rtol=1e-8,atol=2e-6) and certified(b,pp,A,RR,K,True):p=pp;c=1.
    else:p,u,c=guard(b,p,ub,u,S,lo,hi,A,RR,K)
    # Update the original solver certificate for the repaired physical point.
    obj=float(.5*np.sum((p-target)**2/qp.s**2))
    gap=info['gap']+obj-info['objective']
    assert gap<=2e-6*(1+obj),(gap,obj)
    return p,u,{**info,'gap':max(0.,float(gap)),'objective':obj,'contraction':c,'rounding_move':float(abs(p-initial).max()),'lo':lo,'hi':hi}

class Actor:
    """issue accepts current forecasts and only the previous returned label."""
    def __init__(self,S,v,arm):
        self.S=S;self.v=v;self.arm=arm;self.qp=SafeProjection(S,v)
        self.A=[Fraction(0) for _ in v];self.pending=None;self.records=[]
    def issue(self,b,ub,D,expert_targets,previous_label=None,reset=False,calibration=False):
        if self.pending is not None:
            assert previous_label is not None
            old=self.pending
            self.A=advance(self.A,old['b'],old['published'],previous_label)
            fixed=np.nan
            if not reset and np.array_equal(old['P'],np.tile(old['P'][:,0,None],(1,10))):
                assert np.array_equal(old['published'],old['P'][:,0])
                q,_,_=one_action(self.qp,self.S,b,ub,D,self.A)
                fixed=float(np.mean((q-D)**2/self.v))
            self.records.append(dict(P=old['P'].copy(),A=old['A'].copy(),b=old['b'].copy(),y=np.array(previous_label,copy=True),beta=b.copy(),D=D.copy(),beta_u=ub.copy(),reset=bool(reset),issued_theta=old['theta'].copy(),issued_p=old['published'].copy(),scored=old['scored'],fixed_future_value=fixed))
        if reset:self.A=[Fraction(0) for _ in self.v]
        if calibration:theta=np.eye(10)[0];info=dict(calibration=True)
        else:theta,info=fit(self.records,self.S,self.v,future=self.arm=='CoupledRiskFTRL')
        credit=np.array([float(a) for a in self.A]);P=[];U=[];stats=[]
        for target in [D,*expert_targets.T]:
            p,u,mi=one_action(self.qp,self.S,b,ub,target,self.A)
            P.append(p);U.append(u);stats.append({k:a for k,a in mi.items() if k not in ['lo','hi']})
        P=np.array(P).T;U=np.array(U).T
        initial=P@theta;u=U@theta;lo,hi=safe_bounds(b,credit)
        if np.array_equal(P,np.tile(b[:,None],(1,10))):p=b.copy();u=ub.copy();contraction=1.
        else:p,u,contraction=guard(b,initial,ub,u,self.S,lo,hi,self.A,RR,K)
        risk=excess_bound(b,p,RR,K);after=[a-r for a,r in zip(self.A,risk)]
        assert min(after)>=0
        self.pending=dict(P=P,A=credit,b=b.copy(),published=p.copy(),theta=theta.copy(),scored=not calibration)
        return dict(pred=p,witness=u,theta=theta,available_after=np.array([float(a) for a in after]),reserved=np.array([float(a) for a in risk]),lo=lo,hi=hi,contraction=contraction,rounding_move=float(abs(p-initial).max()),fit=info,candidates=stats)

def make_contexts(x,bank,saved):
    S,v=x['S'],x['v'];targets=[]
    for t in range(36):
        w=np.full_like(bank[t],1/9) if t==0 else cov.solve_weights(cov.moment(bank[:t],x['truth'][:t],v,t))[0]
        targets.append(np.sum(bank[t]*w,axis=1))
    cal,_=project(S,np.array(targets),v)
    for t in [36]:
        w=cov.solve_weights(cov.moment(bank[:t],x['truth'][:t],v,t))[0]
        np.testing.assert_allclose(w,saved['weights'][t-36],rtol=1e-10,atol=1e-10)
    # Bank outputs are already causal frozen inputs. Reconcile each known
    # forecast vector before the policy, with the original weighted NNLS.
    e=np.moveaxis(bank,2,1).reshape(-1,len(v));z,_=project(S,e,v)
    expert=np.moveaxis(z.reshape(len(bank),9,len(v)),1,2)
    return np.concatenate([cal,saved['target']]),expert

def freeze():
    adm=json.loads((HERE/'ADMISSION.json').read_text());assert adm['passed']
    for p,h in adm['source_sha256'].items():assert sha(HERE/p)==h
    for p,h in json.loads((C/'freeze_v2.json').read_text())['files'].items():assert sha(p)==h
    source={str(p.resolve()):sha(p) for p in [Path(__file__),HERE/'packet_learner.py',HERE/'DESIGN_AND_THEORY.md']}
    for mod in list(sys.modules.values()):
        pp=getattr(mod,'__file__',None)
        if pp and Path(pp).suffix=='.py' and Path(pp).resolve().is_relative_to(O.resolve()):source[str(Path(pp).resolve())]=sha(pp)
    inputs={str(p):sha(p) for cell in CELLS for p in paths(cell).values()}
    reg=dict(cells=CELLS,arms=ARMS,source_sha256=source,input_sha256=inputs,admission_sha256=sha(HERE/'ADMISSION.json'),seed=42,
        parameters=dict(alpha=ALPHA,inset=INSET,fit_relative_gap=FIT_REL_GAP,epochs='exact global convex FTRL at each date',history='all mature records; calibration36 retained',initial_credit=0,feedback_delay=1,candidate_count=10),
        calendar='36 calibration on shared Cov publication path; reset own A to zero before72 score dates; label t-1 closes packet with known forecasts at t before Y_t',
        primary='10 dataset/backbone groups; windows averaged mean-node RMSE; wins >1e-9 percent; majority >5/10; MAE and nMSE secondary',
        baselines='Reuse original ProCAST72Warm, CovFullFix and matched ProtectedProCAST72; do not rerun external fit',
        limits=['Development data previously inspected; not untouched confirmation.','Next-context D is a causal Cov forecast, not proved a conditional expectation.','Future optimal value is a surrogate; next policy learning drift is charged explicitly.','Proof distinguishes specified prior losses, not global firstness.'],
        numerical='Fixed tolerances; failures stop and retain partial output; no silent model fallback or per-cell retuning')
    path=HERE/'REGISTRATION_V13.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:path.write_text(json.dumps(reg,indent=2),encoding='utf-8')
    return path

def main():
    reg=freeze();selected=sys.argv[1:] or ['all']
    for ci,cell in enumerate(CELLS):
        if 'all' not in selected and str(ci) not in selected:continue
        rel='/'.join(cell)+'/42';src=paths(cell);dest=HERE/f'results/{rel}';dest.mkdir(parents=True,exist_ok=True)
        needed=[arm for arm in ARMS if not (dest/(arm+'.json')).exists()]
        if not needed:continue
        begin=time.perf_counter();x=dict(np.load(src['endpoint']));bank=np.load(src['bank'])['bank'];saved=dict(np.load(src['target']))
        D,expert=make_contexts(x,bank,saved)
        actor=Actor(x['S'],x['v'],'CoupledRiskFTRL')
        for t in range(36):actor.issue(x['b'][t],x['ub'][t],D[t],expert[t],None if t==0 else x['truth'][t-1],calibration=True)
        for arm in needed:
            a=Actor(x['S'],x['v'],arm);a.records=copy.deepcopy(actor.records);a.pending=copy.deepcopy(actor.pending);a.A=actor.A.copy()
            results=[];logs=[]
            for t in range(36,108):
                row=a.issue(x['b'][t],x['ub'][t],D[t],expert[t],x['truth'][t-1],reset=t==36)
                logs.append(dict(t=t,fit=row.pop('fit'),candidates=row.pop('candidates')));results.append(row)
                if (t-36)%6==0:print('PROGRESS',ci,rel,arm,t-35,'/72',round(time.perf_counter()-begin,1),logs[-1]['fit']['active_future'],flush=True)
                # Recoverable checkpoint, not a completed or scored result.
                if (t-36)%12==0:
                    np.savez_compressed(dest/(arm+'_partial.npz'),**{k:np.asarray([r[k] for r in results]) for k in row})
                    (dest/(arm+'_partial.json')).write_text(json.dumps(logs,default=lambda z:z.item() if isinstance(z,np.generic) else z.tolist()))
            out={k:np.asarray([r[k] for r in results]) for k in results[0]};checks=audit(x,out,1)
            output=dest/(arm+'.npz');np.savez_compressed(output,**out)
            pack=dest/(arm+'_packets.npz');np.savez_compressed(pack,**{k:np.asarray([r[k] for r in a.records]) for k in a.records[0]})
            meta=dict(passed=True,cell=cell,arm=arm,checks=checks,endpoint_sha256=sha(src['endpoint']),registration_sha256=sha(reg),output_sha256=sha(output),packet_sha256=sha(pack),seconds=time.perf_counter()-begin,logs=logs,
                rmse=float(np.sqrt(((out['pred']-x['truth'][36:])**2).mean(0)).mean()),max_fit_relative_gap=max(r['fit']['relative_gap'] for r in logs))
            (dest/(arm+'.json')).write_text(json.dumps(meta,indent=2,default=lambda z:z.item() if isinstance(z,np.generic) else z.tolist()))
            print('DONE',ci,rel,arm,meta['rmse'],round(meta['seconds'],1),flush=True)
    completed=[str(HERE/f"results/{'/'.join(c)}/42/{arm}.json") for c in CELLS for arm in ARMS if (HERE/f"results/{'/'.join(c)}/42/{arm}.json").exists()]
    (HERE/'RUN_STATUS.json').write_text(json.dumps(dict(completed_paths=len(completed),required_paths=36,complete=len(completed)==36,completed=completed,registration_sha256=sha(reg)),indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
