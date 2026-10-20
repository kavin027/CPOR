from pathlib import Path
import sys,json,time,hashlib,copy
import numpy as np
from threadpoolctl import threadpool_limits
import conic_learning as kernel
HERE=Path(__file__).resolve().parent;O=HERE.parents[1];ROOT=O.parent
sys.path.insert(0,str(O/'coupled_publication_learning_20261012'))
import run_experiment as original
sys.path.insert(0,str(O/'self_funded_policy_learning_20261013'))
from calibration_contexts import make_contexts
CELLS=original.CELLS;ARMS=['C0','C1','C2'];SHA=original.sha
INWARD=1e-10

def make_bank(qp,S,b,ub,D,experts,A,calibration=False):
    P=[];U=[]
    for target in [D,*experts.T]:
        p,u,info=original.one_action(qp,S,b,ub,target,A)
        if not calibration:
            p=b+(1-INWARD)*(p-b);u=ub+(1-INWARD)*(u-ub)
            lo,hi=original.safe_bounds(b,np.array([float(a) for a in A]))
            p,u,_=original.guard(b,p,ub,u,S,lo,hi,A,original.RR,original.K)
        assert original.certified(b,p,A,original.RR,original.K,True)
        P.append(p);U.append(u)
    P.append(b.copy());U.append(ub.copy())
    return np.asarray(P).T,np.asarray(U).T

class Actor:
    def __init__(self,S,v,arm):
        self.S=S;self.v=v;self.arm=arm;self.qp=original.SafeProjection(S,v)
        self.A=[original.Fraction(0) for _ in v];self.E=np.zeros(len(v));self.records=[];self.pending=None
    def close(self,label):
        old=self.pending
        self.A=original.advance(self.A,old['b'],old['published'],label)
        self.E+=(old['published']-label)**2
        self.records.append({**old,'y':np.array(label,copy=True),'label_event':2*(old['index']+1)})
    def issue(self,index,b,ub,D,experts,previous_label=None,calibration=False,reset=False):
        if self.pending is not None:
            assert previous_label is not None
            self.close(previous_label)
        if reset:self.A=[original.Fraction(0) for _ in self.v];self.E=np.zeros(len(self.v))
        P,U=make_bank(self.qp,self.S,b,ub,D,experts,self.A,calibration)
        failed=False
        if calibration:theta=np.eye(11)[0];info=dict(calibration=True)
        else:
            assert all(r['label_event']<2*index+1 for r in self.records)
            try:theta,info,_=kernel.fit(self.records,self.S,self.arm)
            except Exception as err:
                theta=np.eye(11)[-1];info=dict(passed=False,error=repr(err),fallback='registered_anchor',records=len(self.records));failed=True
        initial=b+(P-b[:,None])@theta;u=ub+(U-ub[:,None])@theta
        lo,hi=original.safe_bounds(b,np.array([float(a) for a in self.A]))
        p,u,rho=original.guard(b,initial,ub,u,self.S,lo,hi,self.A,original.RR,original.K)
        applied=rho*theta+(1-rho)*np.eye(11)[-1]
        risk=original.excess_bound(b,p,original.RR,original.K);available=[a-r for a,r in zip(self.A,risk)]
        assert min(available)>=0
        state=dict(index=index,epoch=0 if calibration else 1,scored=not calibration,issue_event=2*index+1,
                   P=P,U=U,b=b.copy(),ub=ub.copy(),E=self.E.copy(),A=np.array([float(a) for a in self.A]),
                   A_num=np.array([str(a.numerator) for a in self.A]),A_den=np.array([str(a.denominator) for a in self.A]),
                   theta=applied.copy(),published=p.copy())
        self.pending=state
        return dict(pred=p,witness=u,theta=applied,fit_theta=theta,actual_credit=state['A'],energy_before=state['E'],
                    available_after=np.array([float(a) for a in available]),reserved=np.array([float(a) for a in risk]),
                    lo=lo,hi=hi,contraction=rho,rounding_move=float(abs(p-initial).max()),solver_failed=failed,fit=info)

def registration():
    admission=json.loads((HERE/'ADMISSION.json').read_text());assert admission['passed']
    for p,h in admission['source_sha256'].items():assert SHA(p)==h,p
    for p,h in json.loads((original.C/'freeze_v2.json').read_text())['files'].items():assert SHA(p)==h,p
    source={str(p.resolve()):SHA(p) for p in [Path(__file__),HERE/'conic_learning.py',HERE/'causal_continuation.py',HERE/'PAIRED_ROOT_OBJECTIVE_AND_PROOF.md',HERE/'METHOD_DIFF_AND_SOURCE_CHECK.md']}
    for mod in list(sys.modules.values()):
        path=getattr(mod,'__file__',None)
        if path and Path(path).suffix=='.py' and Path(path).resolve().is_relative_to(O.resolve()):source[str(Path(path).resolve())]=SHA(path)
    reg=dict(cells=CELLS,arms=ARMS,seed=42,source_sha256=source,input_sha256={str(p):SHA(p) for c in CELLS for p in original.paths(c).values()},
        admission_sha256=SHA(HERE/'ADMISSION.json'),parameters=dict(alpha=kernel.ALPHA,gap=kernel.GAP,solver_tols=kernel.TOLS,max_iter=kernel.MAX_ITER,time_limit_per_attempt=kernel.TIME_LIMIT,
        history='all mature labels; paired endpoints corrected within calibration/scoring epochs',candidates=11,bank_extra_inward=INWARD,calibration=36,score=72,feedback_delay=1,initial_scoring_credit_and_energy=0),
        primary='10 dataset/backbone groups; average windows before RMSE comparison; MAE/nMSE secondary; retain all nodes',
        decision=dict(C1_vs_C0_min_wins=6,C1_vs_C0_mean_gain_pct=.2,C1_vs_C0_median_positive=True,threshold_is_resource_decision_not_CCF_standard=True),
        control='C0 re-matched to native SOCP precision and per-date budget; old MetricMatchedCurrentOnly retained, not overwritten',
        causal_continuation='Frozen matched-C0 rule; same pre-action history, replace current prediction, close current label, rebuild next bank and update using only information available then',
        fallback='At exhausted registered numerical budget publish anchor; count every prediction and all failures in complete grid',
        limits=['Development data, no untouched confirmation.','Future q is hindsight recourse, not causal prediction.','Finite bank hull is not full feasible set.','No inherited model-path guarantee or unproved whole-policy regret bound.','Pair weights match explicit counts, not necessarily effective surrogate risk scale.','Single seed cached external baseline is not author-original protocol reproduction.'])
    reg=json.loads(json.dumps(reg));path=HERE/'PREREGISTRATION_PAIRED_ROOT.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:path.write_text(json.dumps(reg,indent=2))
    return path

def main():
    reg=registration();selection=sys.argv[1:] or ['all']
    for ci,cell in enumerate(CELLS):
        if 'all' not in selection and str(ci) not in selection:continue
        rel='/'.join(cell)+'/42';dest=HERE/f'results/{rel}';dest.mkdir(parents=True,exist_ok=True)
        paths=original.paths(cell);x=dict(np.load(paths['endpoint']));bank=np.load(paths['bank'])['bank'];saved=dict(np.load(paths['target']))
        D,experts=make_contexts(x,bank,saved,original)
        for arm in ARMS:
            output=dest/(arm+'.npz')
            if output.with_suffix('.json').exists():continue
            started=time.perf_counter();a=Actor(x['S'],x['v'],arm)
            for t in range(36):a.issue(t,x['b'][t],x['ub'][t],D[t],experts[t],None if t==0 else x['truth'][t-1],calibration=True)
            rows=[];logs=[]
            for t in range(36,108):
                row=a.issue(t,x['b'][t],x['ub'][t],D[t],experts[t],x['truth'][t-1],reset=t==36)
                logs.append(dict(t=t,fit=row.pop('fit')));rows.append(row)
                if (t-36)%6==0:print('PROGRESS',ci,arm,rel,t-35,'/72',round(time.perf_counter()-started,1),'fail',sum(r['solver_failed'] for r in rows),flush=True)
                if (t-36)%12==0:
                    np.savez_compressed(dest/(arm+'_partial.npz'),**{k:np.array([r[k] for r in rows]) for k in row})
                    (dest/(arm+'_partial.json')).write_text(json.dumps(logs))
            out={k:np.array([r[k] for r in rows]) for k in rows[0]};checks=original.audit(x,out,1)
            a.close(x['truth'][-1]);a.pending=None
            np.savez_compressed(output,**out)
            stream=dest/(arm+'_snapshots.npz');np.savez_compressed(stream,**{k:np.array([r[k] for r in a.records]) for k in a.records[0]})
            meta=dict(passed=True,cell=cell,arm=arm,checks=checks,registration_sha256=SHA(reg),endpoint_sha256=SHA(paths['endpoint']),
                      output_sha256=SHA(output),snapshot_sha256=SHA(stream),seconds=time.perf_counter()-started,failed_publications=int(out['solver_failed'].sum()),logs=logs)
            output.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('DONE',ci,arm,rel,round(meta['seconds'],1),'fail',meta['failed_publications'],flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
