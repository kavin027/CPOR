from pathlib import Path
import sys, json, hashlib, time, importlib.util, copy
import numpy as np
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'output'
sys.path.insert(0,str(O/'c0_matched_ablation_20261014'))
import experiment as exp
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:dict(np.load(p,allow_pickle=False))
cells=json.loads((O/'kais_review_response_20261016/Online_Resource_1/cells.json').read_text())
def source(c,seed):return O/'c0_seed_validation_20261016/results'/c['dataset']/c['backbone']/c['window']/str(seed)
def snapshot(c,seed):return (R/c['prediction_sources']['C0']).with_name('C0_snapshots.npz') if seed==42 else source(c,seed)/'C0_snapshots.npz'
def context(c,seed):
 d=source(c,seed);x=load(d/'endpoints.npz');ct=load(d/'contexts.npz');sn=load(snapshot(c,seed))
 records=[{k:np.array(a[t],copy=True) for k,a in sn.items()} for t in range(36)]
 assert all(r['label_event']<73 for r in records)
 np.testing.assert_array_equal(np.array([r['y'] for r in records]),x['truth'][:36])
 E=np.sum((sn['published'][:36]-x['truth'][:36])**2,axis=0)
 kappa=float(np.sqrt(E).mean()/E.mean()) if E.mean()>0 else 1/float(np.sqrt(x['v'].mean()))
 return x,ct['D'],ct['expert'],records,kappa

def run(ci,seed,arm,gate=False):
 c=cells[ci];reg=json.loads((H/'REGISTRATION.json').read_text())
 for p,h in reg['sources'].items():assert sha(p)==h,p
 for name in ['endpoints.npz','contexts.npz','C0_snapshots.npz']:
  p=snapshot(c,seed) if name=='C0_snapshots.npz' else source(c,seed)/name;assert sha(p)==reg['inputs'][str(p)],p
 dest=H/('gate' if gate else 'results')/c['dataset']/c['backbone']/c['window']/str(seed);dest.mkdir(parents=True,exist_ok=True);p=dest/(arm+'.npz')
 if p.with_suffix('.json').exists():
  j=json.loads(p.with_suffix('.json').read_text());assert j['sha256']==sha(p);return
 x,D,experts,records,kappa=context(c,seed);a=exp.Actor(x['S'],x['v'],arm,records,kappa)
 rows=[];logs=[];start=time.perf_counter()
 for t in range(36,108):
  z=a.issue(t,x['b'][t],x['ub'][t],D[t],experts[t],None if t==36 else x['truth'][t-1]);logs.append(dict(t=t,fit=z.pop('fit')));rows.append(z)
  if (t-35)%12==0:print('PROGRESS',ci,seed,arm,t-35,round(time.perf_counter()-start,1),flush=True)
 out={k:np.array([z[k] for z in rows]) for k in rows[0]};checks=exp.base.original.audit(x,out,1);a.close(x['truth'][-1]);a.pending=None
 np.savez_compressed(p,**out)
 snaps=dest/(arm+'_snapshots.npz');np.savez_compressed(snaps,**{k:np.array([z[k] for z in a.records]) for k in a.records[0]})
 if gate:
  old=O/'c0_matched_ablation_20261014/results'/c['dataset']/c['backbone']/c['window']/'42'/(arm+'.npz')
  np.testing.assert_array_equal(load(old)['pred'],out['pred'])
 info=dict(cell=ci,seed=seed,arm=arm,sha256=sha(p),snapshot_sha256=sha(snaps),registration_sha256=sha(H/'REGISTRATION.json'),checks=checks,kappa=kappa,seconds=time.perf_counter()-start,failed_publications=int(out['solver_failed'].sum()),gate_exact_replay=gate,logs=logs)
 p.with_suffix('.json').write_text(json.dumps(info,indent=2));print('DONE',ci,seed,arm,round(info['seconds'],1),flush=True)
if __name__=='__main__':
 with threadpool_limits(1):run(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],len(sys.argv)>4)

