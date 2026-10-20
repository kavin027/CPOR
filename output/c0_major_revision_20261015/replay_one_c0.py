"""One-cell C0 replay in the research checkout; never writes old results."""
from pathlib import Path
import importlib.util,sys,argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent; O=H.parent; ROOT=O.parent
ap=argparse.ArgumentParser();ap.add_argument('--cell',type=int,default=0);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
dest=a.out.resolve()
if not dest.is_relative_to(H.resolve()):raise ValueError('Replay output must be inside this revision folder')
dest.mkdir(parents=True,exist_ok=True)
if (dest/'C0.npz').exists():raise ValueError('Use a fresh output directory')
source=O/'paired_root_actual_credit_20261013/numerical_v2'
sys.path.insert(0,str(source));spec=importlib.util.spec_from_file_location('c0_frozen_replay',source/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cell=m.CELLS[a.cell];paths=m.original.paths(cell)
x=dict(np.load(paths['endpoint'],allow_pickle=True)); bank=np.load(paths['bank'])['bank'];saved=dict(np.load(paths['target']))
D,experts=m.make_contexts(x,bank,saved,m.original)
started=time.perf_counter()
with threadpool_limits(1):
    actor=m.Actor(x['S'],x['v'],'C0')
    for t in range(36):actor.issue(t,x['b'][t],x['ub'][t],D[t],experts[t],None if t==0 else x['truth'][t-1],calibration=True)
    rows=[]; logs=[]
    for t in range(36,108):
        row=actor.issue(t,x['b'][t],x['ub'][t],D[t],experts[t],x['truth'][t-1],reset=t==36)
        logs.append(row.pop('fit')); rows.append(row)
    out={k:np.array([r[k] for r in rows]) for k in rows[0]}
    check=m.original.audit(x,out,1)
np.savez_compressed(dest/'C0.npz',**out)
old=np.load(source/'results'/Path(*cell)/'42/C0.npz')['pred']
report=dict(cell=cell,checks=check,seconds=time.perf_counter()-started,max_abs_replay_difference=float(np.max(np.abs(old-out['pred']))),mean_node_rmse=float(np.sqrt(np.mean((out['pred']-x['truth'][36:])**2,axis=0)).mean()),solver_failures=int(out['solver_failed'].sum()),logs=logs)
(dest/'REPLAY_RESULT.json').write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!='logs'},indent=2))
