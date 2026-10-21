from pathlib import Path
import argparse, hashlib, json, time
from typing import Optional, Callable
import numpy as np
import torch

R=Path(__file__).resolve().parent
O=R.parent
B=O/'rr_recap_evidence_extension_20261002'
M=B/'major_revision_20261002'
H=O/'rr_recap_trajectory_learning_20261003'
AUTHOR=B/'procast_author.py'
PinvCallback=Callable
source=AUTHOR.read_text()
exec(source[source.index('class _TorchPinv'):source.index('\ndef torch_oblique')],globals())

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def configs():
    return [(d,b,w) for d in ['RH','ECommerce','Traffic','Wiki2']
            for b in ['PatchTST','DLinear'] for w in ['early','late']]+[
            ('Labour',b,'late') for b in ['PatchTST','DLinear']]

def load(cell):
    ds,bb,win=cell
    cp=(M/'external/forecasts' if ds=='Labour' else B/'forecasts')/f'{ds}/{bb}/{win}/42/raw.npz'
    hp=H/f'inputs/{ds}/{bb}/{win}/train.npz'
    mp=hp.with_name('metadata.json')
    x=dict(np.load(cp)); h=dict(np.load(hp)); meta=json.loads(mp.read_text())
    assert sha(hp)==meta['output_sha256']
    assert meta['last_training_date']<meta['first_scored_date']
    assert len(x['raw'])==108
    np.testing.assert_array_equal(h['raw'][-36:],x['raw'][:36])
    np.testing.assert_array_equal(h['truth'][-36:],x['truth'][:36])
    np.testing.assert_array_equal(h['S'],x['S'])
    for p,digest in meta['source_sha256'].items(): assert sha(p)==digest,p
    return x,h,{str(p):sha(p) for p in [cp,hp,mp]}

def run(cell):
    ds,bb,win=cell; dest=R/f'results/{ds}/{bb}/{win}/42'; dest.mkdir(parents=True,exist_ok=True)
    x,h,inputs=load(cell)
    if (dest/'run.json').exists():
        info=json.loads((dest/'run.json').read_text())
        assert info['inputs']==inputs and sha(dest/'predictions.npz')==info['output_sha256']
        return
    raw=np.concatenate([h['raw'],x['raw'][36:]])
    truth=np.concatenate([h['truth'],x['truth'][36:]])
    start=len(h['raw']); S=x['S']; counts=[]
    yh=torch.tensor(raw,device='cuda',dtype=torch.float64)
    yt=torch.tensor(truth,device='cuda',dtype=torch.float64)
    model=_TorchPinv(torch.tensor(S,dtype=torch.float32),masked=False,max_iter=100000,
        callback=lambda item: counts.append(int(item[1])) if item[0]=='count' else None).to(device='cuda',dtype=torch.float64)
    out=[]; witness=[]; support=[]; begin=time.perf_counter()
    for i in range(start-36,len(raw)):
        n=min(72,i); assert n>0
        weights=torch.tensor([.5/n]*n+[.5],device='cuda',dtype=torch.float64)
        # yt[i] is NEVER part of the fitted target; current target is raw yh[i].
        model.fit(yh[i-n:i+1],torch.cat([yt[i-n:i],yh[i:i+1]]),weights=weights)
        with torch.no_grad():
            u=(yh[i:i+1]@model.pinv).clamp(min=0)
            out.append((u@model.mat).cpu().numpy()[0]); witness.append(u.cpu().numpy()[0])
        support.append(n)
        if (i-(start-36)+1)%36==0:
            print('PROGRESS',ds,bb,win,i-(start-36)+1,'of108',round(time.perf_counter()-begin,1),flush=True)
    p=np.asarray(out); u=np.asarray(witness)
    assert len(p)==108 and len(counts)==108 and np.isfinite(p).all() and u.min()>=0
    np.testing.assert_allclose(p,u@S.T,rtol=1e-9,atol=1e-5)
    np.savez_compressed(dest/'predictions.npz',pred=p[36:],cal_pred=p[:36],witness=u[36:],
        cal_witness=u[:36],truth=x['truth'][36:],raw=x['raw'][36:],support=support[36:],
        cal_support=support[:36],iterations=counts[36:],cal_iterations=counts[:36])
    info=dict(cell=cell,seed=42,seconds=time.perf_counter()-begin,inputs=inputs,
        source_sha256=sha(AUTHOR),runner_sha256=sha(__file__),protocol_sha256=sha(R/'PROTOCOL.txt'),
        output_sha256=sha(dest/'predictions.npz'),scored_count=72,first_support=support[36],
        last_support=support[-1],scored_iteration_limit_hits=sum(n>=100000 for n in counts[36:]),
        calibration_iteration_limit_hits=sum(n>=100000 for n in counts[:36]))
    (dest/'run.json').write_text(json.dumps(info,indent=2))
    print('DONE',ds,bb,win,round(info['seconds'],1),flush=True)
    del model; torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    freeze={str(p):sha(p) for p in [Path(__file__),R/'PROTOCOL.txt',AUTHOR]}
    path=R/'REGISTRATION.json'
    assert path.exists() and json.loads(path.read_text())['sources']==freeze
    for i,cell in enumerate(configs()):
        if i%args.shards==args.shard:run(cell)
    print('COMPLETE',args.shard,flush=True)

if __name__=='__main__':main()
