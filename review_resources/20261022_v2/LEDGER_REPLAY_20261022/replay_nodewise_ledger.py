"""Replay stored CPOR nodewise guarantees without importing the forecasting code.

Requires numpy and the archived NPZ files listed in SOURCE_INDEX.json.
--workspace remaps the original project root; no forecast values are distributed.
No optimizer, model training, candidate repair or coherence check is performed.
"""
from pathlib import Path
from fractions import Fraction as Q
import argparse, csv, hashlib, json
import numpy as np

def run(index, output, workspace=None):
    sources=json.loads(index.read_text())
    rows=[]
    R=Q(40401,40000)
    def locate(raw):
        if workspace:
            parts=raw.replace('\\','/').split('/IJEPES_RR_RECAP/',1)
            assert len(parts)==2,raw
            return workspace/parts[1]
        return Path(raw)
    for src in sources:
        predpath, snap=locate(src['path']),locate(src['snapshot_path'])
        for p,h in [(predpath,src['sha256']),(snap,src['snapshot_sha256'])]:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==h,p
        with np.load(predpath,allow_pickle=False) as z: pred=z['pred'].copy()
        with np.load(snap,allow_pickle=False) as z:
            a={k:z[k][-72:].copy() for k in ['b','y','published','A_num','A_den','scored']}
        assert pred.shape==a['b'].shape==a['y'].shape and len(pred)==72
        assert np.array_equal(pred,a['published']) and np.all(a['scored'])
        assert all(np.isfinite(x).all() and (x>=0).all() for x in [pred,a['b'],a['y']])
        n=pred.shape[1];balance=[Q(0) for _ in range(n)]
        pred_sse=[Q(0) for _ in range(n)];anchor_sse=[Q(0) for _ in range(n)]
        for t in range(72):
            for i in range(n):
                b,p,y=(Q.from_float(float(x[t,i])) for x in [a['b'],pred,a['y']])
                stored=Q(int(a['A_num'][t,i]),int(a['A_den'][t,i]))
                assert stored==balance[i],('archived balance mismatch',src,t,i)
                # Supremum over y>=0, with R>1, evaluated as an exact rational.
                positive=max(R*b-p,Q(0))
                envelope=p*p-R*b*b+positive*positive/(R-1)
                assert envelope<=balance[i],('envelope violation',src,t,i)
                error=(p-y)**2;reference_error=(b-y)**2
                balance[i]+=R*reference_error-error
                pred_sse[i]+=error;anchor_sse[i]+=reference_error
                assert balance[i]>=0,('negative allowance',src,t,i)
                assert pred_sse[i]<=R*anchor_sse[i],('prefix violation',src,t,i)
                assert balance[i]==R*anchor_sse[i]-pred_sse[i]
        rmse=float(np.sqrt(((pred-a['y'])**2).mean(0)).mean())
        assert np.isclose(rmse,src['rmse'],rtol=1e-10,atol=1e-10)
        rows.append({k:src[k] for k in ['dataset','backbone','window','run']}|dict(nodes=n,releases=72,node_checks=72*n,rmse=rmse,envelope_violations=0,prefix_violations=0,balance_mismatches=0))
        print('PASS',src['dataset'],src['backbone'],src['window'],src['run'],flush=True)
    assert len(rows)==66 and sum(r['node_checks'] for r in rows)==431568
    output.mkdir(parents=True,exist_ok=True)
    with (output/'TRAJECTORY_REPLAY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(passed=True,trajectories=len(rows),releases=sum(r['releases'] for r in rows),node_checks=sum(r['node_checks'] for r in rows),envelope_violations=0,prefix_violations=0,balance_mismatches=0,R='40401/40000',scope='Exact rational nodewise envelope, accumulated squared-error limit and archived pre-release balances for stored CPOR forecasts. Separate reimplementation, not an external independent replication; excludes coherence, candidate repairs, other methods and model retraining.')
    (output/'REPLAY_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,default=Path(__file__).with_name('SOURCE_INDEX.json'));p.add_argument('--output',type=Path,default=Path(__file__).parent);p.add_argument('--workspace',type=Path)
    a=p.parse_args();run(a.index,a.output,a.workspace)
