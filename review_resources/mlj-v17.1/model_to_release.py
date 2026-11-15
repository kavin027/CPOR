"""Run the frozen model exports for the complete 96-issue AEMO example."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

def verify(root,here,device):
    def sibling(n):
        return next(p for p in [root/f'v{n}',root/f'mlj-v{n}'] if p.is_dir())
    temporal=sibling(17)/'temporal';prior=sibling(16)/'temporal'
    sys.path.insert(0,str(temporal))
    from replay import State,scores
    from publishers import publish,ar,S,SafeProjection
    receipt=json.loads((here/'model_receipt.json').read_text())
    for name in ('DLinear','PIR'):
        assert hashlib.sha256((here/(name+'.pt')).read_bytes()).hexdigest()==receipt[name+'_export']
    torch.set_num_threads(1)
    base=torch.jit.load(str(here/'DLinear.pt'),map_location=device).eval();pir=torch.jit.load(str(here/'PIR.pt'),map_location=device).eval()
    obs=pd.read_csv(prior/'observed.csv',parse_dates=['ds']).set_index('ds')
    clock=pd.read_csv(prior/'clock.csv',parse_dates=['ds','last_component_issued']).set_index('ds')
    assert clock.index.equals(obs.index)
    dates=obs.index
    loc=obs.to_numpy()
    contexts=[]
    for t in range(24,len(obs)):
        known=np.flatnonzero((clock.last_component_issued<=dates[t]-pd.Timedelta(minutes=30)).to_numpy()[:t])
        ctx=loc[t-24:t].copy()
        for j in range(t-24,t):
            if j in known:continue
            lo,hi=known[known<j][-2:]
            ctx[j-t+24]=loc[hi]+(j-hi)*(loc[hi]-loc[lo])/(hi-lo)
        contexts.append(ctx)
    mean,sd=np.array(receipt['mean']),np.array(receipt['sd'])
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    with torch.no_grad():
        raw=np.concatenate([base(tensor(c)).cpu().numpy() for c in contexts[::12]])
        refined=[]
        for t,(context,forecast) in enumerate(zip(contexts,raw)):
            d=dates[t+24]
            mark=tensor([[[d.minute/59-.5,d.hour/23-.5,d.weekday()/6-.5,(d.day-1)/30-.5,(d.dayofyear-1)/365-.5]]])
            pred=pir(tensor((context-mean)/sd)[None,:,:],tensor((forecast-mean)/sd)[None,None,:],mark)
            refined.append(pred.cpu().numpy()[0,0]*sd+mean)
    saved=prior/'predictions/DLinear/1/forecast.npz'
    assert hashlib.sha256(saved.read_bytes()).hexdigest()==receipt['original_forecast_npz_sha256']
    with np.load(saved) as archived:
        for key,value in [('raw',raw),('pir',refined)]:
            diff=np.abs(value-archived[key]).max()
            print(key,device+' versus archive, maximum difference (MW)',diff)
            np.testing.assert_allclose(value,archived[key],rtol=2e-6,atol=.005)
    inputs=np.load(temporal/'inputs/DLinear_1.npz')
    expected=np.load(temporal/'expected/DLinear_1.npz')
    v=inputs['v'];scale=1/np.sqrt(v)
    u=np.stack([nnls(S*scale[:,None],p*scale)[0] for p in refined]);z=u@S.T
    np.testing.assert_allclose(z,inputs['Z'],rtol=2e-6,atol=.005)
    targets=pd.read_csv(temporal/'scores.csv').query("bb=='DLinear' and run==1").set_index('rule')
    for arm in ('Full','Independent'):
        state=State();solver=SafeProjection(S,v);issued=[]
        for t in range(96):
            b=inputs['B'][t];free=state.available(inputs['issue_ns'][t])
            p,_,pq,_,_=publish(arm,solver,v,b,inputs['UB'][t],z[t],u[t],free)
            state.add(inputs['ready_ns'][t],ar.exact(b),pq,free,ar.exact(inputs['Y'][t]))
            issued.append(p)
        result=scores(np.array(issued),inputs['Y'])
        assert not state.bad and not state.violations.any()
        print(arm,'RMSE MW',result['overall'],'region',result['regions'],'system',result['SYSTEM'],
              'max release deviation MW',np.abs(issued-expected[arm]).max(),flush=True)
        for level in result:
            np.testing.assert_allclose(result[level],targets.loc[arm,level],atol=.005,rtol=2e-6)
        np.testing.assert_allclose(issued,expected[arm],atol=.005,rtol=2e-6)
    print('PASS frozen model to candidate to 96 publications per rule and eight RMSE summaries.')

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=here.parent)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('The checkpoint-to-forecast example uses CUDA. Use the CPU v17/check.py for the complete publication replay from saved forecasts.')
    verify(args.root,here,'cuda')
