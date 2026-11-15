"""Frozen model inference with issue-specific first-report availability."""
from pathlib import Path
import hashlib,json,sys,logging,os
import numpy as np,pandas as pd
W=Path(__file__).resolve().parent
O=W.parents[1]; N=O/'cpor_aemo_november_holdout_20261112'
read=lambda f:pd.read_csv(f,parse_dates=['ds']).set_index('ds')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(W/'REGISTRATION.txt')=='6d8fe62eadcc6e2e84298afff144fb4453e0f58e1f0d88d934a00919f5e3c8a6'
clock=pd.to_datetime(read(W/'clock.csv').last_component_issued)
hold=read(W/'observed.csv')
def available_context(w,i):
    start=i-24
    issue=w.index[i]-pd.Timedelta(minutes=30)
    ready=clock.to_numpy()<=issue.to_datetime64()
    real=np.flatnonzero(ready[:i])
    assert len(real)>=2
    vals=w.iloc[start:i].copy()
    for j in range(start,i):
        if ready[j]:continue
        known=real[real<j]
        assert len(known)>=2
        j0,j1=known[-2:]
        vals.iloc[j-start]=w.iloc[j1]+(j-j1)*(w.iloc[j1]-w.iloc[j0])/(j1-j0)
    return vals
src=(N/'predictions.py').read_text(encoding='utf-8').split('if __name__=="__main__"')[0]
src=src.replace('from data import H,O,REG,START,sha,freeze','')
src=src.replace('def predict(w):','def predict(w, delayed=False):\n        def context(i):return available_context(w,i) if delayed else w.iloc[i-24:i]')
src=src.replace('block(w.iloc[i-24:i])','block(context(i))').replace('w.iloc[i-24:i].values','context(i).values')
src=src.replace('raw,p=predict(hold)','raw,p=predict(hold,delayed=True)').replace('H/"PROTOCOL.md"','H/"REGISTRATION.txt"')
ctx=dict(H=W,O=O,sha=sha,available_context=available_context,__name__='frozen_inference')
exec(compile(src,str(N/'predictions.py'),'exec'),ctx)
ctx['torch'].set_num_threads(1);logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
hist=read(O/'cpor_aemo_underlying_october_20261107/observed.csv')
for bb,seeds in [('DLinear',[42,1,2]),('PatchTST',[42,1,2]),('DUET',[2021,2022,2023])]:
    for seed in seeds:
        ck=W/'predictions'/bb/str(seed)/'receipt.json'
        if ck.exists():assert json.loads(ck.read_text())['sha256']==sha(ck.with_name('forecast.npz'))
        else:ctx['compute'](bb,seed,hist,hold)
        print('READY',bb,seed,flush=True)
