"""Complete Ada-Y/PIR repetitions on the existing matched prediction streams."""
from pathlib import Path
import sys,json,hashlib,time,random,logging,importlib.util
import numpy as np
import pandas as pd
import torch
H=Path(__file__).resolve().parent;O=H.parent;R=O.parent
sys.path.insert(0,str(O/'c0_external_sota_20261014'))
import pir_adapter as pir
sys.path.insert(0,str(O/'c0_external_adapter_20261014'))
import run_aday as aday
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files if k!='dates'}
cells=json.loads((O/'kais_review_response_20261016/Online_Resource_1/cells.json').read_text())
def widepath(ds):
    if ds=='M5':return O/'c0_frozen_m5_20261014/observed.csv'
    if ds=='Labour':return O/'rr_recap_evidence_extension_20261002/major_revision_20261002/external/data/Labour.csv'
    return O/f'procast_joint_safe_20261001_v4/data/{ds}.csv'
def rawpath(ds,bb,w,seed):
    if ds=='M5':
        return (O/'c0_frozen_m5_20261014/results'/bb/w/'42/raw.npz' if seed==42 else
                O/'c0_seed_validation_20261016/results'/ds/bb/w/str(seed)/'raw.npz')
    base=O/'rr_recap_evidence_extension_20261002'
    if ds=='Labour':base=base/'major_revision_20261002/external'
    return base/'forecasts'/ds/bb/w/str(seed)/'raw.npz'
def history(wide,ds,bb,seed,inputs):
    hp=O/'rr_recap_trajectory_learning_20261003/history'/ds/bb
    meta=json.loads((hp/'complete.json').read_text());parts=[]
    inputs[str(hp/'complete.json')]=sha(hp/'complete.json')
    for off in range(0,meta['count'],72):
        fd=O/'c0_seed_validation_20261016/support'/ds/bb/str(seed)/str(off)
        if not (fd/'predictions.csv').exists():
            # Only Labour's older two support blocks are absent. They precede
            # both calibration and scoring; the scored backbone is never refit.
            assert ds=='Labour' and off in [0,72],(ds,off)
            fd=H/'historical_support'/ds/bb/str(seed)/str(off)
            sys.path.insert(0,str(O/'c0_seed_validation_20261016'))
            import gpu as saved_gpu
            saved_gpu.H=H
            saved_gpu.fit(wide,bb,seed,meta['start']+off,min(72,meta['count']-off),fd,'MS',1)
        info=json.loads((fd/'run.json').read_text())
        assert info['seed']==seed and info['backbone']==bb
        assert sha(fd/'predictions.csv')==info['predictions_sha256']
        assert info['validation_end']<info['prediction_start']
        inputs[str(fd/'predictions.csv')]=sha(fd/'predictions.csv')
        inputs[str(fd/'run.json')]=sha(fd/'run.json')
        parts.append(pd.read_csv(fd/'predictions.csv',parse_dates=['ds','origin']))
    frame=pd.concat(parts);assert (frame.origin<frame.ds).all()
    vals=frame.pivot(index='ds',columns='unique_id',values=bb).reindex(
        index=wide.index[meta['start']:meta['end']],columns=wide.columns).to_numpy()
    assert vals.shape==(meta['count'],wide.shape[1]) and np.isfinite(vals).all()
    return vals
def data(c,seed):
    ds,bb,w=c['dataset'],c['backbone'],c['window']
    dp=widepath(ds);wide=pd.read_csv(dp,parse_dates=['ds']).set_index('ds')
    values=wide.to_numpy();cp=rawpath(ds,bb,w,seed);x=loadnp(cp)
    inputs={str(dp):sha(dp),str(cp):sha(cp)}
    if ds=='M5':
        start=int(x['begin']);end=int(x['start']);raw=x['raw'];S=x['S'];dropped=0
    else:
        mp=O/'rr_recap_trajectory_learning_20261003/inputs'/ds/bb/w/'metadata.json'
        meta=json.loads(mp.read_text());start,end=meta['start'],meta['end']
        inputs[str(mp)]=sha(mp)
        if seed==42:
            hp=mp.with_name('train.npz');cached=loadnp(hp);assert sha(hp)==meta['output_sha256']
            raw=np.concatenate([cached['raw'],x['raw'][36:]]);inputs[str(hp)]=sha(hp)
        else:
            parts=[history(wide,ds,bb,seed,inputs)]
            if ds!='Labour' and w=='late':
                ep=rawpath(ds,bb,'early',seed);ea=loadnp(ep)
                parts.append(ea['raw'][:72]);inputs[str(ep)]=sha(ep)
            parts.append(x['raw']);raw=np.concatenate(parts)
        S=x['S'];dropped=max(0,24-start)
    assert len(raw)==end+72-start and end==int(x['start'])
    np.testing.assert_array_equal(x['truth'][-72:],values[end:end+72])
    np.testing.assert_array_equal(raw[-72:],x['raw'][-72:])
    train_end=end-36;pos=np.arange(start+dropped,end+72);raw=raw[dropped:]
    mu=values[:train_end].mean(0);sd=values[:train_end].std(0);sd=np.where(sd>0,sd,1.)
    ctx=np.stack([values[t-24:t] for t in pos]);lp=np.arange(24,train_end)
    d=dict(x=((ctx-mu)/sd).astype('float32'),raw=((raw-mu)/sd)[:,None,:].astype('float32'),
        y=((values[pos]-mu)/sd)[:,None,:].astype('float32'),
        encmark=np.stack([pir.calendar(wide.index[t-24:t]) for t in pos]),
        mark=pir.calendar(wide.index[pos])[:,None,:],
        lib_x=((np.stack([values[t-24:t] for t in lp])-mu)/sd).astype('float32'),
        lib_y=((values[lp]-mu)/sd)[:,None,:].astype('float32'),
        library_index=pos-24,mu=mu,sd=sd,score_raw=raw[-72:],
        truth=values[end:end+72],S=S,train_count=train_end-start-dropped,
        score_start=end-start-dropped,inputs=inputs,
        library_last_label=str(wide.index[train_end-1]),score_first=str(wide.index[end]))
    assert d['score_start']-d['train_count']==36 and len(d['raw'])-d['score_start']==72
    assert d['library_last_label']<d['score_first']
    # Exact common forecast interface with the existing CPOR trajectory.
    ep=O/'c0_seed_validation_20261016/results'/ds/bb/w/str(seed)/'endpoints.npz'
    baseline=loadnp(ep)
    np.testing.assert_array_equal(d['score_raw'],baseline['raw'][-72:])
    np.testing.assert_array_equal(d['truth'],baseline['truth'][-72:])
    if seed==42:
        known=aday.load(('M5' if ds=='M5' else 'original',ds,bb,w))
        assert d['train_count']==known['train_count'] and d['score_start']==known['score_start']
        for key in ['mu','sd','score_raw','truth']:
            np.testing.assert_array_equal(d[key],known[key])
        np.testing.assert_array_equal(d['raw'][:,0],known['raw'])
        if ds!='M5':
            old=pir.load((ds,bb,w))
            for key in ['x','raw','y','lib_x','lib_y','library_index','encmark','mark']:
                np.testing.assert_array_equal(d[key],old[key])
    return d
def fit_aday(d,device):
    raw=torch.as_tensor(d['raw'][:,0],device=device);y=torch.as_tensor(d['y'][:,0],device=device)
    n=raw.shape[1];model=aday.PostProcessingNet(n,512,n,.2).float().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.0001);best=float('inf');state=None;wait=0;logs=[]
    for epoch in range(10):
        model.train();order=torch.randperm(d['train_count'],device=device);train=[]
        for first in range(0,len(order),32):
            ids=order[first:first+32];opt.zero_grad();p0=raw[ids]*model(raw[ids])
            loss=(p0-y[ids]).square().mean();assert torch.isfinite(loss)
            loss.backward();opt.step();train.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val=[]
            for first in range(d['train_count'],d['score_start'],32):
                z=raw[first:min(first+32,d['score_start'])]
                val.append(float((z*model(z)-y[first:first+len(z)]).square().mean()))
        score=float(np.mean(val));logs.append(dict(epoch=epoch+1,train_mse=float(np.mean(train)),validation_mse=score))
        if score<best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0
        else:wait+=1
        if wait>=3:break
        for g in opt.param_groups:g['lr']=.0001*(.5**epoch)
    model.load_state_dict(state);model.eval()
    with torch.no_grad():
        factor=model(raw[d['score_start']:])
        assert factor.min()>=.8-1e-6 and factor.max()<=1.2+1e-6
        pred=(raw[d['score_start']:]*factor).cpu().numpy()*d['sd']+d['mu']
    return pred,state,logs
def run():
    torch.set_num_threads(1);logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    reg=dict(purpose='Three matched repetitions for external comparisons; requested by author',
        seeds=[42,1,2],cells=[{k:c[k] for k in ['dataset','backbone','window']} for c in cells],
        reused_first_experiment=True,scored_backbones_frozen=True,
        missing_support_only='8 Labour historical fits, same original schedule, before calibration',
        new_adapter_runs=88,no_parameter_search=True,
        pir='unchanged author forward and prior fit/config; original shuffle order remains fixed',
        aday=dict(hidden=512,delta=.2,epochs=10,lr=.0001,batch=32,patience=3),
        sources={str(p):sha(p) for p in [Path(__file__),Path(pir.__file__),Path(aday.__file__),
                  O/'rr_recap_trajectory_learning_20261003/history.py',O/'c0_seed_validation_20261016/gpu.py']})
    rp=H/'REGISTRATION.json'
    if rp.exists():assert json.loads(rp.read_text())==reg
    else:js(rp,reg)
    rows=[]
    for c in cells:
        for seed in [42,1,2]:
            d=data(c,seed);dest=H/'results'/c['dataset']/c['backbone']/c['window']/str(seed)
            dest.mkdir(parents=True,exist_ok=True)
            for method,key in [('Ada-Y','AdaYBatch'),('PIR','PIRFrozenCache')]:
                out=dest/(method+'.npz');meta=out.with_suffix('.json')
                if seed==42:
                    src=R/c['prediction_sources'][key];saved=loadnp(src)
                    np.testing.assert_array_equal(saved['raw'],d['score_raw'])
                    np.testing.assert_array_equal(saved['truth'],d['truth'])
                    info=dict(reused=str(src),sha256=sha(src),inputs=d['inputs'],
                              seed=seed,train_count=d['train_count'],validation_count=36,scored_count=72)
                    js(meta,info);rows.append(dict(cell=c['file'],seed=seed,method=method,path=str(src),reused=True))
                    continue
                if meta.exists():
                    info=json.loads(meta.read_text());assert sha(out)==info['sha256']
                    assert info['registration_sha256']==sha(rp) and info['inputs']==d['inputs']
                else:
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    device='cuda' if torch.cuda.is_available() else 'cpu';beg=time.perf_counter()
                    if method=='Ada-Y':pred,state,logs=fit_aday(d,device)
                    else:
                        tt=pir.tensors(d,device);model=pir.CachedPIR(pir.config(len(d['mu']))).float().to(device)
                        logs,state=pir.fit(model,tt,d['train_count'],d['score_start']);pred=[]
                        with torch.no_grad():
                            for i in range(d['score_start'],len(d['raw'])):
                                p,_=pir.forward(model,tt,torch.tensor([i],device=device))
                                pred.append(p.cpu().numpy()[0,0]*d['sd']+d['mu'])
                        pred=np.asarray(pred);del model,tt
                    assert pred.shape==d['truth'].shape and len(pred)==72 and np.isfinite(pred).all()
                    np.savez_compressed(out,pred=pred,truth=d['truth'],raw=d['score_raw'])
                    cp=dest/(method+'_checkpoint.pt');torch.save(state,cp)
                    js(meta,dict(sha256=sha(out),checkpoint_sha256=sha(cp),registration_sha256=sha(rp),
                        inputs=d['inputs'],seed=seed,epochs=logs,train_count=d['train_count'],
                        validation_count=36,scored_count=72,library_last_label=d['library_last_label'],
                        first_score=d['score_first'],seconds=time.perf_counter()-beg))
                    torch.cuda.empty_cache()
                rows.append(dict(cell=c['file'],seed=seed,method=method,path=str(out),reused=False))
                print('COMPLETE',c['dataset'],c['backbone'],c['window'],seed,method,flush=True)
            js(H/'RUN_STATUS.json',dict(completed=len(rows),required=132,new_completed=sum(not r['reused'] for r in rows)))
    js(H/'RESULT_INDEX.json',rows);print('ALL_EXTERNAL_REPETITIONS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':run()
