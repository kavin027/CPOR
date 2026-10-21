from common import *
import torch,logging,shutil,argparse,functools
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST,DLinear
from pytorch_lightning.callbacks import ModelCheckpoint
def serialized_training(fn):
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        lock=H/'training_lock'
        while True:
            try:lock.mkdir();break
            except FileExistsError:time.sleep(2)
        try:return fn(*args,**kwargs)
        finally:lock.rmdir()
    return wrapped
@serialized_training
def fit(wide,bb,seed,ix,count,fd,freq,threads):
    fd.mkdir(parents=True,exist_ok=True);p=fd/'run.json'
    if p.exists():
        j=json.loads(p.read_text());assert sha(fd/'predictions.csv')==j['predictions_sha256'];return pd.read_csv(fd/'predictions.csv',parse_dates=['ds','origin'])
    torch.set_num_threads(threads);start=time.perf_counter();cls=PatchTST if bb=='PatchTST' else DLinear
    cb=ModelCheckpoint(dirpath=str(fd/'checkpoints'),monitor='ptl/val_loss',save_top_k=1)
    opts=dict(h=12,input_size=24,val_check_steps=1,early_stop_patience_steps=50,random_seed=seed,max_steps=500,enable_checkpointing=True,callbacks=[cb],enable_progress_bar=False,enable_model_summary=False,logger=False,accelerator='gpu',devices=1,default_root_dir=str(fd))
    if bb=='PatchTST':opts['n_series']=wide.shape[1]
    long=wide.reset_index().melt(id_vars='ds',var_name='unique_id',value_name='y').sort_values(['ds','unique_id'])
    nf=NeuralForecast(models=[cls(**opts)],freq=freq);nf.fit(df=long[long.ds<wide.index[ix]],val_size=36)
    cb=next(c for c in nf.models[0].trainer_kwargs['callbacks'] if isinstance(c,ModelCheckpoint));best=cb.best_model_path;assert best
    nf.models=[cls.load_from_checkpoint(best,weights_only=False,early_stop_patience_steps=0,enable_checkpointing=False,callbacks=[],enable_progress_bar=False,enable_model_summary=False,logger=False)]
    parts=[]
    for offset in range(0,count,12):
        i=ix+offset;z=nf.predict(df=long[long.ds<wide.index[i]])
        if 'unique_id' not in z:z=z.reset_index()
        z['origin']=wide.index[i-1];parts.append(z)
    frame=pd.concat(parts,ignore_index=True);assert len(frame)==count*wide.shape[1] and not frame.duplicated(['unique_id','ds']).any() and (frame.origin<frame.ds).all()
    frame.to_csv(fd/'predictions.csv',index=False)
    js(p,dict(seed=seed,backbone=bb,gradient_train_end=str(wide.index[ix-37]),validation_end=str(wide.index[ix-1]),prediction_start=str(wide.index[ix]),prediction_end=str(wide.index[ix+count-1]),checkpoint=best,checkpoint_sha256=sha(best),best_val=float(cb.best_model_score),best_step=int(torch.load(best,map_location='cpu',weights_only=False)['global_step']),seconds=time.perf_counter()-start,predictions_sha256=sha(fd/'predictions.csv'),torch=torch.__version__,gpu=torch.cuda.get_device_name(),registration_sha256=sha(H/'REGISTRATION.json')))
    del nf;torch.cuda.empty_cache();print('FIT_DONE',fd,round(time.perf_counter()-start,1),flush=True);return frame
def pivot(frame,wide,bb,ix,count):
    x=frame.pivot(index='ds',columns='unique_id',values=bb).reindex(index=wide.index[ix:ix+count],columns=wide.columns).to_numpy();assert x.shape==(count,wide.shape[1]) and np.isfinite(x).all();return x
def m5(c,seed):
    d=dest(c,seed);d.mkdir(parents=True,exist_ok=True)
    if (d/'raw.json').exists():return
    wide=pd.read_csv(datafile(c),parse_dates=['ds']).set_index('ds');S=np.load(O/'c0_frozen_m5_20261014/S.npy');start=1797 if c['window']=='early' else 1869;begin=start-144
    frame=fit(wide,c['backbone'],seed,begin,216,d/'base_fit','D',1);raw=pivot(frame,wide,c['backbone'],begin,216)
    np.savez_compressed(d/'raw.npz',raw=raw,truth=wide.to_numpy()[begin:begin+216],S=S,begin=begin,start=start,cal_start=start-36)
    js(d/'raw.json',dict(sha256=sha(d/'raw.npz'),data_sha256=sha(datafile(c)),seed=seed,registration_sha256=sha(H/'REGISTRATION.json')))
    print('M5_BASE_READY',c['backbone'],c['window'],seed,flush=True)
def support(c,seed):
    rp=load(rawfile(c,seed));wide=pd.read_csv(datafile(c),parse_dates=['ds']).set_index('ds');cal=int(rp['cal_start'])
    if c['window']=='late' and c['dataset']!='Labour':
        early={**c,'window':'early'};pre=load(rawfile(early,seed))['raw'][:72]
    else:
        d=histories(c,seed);p=d/'support.npz'
        if p.exists():assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];pre=load(p)['raw']
        else:
            meta=json.loads((O/'rr_recap_trajectory_learning_20261003/history'/c['dataset']/c['backbone']/'complete.json').read_text());end=meta['end'];start=meta['start'];count=meta['count'];assert end==cal
            parts=[];needed=min(72,count)
            for offset in range(0,count,72):
                ix=start+offset;steps=min(72,count-offset)
                if ix+steps<=end-needed:continue
                freq='MS' if c['dataset']=='Labour' else 'B' if c['dataset']=='ECommerce' else 'D'
                parts.append(fit(wide,c['backbone'],seed,ix,steps,d/str(offset),freq,2))
            pre=pivot(pd.concat(parts,ignore_index=True),wide,c['backbone'],end-needed,needed)
            np.savez_compressed(p,raw=pre,truth=wide.to_numpy()[end-needed:end],begin=end-needed,end=end)
            js(p.with_suffix('.json'),dict(sha256=sha(p),data_sha256=sha(datafile(c)),seed=seed,source_schedule=meta,registration_sha256=sha(H/'REGISTRATION.json')))
    return np.concatenate([pre,rp['raw']]),wide.to_numpy()[cal-len(pre):cal+108],rp['S']
def procast(c,seed):
    d=dest(c,seed);d.mkdir(parents=True,exist_ok=True);p=d/'ProCAST72Warm.npz'
    if p.with_suffix('.json').exists():assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];return
    if seed==42:
        src=ROOT/c['prediction_sources']['ProCAST72Warm'];shutil.copyfile(src,p);js(p.with_suffix('.json'),dict(sha256=sha(p),reused=str(src),source_sha256=sha(src)));return
    if c['dataset']=='M5':
        x=load(d/'raw.npz');raw,y,S=x['raw'],x['truth'],x['S'];first=144
    else:raw,y,S=support(c,seed);first=len(raw)-108
    counts=[];torch.set_num_threads(2);yh=torch.tensor(raw,device='cuda',dtype=torch.float64);yt=torch.tensor(y,device='cuda',dtype=torch.float64)
    model=fresh.author_pro._TorchPinv(torch.tensor(S,dtype=torch.float32),masked=False,max_iter=100000,callback=lambda item:counts.append(int(item[1])) if item[0]=='count' else None).to(device='cuda',dtype=torch.float64)
    rows=[];us=[];started=time.perf_counter()
    for i in range(first,len(raw)):
        n=min(72,i);ww=torch.tensor([.5/n]*n+[.5],device='cuda',dtype=torch.float64)
        model.fit(yh[i-n:i+1],torch.cat([yt[i-n:i],yh[i:i+1]]),weights=ww)
        with torch.no_grad():u=(yh[i:i+1]@model.pinv).clamp(min=0);rows.append((u@model.mat).cpu().numpy()[0]);us.append(u.cpu().numpy()[0])
        if (i-first+1)%36==0:print('PROCAST_PROGRESS',c['dataset'],c['backbone'],c['window'],seed,i-first+1,round(time.perf_counter()-started,1),flush=True)
    np.savez_compressed(p,pred=np.array(rows)[-72:],witness=np.array(us)[-72:],iterations=counts[-72:],truth=y[-72:],raw=raw[-72:])
    js(p.with_suffix('.json'),dict(sha256=sha(p),seconds=time.perf_counter()-started,seed=seed,scored_iteration_limit_hits=sum(v>=100000 for v in counts[-72:]),calibration_calls=len(rows)-72,registration_sha256=sha(H/'REGISTRATION.json')))
    del model,yh,yt;torch.cuda.empty_cache();print('PROCAST_DONE',c['dataset'],c['backbone'],c['window'],seed,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);args=ap.parse_args()
    verify_registration();assert json.loads((H/'GATE.json').read_text())['passed'];logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    for c in cells:
        if c['dataset']=='M5':
            for seed in [1,2]:m5(c,seed)
    jobs=[(c,seed) for c in cells for seed in [42,1,2]]
    for c,seed in jobs[args.shard::args.shards]:procast(c,seed)
    print('GPU_COMPLETE',args.shard,flush=True)
