from pathlib import Path
import os,json,time,logging,hashlib,shutil,argparse
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'procast_joint_safe_20261001_v4'
for key,folder in [('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba')]:os.environ[key]=str(ROOT/'cache'/folder)
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST,DLinear
from pytorch_lightning.callbacks import ModelCheckpoint
DATASETS=['RH','ECommerce','Traffic','Wiki2'];SEEDS=[42,1,2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    m={name:sha(ROOT/name) for name in ['train.py','PROTOCOL.txt']}
    m['data']={ds:sha(OLD/f'data/{ds}.csv') for ds in DATASETS}
    p=ROOT/'training_freeze.json'
    if p.exists():assert json.loads(p.read_text())==m
    else:p.write_text(json.dumps(m,indent=2))
def main():
    freeze();torch.set_num_threads(4);logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=DATASETS);args=parser.parse_args()
    for ds in args.datasets:
        wide=pd.read_csv(OLD/f'data/{ds}.csv',parse_dates=['ds']).set_index('ds')
        long=wide.reset_index().melt(id_vars='ds',var_name='unique_id',value_name='y').sort_values(['ds','unique_id'])
        for backbone,cls in [('PatchTST',PatchTST),('DLinear',DLinear)]:
            for window,offset in [('early',144),('late',72)]:
                start=len(wide)-offset
                for seed in SEEDS:
                    dest=ROOT/f'forecasts/{ds}/{backbone}/{window}/{seed}';dest.mkdir(parents=True,exist_ok=True)
                    if (dest/'complete.json').exists():continue
                    if backbone=='PatchTST' and window=='late':
                        src=OLD/f'inputs/{ds}/{seed}/endpoints.npz';x=np.load(src)
                        np.savez_compressed(dest/'raw.npz',raw=x['raw'],truth=x['truth'],S=x['S'],start=start,cal_start=start-36)
                        (dest/'complete.json').write_text(json.dumps(dict(reused=str(src),sha256=sha(src),dataset=ds,backbone=backbone,window=window,seed=seed),indent=2));print('REUSED',ds,seed,flush=True);continue
                    for fold in range(4):
                        ix=start-36+12*fold;count=72 if fold==3 else 12
                        fd=dest/str(fold);fd.mkdir(exist_ok=True)
                        if (fd/'run.json').exists():continue
                        print('TRAIN',ds,backbone,window,seed,fold,flush=True);begin=time.perf_counter()
                        cb=ModelCheckpoint(dirpath=str(fd/'checkpoints'),monitor='ptl/val_loss',save_top_k=1)
                        opts=dict(h=12,input_size=24,val_check_steps=1,early_stop_patience_steps=50,random_seed=seed,max_steps=500,enable_checkpointing=True,callbacks=[cb],enable_progress_bar=False,enable_model_summary=False,logger=False,accelerator='gpu',devices=1,default_root_dir=str(fd))
                        if backbone=='PatchTST':opts['n_series']=wide.shape[1]
                        model=cls(**opts);nf=NeuralForecast(models=[model],freq='B' if ds=='ECommerce' else 'D')
                        nf.fit(df=long[long.ds<wide.index[ix]],val_size=36)
                        cb=next(c for c in nf.models[0].trainer_kwargs['callbacks'] if isinstance(c,ModelCheckpoint));best=cb.best_model_path;assert best
                        nf.models=[cls.load_from_checkpoint(best,weights_only=False,early_stop_patience_steps=0,enable_checkpointing=False,callbacks=[],enable_progress_bar=False,enable_model_summary=False,logger=False)]
                        predictions=[]
                        for step in range(0,count,12):
                            j=ix+step;p=nf.predict(df=long[long.ds<wide.index[j]])
                            if 'unique_id' not in p.columns:p=p.reset_index()
                            p['origin']=wide.index[j-1];predictions.append(p)
                        pred=pd.concat(predictions,ignore_index=True);assert len(pred)==count*wide.shape[1]
                        assert pd.to_datetime(pred.origin).lt(pd.to_datetime(pred.ds)).all()
                        assert not pred.duplicated(['unique_id','ds']).any();pred.to_csv(fd/'predictions.csv',index=False)
                        info=dict(dataset=ds,backbone=backbone,window=window,seed=seed,fold=fold,prediction_start=str(wide.index[ix]),prediction_end=str(wide.index[ix+count-1]),gradient_train_end=str(wide.index[ix-37]),validation_end=str(wide.index[ix-1]),best_val_loss=float(cb.best_model_score),best_checkpoint=best,checkpoint_sha256=sha(Path(best)),seconds=time.perf_counter()-begin,best_step=int(torch.load(best,map_location='cpu',weights_only=False)['global_step']))
                        (fd/'run.json').write_text(json.dumps(info,indent=2));print('DONE_FIT',ds,backbone,window,seed,fold,round(info['seconds'],1),flush=True)
                        del nf,model;torch.cuda.empty_cache()
                    frame=pd.concat([pd.read_csv(dest/str(f)/'predictions.csv',parse_dates=['ds']) for f in range(4)])
                    raw=frame.pivot(index='ds',columns='unique_id',values=backbone).reindex(index=wide.index[start-36:start+72],columns=wide.columns).to_numpy()
                    assert raw.shape==(108,wide.shape[1]) and np.isfinite(raw).all()
                    S=np.load(OLD/f'inputs/{ds}/42/endpoints.npz')['S']
                    np.savez_compressed(dest/'raw.npz',raw=raw,truth=wide.to_numpy()[start-36:start+72],S=S,start=start,cal_start=start-36)
                    (dest/'complete.json').write_text(json.dumps(dict(dataset=ds,backbone=backbone,window=window,seed=seed,raw_sha256=sha(dest/'raw.npz')),indent=2));print('CELL_READY',ds,backbone,window,seed,flush=True)
    print('ALL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
