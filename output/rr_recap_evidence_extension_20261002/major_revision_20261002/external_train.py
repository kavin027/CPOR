from pathlib import Path
import os,json,time,hashlib,zipfile,logging
ROOT=Path(__file__).resolve().parent;DEST=ROOT/'external';BASE=ROOT.parent
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['NUMBA_CACHE_DIR']=str(ROOT/'cache/numba')
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST,DLinear
from pytorch_lightning.callbacks import ModelCheckpoint
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(2);logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    data=DEST/'data';data.mkdir(parents=True,exist_ok=True)
    archive=BASE.parent/'procast_external_20261001/source/datasets.zip'
    frozen={str(p.relative_to(BASE)):sha(p) for p in [ROOT/'PROTOCOL.txt',Path(__file__),BASE/'controller.py',BASE/'endpoints.py',BASE/'fast_solver.py']};frozen['archive_sha256']=sha(archive)
    fp=ROOT/'external_freeze.json'
    if fp.exists():assert json.loads(fp.read_text())==frozen
    else:fp.write_text(json.dumps(frozen,indent=2))
    with zipfile.ZipFile(archive) as z:
        for name in ['data.csv','agg_mat.csv']:(data/name).write_bytes(z.read('Labour/'+name))
    wide=pd.read_csv(data/'data.csv',index_col=0);sm=pd.read_csv(data/'agg_mat.csv',index_col=0)
    wide.columns=wide.columns.astype(str);sm.index=sm.index.astype(str);sm.columns=sm.columns.astype(str)
    assert set(wide.columns)==set(sm.index)
    wide=wide.reindex(columns=sm.index);wide.index=pd.to_datetime(wide.index);wide=wide.sort_index()
    S=sm.to_numpy(float);bottom=[]
    for j in range(S.shape[1]):
        e=np.eye(S.shape[1])[j];ids=np.flatnonzero(np.all(S==e,axis=1));assert len(ids)==1;bottom.append(int(ids[0]))
    assert np.isfinite(wide.to_numpy()).all() and wide.to_numpy().min()>=0
    np.testing.assert_allclose(wide.to_numpy(),wide.to_numpy()[:,bottom]@S.T,rtol=1e-7,atol=1e-6)
    assert pd.infer_freq(wide.index)=='MS'
    assert len(wide)>=216
    wide.to_csv(data/'Labour.csv',index_label='ds');np.save(data/'Labour_S.npy',S)
    start=len(wide)-72
    (data/'metadata.json').write_text(json.dumps(dict(dataset='Labour',rows=len(wide),nodes=S.shape[0],bottom=S.shape[1],start=str(wide.index[0]),end=str(wide.index[-1]),eval_start=str(wide.index[start]),cal_start=str(wide.index[start-36]),bottom_rows=bottom,frequency='MS',selection='predeclared public monthly regional employment hierarchy; no previous Labour experiments found by workspace source/result search'),indent=2))
    print('EXTERNAL_DATA',wide.shape,S.shape,flush=True)
    long=wide.reset_index(names='ds').melt(id_vars='ds',var_name='unique_id',value_name='y').sort_values(['ds','unique_id'])
    for bb,cls in [('PatchTST',PatchTST),('DLinear',DLinear)]:
        for seed in [42,1,2]:
            dest=DEST/f'forecasts/Labour/{bb}/late/{seed}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():continue
            for fold in range(4):
                fd=dest/str(fold);fd.mkdir(exist_ok=True)
                if (fd/'run.json').exists():continue
                ix=start-36+12*fold;count=72 if fold==3 else 12;begin=time.perf_counter()
                cb=ModelCheckpoint(dirpath=str(fd/'checkpoints'),monitor='ptl/val_loss',save_top_k=1)
                opts=dict(h=12,input_size=24,val_check_steps=1,early_stop_patience_steps=50,random_seed=seed,max_steps=500,enable_checkpointing=True,callbacks=[cb],enable_progress_bar=False,enable_model_summary=False,logger=False,accelerator='gpu',devices=1,default_root_dir=str(fd))
                if bb=='PatchTST':opts['n_series']=wide.shape[1]
                nf=NeuralForecast(models=[cls(**opts)],freq='MS');nf.fit(df=long[long.ds<wide.index[ix]],val_size=36)
                cb=next(c for c in nf.models[0].trainer_kwargs['callbacks'] if isinstance(c,ModelCheckpoint));best=cb.best_model_path;assert best
                nf.models=[cls.load_from_checkpoint(best,weights_only=False,early_stop_patience_steps=0,enable_checkpointing=False,callbacks=[],enable_progress_bar=False,enable_model_summary=False,logger=False)]
                frames=[]
                for step in range(0,count,12):
                    j=ix+step;pred=nf.predict(df=long[long.ds<wide.index[j]])
                    if 'unique_id' not in pred.columns:pred=pred.reset_index()
                    pred['origin']=wide.index[j-1];frames.append(pred)
                pred=pd.concat(frames,ignore_index=True)
                assert pd.to_datetime(pred.origin).lt(pd.to_datetime(pred.ds)).all() and len(pred)==count*wide.shape[1]
                pred.to_csv(fd/'predictions.csv',index=False)
                info=dict(dataset='Labour',backbone=bb,seed=seed,fold=fold,prediction_start=str(wide.index[ix]),prediction_end=str(wide.index[ix+count-1]),gradient_train_end=str(wide.index[ix-37]),validation_end=str(wide.index[ix-1]),best_val_loss=float(cb.best_model_score),checkpoint_sha256=sha(Path(best)),seconds=time.perf_counter()-begin)
                (fd/'run.json').write_text(json.dumps(info,indent=2));print('EXTERNAL_FIT',bb,seed,fold,round(info['seconds'],1),flush=True)
                del nf;torch.cuda.empty_cache()
            pred=pd.concat([pd.read_csv(dest/str(i)/'predictions.csv',parse_dates=['ds']) for i in range(4)])
            raw=pred.pivot(index='ds',columns='unique_id',values=bb).reindex(index=wide.index[start-36:],columns=wide.columns).to_numpy()
            assert raw.shape==(108,S.shape[0]) and np.isfinite(raw).all()
            np.savez_compressed(dest/'raw.npz',raw=raw,truth=wide.to_numpy()[start-36:],S=S,start=start,cal_start=start-36)
            (dest/'complete.json').write_text(json.dumps(dict(passed=True,sha256=sha(dest/'raw.npz')),indent=2));print('EXTERNAL_CELL',bb,seed,flush=True)
    print('EXTERNAL_TRAIN_COMPLETE',flush=True)
if __name__=='__main__':main()
