from pathlib import Path
import sys,importlib.util,json,hashlib,datetime,argparse,zipfile,time,logging
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;O=HERE.parent
spec=importlib.util.spec_from_file_location('frozen_confirmation_driver',O/'c0_frozen_confirmation_20261014/run.py')
fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
sha=fresh.sha;CELLS=fresh.CELLS

def registration():
    p=HERE/'REGISTRATION.json';sources=dict(json.loads((O/'c0_frozen_confirmation_20261014/REGISTRATION.json').read_text())['sources'])
    for f in [Path(__file__),HERE/'acquire.py',HERE/'PROTOCOL.md',HERE/'external_models.py']:sources[str(f.resolve())]=sha(f)
    for f,h in sources.items():assert sha(f)==h,f
    r=dict(sources=sources,cells=CELLS,seed=42,archive_git_blob='925e92ee03cdb83c54a1dbbc12260a3b1939753c',
        native_training_days=1941,windows={'early':[1798,1869],'late':[1870,1941]},nodes=44,bottom=30,
        principal_groups=2,original_M5_competition_protocol=False,source_blind_at_registration=True,
        prior_transport_failure_sha256=sha(O/'c0_frozen_confirmation_20261014/SOURCE_BLOCKER.json'))
    r=json.loads(json.dumps(r))
    if p.exists():old=json.loads(p.read_text());r['registered_utc']=old['registered_utc'];assert r==old
    else:
        assert not (HERE/'m5.zip').exists();r['registered_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(r,indent=2))
    return p

def data():
    reg=registration();p=HERE/'m5.zip';receipt=json.loads((HERE/'SOURCE_RECEIPT.json').read_text());assert sha(p)==receipt['archive_sha256']
    with zipfile.ZipFile(p) as z:
        names=z.namelist();sales=next(n for n in names if n.endswith('/sales_train_evaluation.csv') or n=='sales_train_evaluation.csv')
        cal=next(n for n in names if n.endswith('/calendar.csv') or n=='calendar.csv')
        calendar=pd.read_csv(z.open(cal));mapping=calendar.set_index('d')['date']
        metadata=['state_id','store_id','cat_id'];days=[f'd_{i}' for i in range(1,1942)];sums={};geography={};count=0
        for f in pd.read_csv(z.open(sales),usecols=metadata+days,chunksize=512):
            values=f[days].to_numpy(dtype='float64');assert np.isfinite(values).all() and values.min()>=0
            for (state,store,cat),g in f.groupby(metadata,sort=False):
                key=store+'/'+cat;assert store not in geography or geography[store]==state;geography[store]=state
                s=g[days].to_numpy(dtype='float64').sum(0);sums[key]=sums.get(key,np.zeros(1941))+s
            count+=len(f)
    stores=sorted(geography);states=sorted(set(geography.values()));bottom=sorted(sums)
    assert len(stores)==10 and states==['CA','TX','WI'] and len(bottom)==30 and count==30490
    assert all(sum(k.startswith(s+'/') for k in bottom)==3 for s in stores)
    rows=[np.ones(30)];nodes=['SYSTEM']
    for state in states:rows.append(np.array([geography[k.split('/')[0]]==state for k in bottom],dtype=float));nodes.append('state/'+state)
    for store in stores:rows.append(np.array([k.startswith(store+'/') for k in bottom],dtype=float));nodes.append('store/'+store)
    rows.extend(np.eye(30));nodes.extend('store_category/'+k for k in bottom);S=np.asarray(rows)
    dates=pd.to_datetime(mapping.reindex(days).to_numpy());values=np.stack([sums[k] for k in bottom],axis=1)@S.T
    wide=pd.DataFrame(values,index=dates,columns=nodes);wide.index.name='ds';assert wide.index.is_monotonic_increasing and not wide.isna().any().any()
    assert (wide.index[1:]-wide.index[:-1]==pd.Timedelta(days=1)).all()
    wide.to_csv(HERE/'observed.csv');np.save(HERE/'S.npy',S)
    (HERE/'DATA_PROVENANCE.json').write_text(json.dumps(dict(archive_sha256=sha(p),raw_item_store_series=count,bottom_labels=bottom,
        nodes=nodes,geography=geography,observed_sha256=sha(HERE/'observed.csv'),S_sha256=sha(HERE/'S.npy'))))
    return wide,S

def forecast(wide,S,bb,w):
    dest=HERE/'results'/bb/w/'42';dest.mkdir(parents=True,exist_ok=True);p=dest/'raw.npz';start=1797 if w=='early' else 1869;begin=start-144;end=start+72;n=len(wide.columns)
    if p.exists():
        j=json.loads(p.with_suffix('.json').read_text());assert j['sha256']==sha(p);return dest,dict(np.load(p)),start
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST,DLinear
    from pytorch_lightning.callbacks import ModelCheckpoint
    import torch
    torch.set_num_threads(1);logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    cls=PatchTST if bb=='PatchTST' else DLinear;cb=ModelCheckpoint(dirpath=str(dest/'checkpoint'),monitor='ptl/val_loss',save_top_k=1)
    opts=dict(h=12,input_size=24,val_check_steps=1,early_stop_patience_steps=50,random_seed=42,max_steps=500,
        enable_checkpointing=True,callbacks=[cb],enable_progress_bar=False,enable_model_summary=False,logger=False,
        accelerator='gpu',devices=1,default_root_dir=str(dest))
    if bb=='PatchTST':opts['n_series']=n
    model=cls(**opts);nf=NeuralForecast(models=[model],freq='D')
    df=wide.reset_index().melt(id_vars='ds',var_name='unique_id',value_name='y').sort_values(['ds','unique_id'])
    nf.fit(df=df[df.ds<wide.index[begin]],val_size=36)
    cb=next(c for c in nf.models[0].trainer_kwargs['callbacks'] if isinstance(c,ModelCheckpoint));best=cb.best_model_path;assert best
    nf.models=[cls.load_from_checkpoint(best,weights_only=False,early_stop_patience_steps=0,enable_checkpointing=False,callbacks=[],enable_progress_bar=False,enable_model_summary=False,logger=False)]
    predictions=[]
    for i in range(begin,end,12):
        z=nf.predict(df=df[df.ds<wide.index[i]])
        if 'unique_id' not in z:z=z.reset_index()
        z['origin']=wide.index[i-1];assert (z.ds>z.origin).all();predictions.append(z)
    f=pd.concat(predictions);assert not f.duplicated(['unique_id','ds']).any();f.to_csv(dest/'base_forecasts.csv',index=False)
    raw=f.pivot(index='ds',columns='unique_id',values=bb).reindex(index=wide.index[begin:end],columns=wide.columns).to_numpy();assert raw.shape==(216,n) and np.isfinite(raw).all()
    np.savez_compressed(p,raw=raw,truth=wide.to_numpy()[begin:end],S=S,begin=begin,cal_start=start-36,start=start,dates=wide.index[begin:end].astype(str).to_numpy())
    p.with_suffix('.json').write_text(json.dumps(dict(sha256=sha(p),registration_sha256=sha(HERE/'REGISTRATION.json'),checkpoint=best,
        checkpoint_sha256=sha(best),last_base_fit_label=str(wide.index[begin-1]),first_forecast=str(wide.index[begin]),first_scored=str(wide.index[start]),best_val=float(cb.best_model_score)),indent=2))
    del nf,model;torch.cuda.empty_cache();print('M5_BASE_READY',bb,w,flush=True)
    return dest,dict(np.load(p)),start

def external(dest,wide,S,stream,start):
    from external_models import external as common
    common(dest,wide,S,stream,start,fresh)

def main():
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--cell',type=int);args=p.parse_args();r=registration()
    if args.register:print('M5_FROZEN',sha(r),flush=True);return
    fresh.HERE=HERE;fresh.registration=registration;wide,S=data()
    for ci in ([args.cell] if args.cell is not None else range(4)):
        bb,w=CELLS[ci];dest,stream,start=forecast(wide,S,bb,w);x,D,expert=fresh.prepare_context(wide,S,stream['raw'],start)
        np.savez_compressed(dest/'endpoints.npz',**x);np.savez_compressed(dest/'contexts.npz',D=D,expert=expert)
        fresh.protected(dest,x,D,expert)
        external(dest,wide,S,stream,start)
    (HERE/'RUN_STATUS.json').write_text(json.dumps(dict(required=4,completed=sum(all((HERE/'results'/b/w/'42'/(a+'.npz')).exists() for a in ['C0','SquareFull','CovFullFix','ProCAST72Warm','PIR']) for b,w in CELLS))))

if __name__=='__main__':main()
