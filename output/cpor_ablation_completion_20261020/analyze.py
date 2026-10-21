from worker import *
import pandas as pd
idx=json.loads((H/'RESULT_INDEX.json').read_text());rows=[];hashes={};audits=[]
for item in idx:
 p=Path(item['path']);z=load(p);d=source(cells[item['cell']],item['seed']);x=load(d/'endpoints.npz');y=x['truth'][36:];pred=z['pred']
 assert pred.shape==y.shape and np.isfinite(pred).all();hashes[str(p)]=sha(p)
 rows.append({k:item[k] for k in ['cell','dataset','backbone','window','seed','arm','reused']}|dict(rmse=float(np.sqrt(np.mean((pred-y)**2,axis=0)).mean())))
 if not item['reused']:
  j=json.loads(p.with_suffix('.json').read_text());assert j['sha256']==sha(p) and j['registration_sha256']==sha(H/'REGISTRATION.json')
  audits.append({k:item[k] for k in ['cell','seed','arm']}|{k:j[k] for k in ['seconds','failed_publications','checks']})
f=pd.DataFrame(rows);assert len(f)==264
f.to_csv(H/'CELL_METRICS.csv',index=False)
g=f.groupby(['dataset','backbone','seed','arm'],as_index=False).rmse.mean();g.to_csv(H/'GROUP_EXPERIMENT_METRICS.csv',index=False)
v=g.groupby(['dataset','backbone','arm']).agg(mean_rmse=('rmse','mean'),sd_rmse=('rmse','std'),repeats=('rmse','size')).reset_index();assert (v.repeats==3).all();v.to_csv(H/'THREE_EXPERIMENT_RMSE.csv',index=False)
old=pd.read_csv(O/'cpor_table7_revision_20261020/THREE_EXPERIMENT_RMSE.csv')
for arm,method in [('RootFull','CPOR'),('SquareFull','SquareFull')]:
 q=v[v.arm==arm].merge(old[old.method==method],on=['dataset','backbone']);assert len(q)==12;np.testing.assert_allclose(q.mean_rmse_x,q.mean_rmse_y,rtol=1e-12)
comparisons=[('root_vs_square_full','RootFull','SquareFull'),('root_vs_square_directional','RootDirectional','SquareDirectional'),('full_vs_directional_root','RootFull','RootDirectional'),('full_vs_directional_square','SquareFull','SquareDirectional')]
summary={};gs=[]
for name,arm,comp in comparisons:
 a=g[g.arm==arm].merge(g[g.arm==comp],on=['dataset','backbone','seed'],suffixes=('_method','_comparator'));a['gain_pct']=100*(1-a.rmse_method/a.rmse_comparator);a['comparison']=name
 ag=a.groupby(['dataset','backbone']).gain_pct.mean();ex=a.groupby('seed').gain_pct.mean()
 summary[name]=dict(mean_gain_pct=float(a.gain_pct.mean()),median_group_gain_pct=float(ag.median()),positive_groups=int((ag>0).sum()),positive_all_three=int((a.groupby(['dataset','backbone']).gain_pct.min()>0).sum()),experiment_means={str(k):float(x) for k,x in ex.items()},experiment_sd=float(ex.std()),m5_mean=float(a[a.dataset=='M5'].gain_pct.mean()))
 gs.append(a[['dataset','backbone','seed','comparison','gain_pct']])
pd.concat(gs).to_csv(H/'PAIRED_GAINS.csv',index=False)
(H/'SUMMARY.json').write_text(json.dumps(dict(configurations=22,experiments=3,arms=4,trajectories=264,reused=168,new=96,debug_gate_replays=2,cpor_and_square_match_table7=True,comparisons=summary,failed_publications=sum(x['failed_publications'] for x in audits)),indent=2))
(H/'INPUT_SHA256.json').write_text(json.dumps(json.loads((H/'REGISTRATION.json').read_text())['inputs'],indent=2))
(H/'OUTPUT_SHA256.json').write_text(json.dumps(hashes,indent=2));(H/'NUMERICAL_AUDITS.json').write_text(json.dumps(audits,indent=2));print((H/'SUMMARY.json').read_text())
