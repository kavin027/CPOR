from worker import *
import shutil,datetime
P=O/'kais_presentation_revision_20261018';M=O/'rr_recap_policy_bound_20261006/Policy_Bound.tex';S=P/'supplement/Supplementary_Note_S1.tex'
if not (H/'PAPER_BACKUP.txt').exists():
 B=R/'archives/research_backups'/('table8_complete_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));B.mkdir(parents=True)
 for p in [M,S,P/'CPOR_DELIVERY.json']:shutil.copy2(p,B/p.name)
 figs=[P/'figures/Fig1_CPOR_rich.svg',P/'figures/Fig1_CPOR_rich.pdf'];(B/'figure_hashes.json').write_text(json.dumps([dict(Path=str(p),Hash=sha(p).upper()) for p in figs]))
 (H/'PAPER_BACKUP.txt').write_text(str(B))
exp.registration()
inputs={};index=[];jobs=[]
for ci,c in enumerate(cells):
 for seed in [42,1,2]:
  d=source(c,seed)
  for n in ['endpoints.npz','contexts.npz','C0.npz','SquareFull.npz']:inputs[str(d/n)]=sha(d/n)
  inputs[str(snapshot(c,seed))]=sha(snapshot(c,seed))
  for arm,name in [('RootFull','C0'),('SquareFull','SquareFull')]:index.append(dict(cell=ci,dataset=c['dataset'],backbone=c['backbone'],window=c['window'],seed=seed,arm=arm,path=str(d/(name+'.npz')),reused=True))
  if seed==42 and c['dataset']!='M5':
   x,D,e,recs,k=context(c,seed);xo,Do,eo,ro,ko=exp.contexts((c['dataset'],c['backbone'],c['window']))
   for key in ['truth','S','v','b','ub']:np.testing.assert_array_equal(x[key],xo[key])
   np.testing.assert_array_equal(D,Do);np.testing.assert_array_equal(e,eo);assert k==ko
   for ra,rb in zip(recs,ro):
    for key in ra:np.testing.assert_array_equal(ra[key],rb[key])
  for arm in ['RootDirectional','SquareDirectional']:
   reuse=seed==42 and c['dataset']!='M5'
   p=(O/'c0_matched_ablation_20261014/results' if reuse else H/'results')/c['dataset']/c['backbone']/c['window']/str(seed)/(arm+'.npz')
   if reuse:
    j=json.loads(p.with_suffix('.json').read_text());assert sha(p)==j['output_sha256'];inputs[str(p)]=sha(p)
   else:jobs.append([ci,seed,arm])
   index.append(dict(cell=ci,dataset=c['dataset'],backbone=c['backbone'],window=c['window'],seed=seed,arm=arm,path=str(p),reused=reuse))
assert len(index)==264 and len(jobs)==96
sources={str(p):sha(p) for p in [H/'worker.py',Path(exp.__file__),Path(exp.square.__file__),Path(exp.root_kernel.__file__),Path(exp.base.__file__)]}
reg=dict(sources=sources,inputs=inputs,jobs=jobs,required=264,reused=168,new=96,repetitions=[42,1,2],calibration='same archived first 36 C0 snapshots; live allowances/errors reset before 72 scoring releases',evaluation='mean-node RMSE; average windows within dataset/backbone/repetition, then average three repetitions; paired percentage gains calculated within repetition before averaging',retrospective=True)
if (H/'REGISTRATION.json').exists():assert json.loads((H/'REGISTRATION.json').read_text())==reg
else:(H/'REGISTRATION.json').write_text(json.dumps(reg,indent=2))
(H/'RESULT_INDEX.json').write_text(json.dumps(index,indent=2));print('REGISTERED',len(index),'REUSE',168,'NEW',len(jobs))

