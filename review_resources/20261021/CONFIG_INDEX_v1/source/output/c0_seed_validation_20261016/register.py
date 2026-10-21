from common import *
import datetime
sources={}
prior=json.loads((base.HERE/'PREREGISTRATION_PAIRED_ROOT.json').read_text())
for p,h in prior['source_sha256'].items():assert sha(p)==h,p;sources[p]=h
for p in [*H.glob('*.py'),H/'PROTOCOL.txt',O/'c0_frozen_confirmation_20261014/run.py',O/'c0_matched_ablation_20261014/experiment.py',O/'c0_matched_ablation_20261014/square_learning.py',O/'rr_recap_trajectory_learning_20261003/history.py',O/'rr_recap_strong_baseline_20261006/run.py',O/'rr_recap_evidence_extension_20261002/procast_author.py',O/'rr_recap_evidence_extension_20261002/train.py',O/'rr_recap_evidence_extension_20261002/major_revision_20261002/external_train.py']:
    sources[str(p.resolve())]=sha(p)
inputs={};audit=[]
for c in cells:
    wide=pd.read_csv(datafile(c),parse_dates=['ds']).set_index('ds');inputs[str(datafile(c))]=sha(datafile(c))
    if c['family']!='development':continue
    x42=load(endpoint(c,42))
    for seed in [42,1,2]:
        ep=endpoint(c,seed);rp=rawfile(c,seed);raw=load(rp);x=load(ep);start=int(raw['cal_start'])
        for k in ['truth','S']:np.testing.assert_array_equal(x[k],x42[k])
        np.testing.assert_array_equal(x['truth'],wide.iloc[start:start+108].to_numpy());np.testing.assert_array_equal(raw['raw'],x['raw'])
        assert str(wide.index[start+36])==c['score_start'] and str(wide.index[start+107])==c['score_end']
        cp=rp.with_name('complete.json');meta=json.loads(cp.read_text())
        for p in [ep,rp,cp]:inputs[str(p)]=sha(p)
        if 'reused' in meta:
            source=Path(meta['reused']);assert sha(source)==meta['sha256'];inputs[str(source)]=sha(source)
            np.testing.assert_array_equal(load(source)['raw'],raw['raw'])
            receipt='legacy_reused_source_hash_verified; historical schedule reconstructed, not new execution proof'
        else:
            expected=meta.get('raw_sha256',meta.get('sha256'));assert expected==sha(rp)
            for fold in range(4):
                fp=rp.parent/str(fold)/'run.json';fm=json.loads(fp.read_text());inputs[str(fp)]=sha(fp)
                assert fm['seed']==seed and fm['backbone']==c['backbone']
                ix=start+12*fold;assert pd.Timestamp(fm['gradient_train_end'])==wide.index[ix-37] and pd.Timestamp(fm['validation_end'])==wide.index[ix-1]
                assert pd.Timestamp(fm['prediction_start'])==wide.index[ix]
            receipt='four original fit receipts: seed, train/validation/prediction cutoffs verified'
        rel=f"{c['dataset']}/{c['backbone']}/{c['window']}/{seed}"
        for name in ['bank.npz','CovFull.npz','CovFull.json']:
            p=O/'rr_recap_error_dependence_20261003/results'/rel/name;inputs[str(p)]=sha(p)
        p=O/'rr_recap_error_dependence_20261003/results'/rel/'CovFull.npz';meta=json.loads(p.with_suffix('.json').read_text());assert meta['input_sha256']==sha(ep) and meta['output_sha256']==sha(p)
        for name,h in meta['source_sha256'].items():assert sha(name)==h,name
        p=O/'rr_recap_relative_dominance_20261003/results_v2'/rel/'CovFullFix.npz';meta=json.loads(p.with_suffix('.json').read_text());assert meta['input_sha256']==sha(ep) and meta['output_sha256']==sha(p)
        inputs[str(p)]=sha(p);inputs[str(p.with_suffix('.json'))]=sha(p.with_suffix('.json'))
        audit.append(dict(**key(c),seed=seed,provenance=receipt,truth_S_dates_match=True,raw_differs_from42=not np.array_equal(x['raw'],x42['raw'])))
pd.DataFrame(audit).to_csv(H/'REUSE_PROVENANCE.csv',index=False)
reg=dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,inputs=inputs,seeds=[42,1,2],cells=[key(c) for c in cells],new_training='M5 8 fits plus only necessary seed-matched historical support folds',new_methods=False,retrospective=True)
p=H/'REGISTRATION.json';assert not p.exists(),'registration is immutable'
js(p,reg);print('REGISTERED',len(audit),'streams',len(inputs),'inputs',len(sources),'sources')
