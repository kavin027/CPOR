"""Recompute the grouped-control tables from shareable per-node scores and audit receipts."""
from pathlib import Path
import itertools,json,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ARMS=['Common','Fixed2','Fixed4','Fixed8','Adaptive2','Adaptive4','Adaptive8','Independent']
DIMS=['dataset','backbone','run','protocol']

def analyze(root,write=True):
    f=root/'records';nodes=pd.read_csv(f/'node_rmse.csv.gz');audit=pd.read_csv(f/'arithmetic_audit.csv');timing=pd.read_csv(f/'publication_ms.csv.gz')
    assert nodes.case.nunique()==189 and nodes.dataset.nunique()==7
    assert not nodes.duplicated(['case','node']).any() and len(audit)==189*(len(ARMS)+3)
    safe=audit.query("rule!='Direct'")
    assert safe[['envelope_events','prefix_events']].sum().sum()==0
    assert (audit.query("rule in ['Fixed4','Fixed8','Adaptive2','Adaptive4','Adaptive8','Independent']").max_gap <= 2e-6).all()
    eff=[];node_rows=[];levels=['all','system','region','bottom','store','state','store_category']
    arms=['Full','Reference','Direct',*ARMS]
    for lev in levels:
        d=nodes[nodes.scopes.str.split(';').map(lambda s:lev in s)]
        if d.empty:continue
        means=d.groupby(DIMS+['part'])[arms].mean().groupby(DIMS).mean()
        effects=100*(1-means[ARMS].rdiv(means.Full,axis=0))
        # positive = Full lower, ratio denominator is the matched grouped rule.
        effects['full_vs_reference']=100*(1-means.Full/means.Reference)
        effects['independent_vs_reference']=100*(1-means.Independent/means.Reference)
        effects['independent_vs_fixed2']=100*(1-means.Independent/means.Fixed2)
        z=effects.groupby(['dataset','protocol']).mean().reset_index();z.insert(0,'level',lev);eff.append(z)
    effects=pd.concat(eff,ignore_index=True)
    # A dataset first averages RMSE over train runs/backbones/windows for each named
    # node, after which nodes count once. Exclude only zero denominators in ratios.
    for (ds,prot),g in nodes.groupby(['dataset','protocol']):
        z=g.groupby(['node','name','scopes'])[['Full','Independent','Reference']].mean().reset_index()
        relative=100*(1-z.Full/z.Independent.where(z.Independent>0))
        for level in levels:
            m=z.scopes.str.split(';').map(lambda s:level in s)
            if not m.any():continue
            for arm in ['Full','Independent']:
                r=z.loc[m,arm]/z.loc[m,'Reference'].where(z.loc[m,'Reference']>0)
                node_rows.append(dict(dataset=ds,protocol=prot,level=level,rule=arm,nodes=m.sum(),ref_zero=(z.loc[m,'Reference']==0).sum(),median_to_reference_ratio=r.median(),worst_to_reference_ratio=r.max(),worst_node=(z.loc[r.idxmax(),'name'] if r.notna().any() else None),full_node_wins=(z.loc[m,'Full']<z.loc[m,'Independent']).sum(),full_node_losses=(z.loc[m,'Full']>z.loc[m,'Independent']).sum(),ties=(z.loc[m,'Full']==z.loc[m,'Independent']).sum(),median_full_gain=relative[m].median(),worst_full_gain=relative[m].min()))
    main=effects.query("level=='all'")
    equal=main.drop(columns=['dataset','level']).groupby('protocol').mean()
    test={}
    for key,g in main.groupby('protocol'):
        v=g.Independent.to_numpy()
        test[key]=dict(n_datasets=len(v),datasets_favor_full=int((v>0).sum()),mean=float(v.mean()),median=float(np.median(v)),min_leave_one_out=float(min((v.sum()-x)/(len(v)-1) for x in v)),max_leave_one_out=float(max((v.sum()-x)/(len(v)-1) for x in v)),descriptive_signflip_two_sided=float(np.mean([abs((np.array(signs)*v).mean())>=abs(v.mean())-1e-12 for signs in itertools.product([-1,1],repeat=len(v))])))
    tm=timing.groupby('rule').ms.agg(mean_ms='mean',median_ms='median',p95_ms=lambda x:x.quantile(.95),issues='size')
    stat=audit.groupby('rule').agg({'case':'count','node_dates':'sum','envelope_events':'sum','prefix_events':'sum','node_sequences_ever':'sum','zeros':'sum','targets':'sum','retries':'sum','guarded':'sum','anchors_on_failure':'sum','max_gap':'max','max_guard':'max'})
    stat.loc[['Reference','Direct','Common','Fixed2','Full'],['zeros','targets','retries','guarded','anchors_on_failure','max_gap','max_guard']]=np.nan
    tab={'dataset_effects.csv':effects,'node_reference.csv':pd.DataFrame(node_rows),'primary_summary.csv':equal.reset_index(),'timing.csv':tm.reset_index(),'audits.csv':stat.reset_index()}
    # Omit both windows and continuous counterparts for the eight historic
    # PatchTST fits whose late-window checkpoint identity was not recovered.
    absent=(nodes.dataset.isin(['RH','ECommerce','Traffic','Wiki2']) &
            nodes.backbone.eq('PatchTST') & nodes.run.isin([1,2]))
    complete=nodes.loc[~absent].groupby(DIMS+['part'])[arms].mean().groupby(DIMS).mean()
    assert len(complete)==110
    sub=100*(1-complete[ARMS].rdiv(complete.Full,axis=0))
    # Retain equal backbone weights when two PatchTST runs are omitted in a cell.
    sub=sub.groupby(['dataset','backbone','protocol']).mean().groupby(['dataset','protocol']).mean().groupby('protocol').mean()
    assert np.isclose(sub.loc['continuous','Fixed2'],3.2762,atol=0.0001,rtol=0)
    assert np.isclose(sub.loc['window','Fixed2'],2.7162,atol=0.0001,rtol=0)
    tab['verified_subset.csv']=sub.reset_index()
    tmfile=f/'matched_time.csv.gz'
    if tmfile.exists():
        mt=pd.read_csv(tmfile)
        assert len(mt)==2016 and mt.groupby('rule').size().eq(1008).all()
        assert mt.max_abs_reproduction_difference.max()<=2e-6
        timed=mt.groupby(['dataset','rule']).ms.agg(mean_ms='mean',p95_ms=lambda x:x.quantile(.95),releases='size').reset_index()
        alltime=mt.groupby('rule').ms.agg(mean_ms='mean',p95_ms=lambda x:x.quantile(.95),releases='size').reset_index()
        alltime.insert(0,'dataset','POOLED')
        tab['matched_time.csv']=pd.concat([timed,alltime],ignore_index=True)
    if write: (root/'summary').mkdir(exist_ok=True)
    for name,d in tab.items():
        if write:d.to_csv(root/'summary'/name,index=False)
        else:pd.testing.assert_frame_equal(pd.read_csv(root/'summary'/name),d,check_dtype=False,rtol=1e-10,atol=1e-12)
    output=root/'summary/primary_uncertainty.json'
    if write:output.write_text(json.dumps(test,indent=2))
    else:assert json.loads(output.read_text())==test
    print(equal.T.to_string(),test,flush=True)
    print(main[['dataset','protocol','Independent']].to_string(index=False))
    print(stat[['case','node_dates','retries','guarded','max_gap','max_guard']])
    print(tm)
if __name__=='__main__':analyze(HERE,'--check' not in sys.argv)


