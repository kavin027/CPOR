"""Rebuild box counts, signed RMSE contributions and equal-dataset summaries."""
from pathlib import Path
import numpy as np,pandas as pd,sys
W=Path(__file__).resolve().parent
def export(d,n,check):
    if check:pd.testing.assert_frame_equal(d.reset_index(drop=True),pd.read_csv(W/n),check_dtype=False,atol=1e-10,rtol=1e-10)
    else:d.to_csv(W/n,index=False)
def main(check):
    d=pd.read_csv(W/'node_geometry.csv.gz',float_precision='round_trip');i=pd.read_csv(W/'issue_geometry.csv.gz',float_precision='round_trip')
    keys=['dataset','protocol','backbone','run','case'];c=d.groupby(keys).mean(numeric_only=True)
    assert len(c)==189 and len(i)==18144 and i.groupby('case').size().isin([72,144]).all()
    assert not i.duplicated(['case','issue']).any()
    # Additive costs use the online Full and Independent node-norm denominators.
    assert np.allclose(d.Independent-d.Full,d.out_issue_delta+d.inside_issue_delta,atol=2e-8,rtol=1e-9)
    assert np.allclose(d.out_local+d.out_state,d.out_issue_delta,atol=2e-8,rtol=1e-9)
    assert np.allclose(d.in_local+d.in_state,d.inside_issue_delta,atol=2e-8,rtol=1e-9)
    for stamp,col in [('delta_rmse','out_issue_delta'),('matched_action_contribution','out_local'),('change_of_state_contribution','out_state')]:
        actual=i[i.box.str.startswith('out_')].groupby('case')[stamp].sum()
        expected=c.reset_index().set_index('case')[col]
        assert np.allclose(actual.reindex(expected.index,fill_value=0),expected,atol=1e-9)
    c['real_gain']=c.Independent-c.Full
    c['matched_rmse_gain']=c.SameState-c.Full
    c['state_rmse_gain']=c.Independent-c.SameState
    effects=['real_gain','matched_rmse_gain','state_rmse_gain','out_issue_delta','inside_issue_delta','observed_out_delta','out_local','in_local','out_state','in_state']
    # The denominator and both windows' errors are averaged prior to the ratio.
    c=c.groupby(keys[:-1]).mean()
    rel=c[effects].div(c.Independent,axis=0)*100
    m=rel.groupby(['dataset','protocol']).mean().reset_index()
    eq=m.drop(columns='dataset').groupby('protocol').mean().reset_index()
    tags=d[keys].drop_duplicates()
    i=i.merge(tags,on='case',validate='many_to_one')
    i['outside_event']=i.box.str.startswith('out_')
    i['joint_only']=i.box.eq('out_joint')
    i['unresolved']=i.box.eq('unresolved')
    i['witness_out']=i.latent_witness_out>0
    i['alternate_witness_feasible']=i.witness_out & i.box.str.startswith('in_')
    i['actions_differ']=i.same_state_max_in_tolerances>1
    i['weighted_distance_full_worse']=(i.full_distance-i.matched_independent_distance)>1e-6*(1+i.matched_independent_distance)
    groups=['dataset','protocol']
    counts=i.groupby(groups).agg(issues=('case','size'),outside=('outside_event','sum'),joint_only=('joint_only','sum'),unresolved=('unresolved','sum'),witness_out=('witness_out','sum'),alternative_feasible=('alternate_witness_feasible','sum'),max_abs_action_exceeds_tolerance=('actions_differ','sum'),full_distance_worse_over_tol=('weighted_distance_full_worse','sum'),node_below=('observed_low','sum'),node_above=('observed_high','sum'),total_node_issues=('observed_nodes','sum')).reset_index()
    ct=d.groupby(groups)[['full_better_at_out','full_worse_at_out','near_equal_at_out']].sum().reset_index()
    counts=counts.merge(ct,on=groups)
    assert ((counts.node_below+counts.node_above)==(counts.full_better_at_out+counts.full_worse_at_out+counts.near_equal_at_out)).all()
    bins=[]
    for k in ['all','system','region','bottom','store','state','store_category']:
        sub=d[d.scopes.str.split(';').map(lambda s:k in s)]
        if sub.empty:continue
        z=sub.groupby(groups).agg(nodes=('node','nunique'),node_issues=('T','sum'),below=('node_below','sum'),above=('node_above','sum'),full_better_at_out=('full_better_at_out','sum'),full_worse_at_out=('full_worse_at_out','sum'),near_equal_at_out=('near_equal_at_out','sum')).reset_index()
        z['level']=k;bins.append(z)
    lev=pd.concat(bins,ignore_index=True)
    pernode=d.groupby(groups+['node','name','scopes']).agg(T=('T','sum'),observed_outs=('node_below','sum'),above=('node_above','sum'),full_wins_on_exits=('full_better_at_out','sum'),full_losses_on_exits=('full_worse_at_out','sum'),full_node_rmse=('Full','mean'),independent_node_rmse=('Independent','mean'),delta_on_exit_dates=('out_issue_delta','mean'),delta_at_exit_coordinate=('observed_out_delta','mean')).reset_index()
    pernode['observed_outs']+=pernode.above;pernode['exit_rate']=pernode.observed_outs/pernode['T']
    for df,f in [(counts,'geometry_counts.csv'),(m,'decomposition.csv'),(eq,'decomposition_equal.csv'),(lev,'geometry_levels.csv'),(pernode,'geometry_nodes.csv')]:export(df,f,check)
    print(counts.to_string(index=False))
    print(eq.to_string(index=False))
    print(m[['dataset','protocol','real_gain','matched_rmse_gain','out_issue_delta','observed_out_delta']].to_string(index=False))
if __name__=='__main__':main('--check' in sys.argv)

