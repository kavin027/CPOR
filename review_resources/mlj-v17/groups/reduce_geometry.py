"""Reduce original trajectory replays to shareable issue and node statistics."""
from pathlib import Path
import json,csv
import numpy as np
import pandas as pd
W=Path(__file__).resolve().parent;ROOT=W.parents[2]
A=ROOT/'output/cpor_unified_protocol_duet_20261108'
INDEX=ROOT/'output/cpor_mlj_v15_20261115/kit/inputs.csv'
def main():
    rows=list(csv.DictReader(INDEX.open(encoding='utf-8-sig')))
    nodemap=pd.read_csv(A/'node_map.csv');nodes=[];issues=[];markers=[]
    examples=set();E=W/'examples';E.mkdir(exist_ok=True)
    for r in rows:
        case=r['case'];ds=r['dataset'];f=W/'scratch'/(case+'.npz')
        with np.load(f) as o:
            h=o['classes'];b=o['below'];a=o['above'];wh=o['wit'];outside=np.char.startswith(h,'out_')
            d=o['contrib'];local=o['local'];state=o['state'];obj=o['obj'];T,n=d.shape
            uz=o['full'];ui=o['ind'];us=o['samestate']
            if ds=='AEMO':names=['System','NSW1','QLD1','SA1','TAS1','VIC1'];scopes=['all;system']+['all;region']*5
            else:
                z=nodemap[nodemap.dataset==ds].sort_values('node');names=z.name.to_list();scopes=z.scopes.to_list()
            for i,name in enumerate(names):
                c=b[:,i]|a[:,i];delta=d[:,i]
                nodes.append(dict(case=case,dataset=ds,protocol=r['protocol'],part=r['part'],backbone=r['backbone'],run=r['run'],node=i,name=name,scopes=scopes[i],Full=uz[i],Independent=ui[i],SameState=us[i],T=T,
                 node_below=b[:,i].sum(),node_above=a[:,i].sum(),out_issue=(outside).sum(),
                 near_equal_at_out=np.count_nonzero(c&(delta==0)),full_better_at_out=np.count_nonzero(c&(delta>0)),full_worse_at_out=np.count_nonzero(c&(delta<0)),
                 observed_out_delta=delta[c].sum(),out_issue_delta=delta[outside].sum(),inside_issue_delta=delta[~outside].sum(),
                 out_local=local[outside,i].sum(),in_local=local[~outside,i].sum(),out_state=state[outside,i].sum(),in_state=state[~outside,i].sum()))
            for t in range(T):
                issues.append(dict(case=case,issue=t+1,box=h[t],latent_witness_out=int(wh[t].sum()),observed_low=int(b[t].sum()),observed_high=int(a[t].sum()),observed_nodes=n,latent_components=wh.shape[1],full_distance=obj[t,0],matched_independent_distance=obj[t,1],same_state_max_in_tolerances=o['actions'][t],delta_rmse=d[t].mean(),matched_action_contribution=local[t].mean(),change_of_state_contribution=state[t].mean()))
                category=h[t] if wh[t].any() else h[t]+'_witness_inside'
                if category.startswith('in_') and not wh[t].any():continue
                key=(ds,category)
                if key not in examples:
                    examples.add(key)
                    with np.load(A/r['context_archive']) as x,np.load(A/r['PIRCoherent_archive']) as z,np.load(A/r['PIRFull_archive']) as pf:
                        low=np.minimum(x['ub'][36+t],z['witness'][t]);high=np.maximum(x['ub'][36+t],z['witness'][t]);p=pf['pred'][t];w=pf['witness'][t]
                        e=f'{ds}_{category}.npz'; np.savez_compressed(E/e,S=x['S'],low=low,high=high,p=p,w=w)
                        markers.append(dict(dataset=ds,case=case,issue=t+1,box=h[t],latent_witness_out=int(wh[t].sum()),node_low=int(b[t].sum()),node_high=int(a[t].sum()),file=e))
    pd.DataFrame(nodes).to_csv(W/'node_geometry.csv.gz',index=False)
    pd.DataFrame(issues).to_csv(W/'issue_geometry.csv.gz',index=False)
    pd.DataFrame(markers).to_csv(E/'index.csv',index=False)
if __name__=='__main__':main()

