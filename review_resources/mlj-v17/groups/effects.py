"""Seven-dataset contrasts from the original (unrounded) paired-node scores."""
from pathlib import Path
import json,sys,itertools
import pandas as pd
import numpy as np
W=Path(__file__).resolve().parent
UNITS={'RH':'unspecified original unit','ECommerce':'items','Wiki2':'page views','Labour':'thousand persons','Traffic':'dimensionless','AEMO':'MW','M5':'items'}
def write_or_check(d,name,check):
    f=W/name
    if check:pd.testing.assert_frame_equal(d.reset_index(drop=True),pd.read_csv(f),check_dtype=False,atol=1e-11,rtol=1e-11)
    else:d.to_csv(f,index=False)
def main(source,check):
    a=pd.read_csv(source/'node_geometry.csv.gz',float_precision='round_trip')
    keep=['Full','Independent'];g=a.groupby(['dataset','protocol','backbone','run','part'])[keep].mean().groupby(['dataset','protocol','backbone','run']).mean()
    g['delta_original_units']=g.Independent-g.Full;g['gain_pct']=100*(1-g.Full/g.Independent)
    z=g.groupby(['dataset','protocol']).mean().reset_index();z['unit']=z.dataset.map(UNITS)
    rows=[]; sig=[]
    signs=np.array(list(itertools.product([-1.,1.],repeat=7)))
    idx=np.indices((7,)*7,dtype='uint8').reshape(7,-1).T
    for protocol,grp in z.groupby('protocol'):
        ef=grp.gain_pct.to_numpy()
        means=ef[idx].mean(1);lo,hi=np.quantile(means,[.025,.975])
        p=(abs(signs@ef/7)>=abs(ef.mean())-1e-12).mean()
        sig.append(dict(protocol=protocol,mean=ef.mean(),median=np.median(ef),min=ef.min(),max=ef.max(),positive=sum(ef>0),n=7,percentile025=lo,percentile975=hi,signflip_p=p,LOO_min=min((ef.sum()-ef)/6),LOO_max=max((ef.sum()-ef)/6)))
    sig=pd.DataFrame(sig);ordered=np.argsort(sig.signflip_p.to_numpy());correct=np.maximum.accumulate(np.minimum(1,sig.signflip_p.to_numpy()[ordered]*np.arange(2,0,-1)))
    sig.loc[ordered,'two_protocol_holm_p']=correct
    for (ds,prot),d in a.groupby(['dataset','protocol']):
        m=d.groupby(['node','name','scopes'])[keep].mean().reset_index()
        for lev in ['all','system','region','bottom','store','state']:
            sub=m[m.scopes.str.split(';').map(lambda s:lev in s)]
            if len(sub)==0:continue
            e=100*(1-sub.Full/sub.Independent.where(sub.Independent>0));ix=e.idxmin()
            rows.append(dict(dataset=ds,protocol=prot,level=lev,nodes=len(sub),wins=sum(sub.Full<sub.Independent),losses=sum(sub.Full>sub.Independent),ties=sum(sub.Full==sub.Independent),nonzero_denominators=e.notna().sum(),median_node_gain=e.median(),min_node_gain=e.min(),max_node_gain=e.max(),worst_node=(sub.loc[ix,'name'] if e.notna().any() else None)))
    mt=pd.read_csv(source/'matched_time.csv.gz',float_precision='round_trip')
    tm=mt.groupby(['dataset','rule']).ms.agg(mean_ms='mean',P95_ms=lambda s:s.quantile(.95),issues='size').reset_index()
    pooled=mt.groupby('rule').ms.agg(mean_ms='mean',P95_ms=lambda s:s.quantile(.95),issues='size').reset_index();pooled.insert(0,'dataset','Pooled')
    for d,n in [(z,'dataset_effects.csv'),(sig,'uncertainty.csv'),(pd.DataFrame(rows),'node_effects.csv'),(pd.concat([tm,pooled],ignore_index=True),'timing.csv')]:
        write_or_check(d,n,check)
    print(sig.to_string(index=False));print(z.to_string(index=False))
if __name__=='__main__':main(W,'--check' in sys.argv)

