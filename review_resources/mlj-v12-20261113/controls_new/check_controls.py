"""Recompute seven-dataset summaries from the unrounded archival replay records."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
KEYS=['dataset','backbone','run','protocol','arm']

def generate():
    d=pd.read_csv(HERE/'stream_records.csv')
    assert len(d)==1134 and len(d.drop_duplicates(['dataset','backbone','run','part','arm']))==len(d)
    assert d.groupby('dataset').size().eq(162).all()
    d['protocol']=np.where(d.part=='continuous','continuous','windows')
    grouped=d.groupby(KEYS).agg(rmse=('mean_node_rmse','mean'),ever=('ever','sum'),
                                   violations=('violations','sum'),incoherent=('incoherent','sum'),
                                   node_sequences=('nodes','sum'),issues=('issues','sum')).reset_index()
    for arm, name in [('PIRFull','full'),('PIRCoherent','direct')]:
        comp=grouped[grouped.arm==arm][KEYS[:-1]+['rmse']].rename(columns={'rmse':name})
        grouped=grouped.merge(comp,on=KEYS[:-1],validate='many_to_one')
    grouped['vs_direct_cost']=100*(grouped.rmse/grouped.direct-1)
    grouped['gain_full_vs_rule']=100*(1-grouped.full/grouped.rmse)
    summary=grouped.groupby(['arm','protocol']).agg(
        cost=('vs_direct_cost','mean'),full_gain=('gain_full_vs_rule','mean'),ever=('ever','sum'),
        violations=('violations','sum'),incoherent=('incoherent','sum'),node_sequences=('node_sequences','sum'),
        issues=('issues','sum')).reset_index()
    assert summary.node_sequences.isin([9612,4806]).all() and summary.issues.eq(9072).all()
    assert summary.loc[summary.arm=='PIRFull',['violations','incoherent']].eq(0).all().all()
    return summary,grouped

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true'); args=ap.parse_args()
    s,g=generate()
    if args.write:
        s.to_csv(HERE/'summary.csv',index=False)
        g.groupby(['dataset','arm','protocol'])[['vs_direct_cost','gain_full_vs_rule']].mean().to_csv(HERE/'dataset_effects.csv')
    else:
        pd.testing.assert_frame_equal(pd.read_csv(HERE/'summary.csv'),s,atol=1e-12,rtol=1e-12)
    print(s.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
