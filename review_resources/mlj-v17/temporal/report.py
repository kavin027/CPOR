from pathlib import Path
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parent
s=pd.read_csv(p/'scores.csv');c=pd.read_csv(p/'comparisons.csv');t=pd.read_csv(p/'timings.csv')
nodes=['SYSTEM','NSW1','QLD1','SA1','TAS1','VIC1']
rows=[]
for bb,g in s.groupby('bb'):
    means=g.groupby('rule')[['overall','regions','SYSTEM']].mean()
    for k in ['Independent','Direct']:
        q=c[(c.bb==bb)&(c.comparator==k)]
        for l in ['overall','regions','SYSTEM']:
            rows.append({'bb':bb,'comparator':k,'level':l,'Full_MW':means.loc['Full',l],'comparator_MW':means.loc[k,l],
                        'mean_paired_gain_percent':q[l].mean(),'wins':int((q[l]>0).sum()),'runs':len(q)})
out=pd.DataFrame(rows);out.to_csv(p/'by_backbone.csv',index=False)
print(out.to_string(index=False))
for k in ['Independent','Direct']:
    q=c[c.comparator==k]
    print(k,'ranges, wins by level')
    for l in ['overall','regions',*nodes]:
        print(l,round(q[l].min(),4),round(q[l].max(),4),sum(q[l]>0))
    stack=q.set_index(['bb','run'])[nodes].stack();print('node wins',sum(stack>0),'/',len(stack),'worst',stack.idxmin(),stack.min(),'median',stack.median())
print(s.groupby('rule')[['envelope','prefix','ever']].sum())
print('clock lag',pd.read_csv(p/'inputs/clock.csv').iloc[24:].lag_minutes.agg(['min','max']))
print(t.groupby(['rule','order']).ms.agg(['count','mean','median']))

