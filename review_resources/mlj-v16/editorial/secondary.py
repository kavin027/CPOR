"""Historical selection and source-overlap sensitivity, preserving registered primary dates."""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
W=Path(__file__).resolve().parent
if (W.parent/'temporal/records').is_dir(): N=W.parent/'temporal/records'
else: raise FileNotFoundError('Place beside temporal/records')
V=next((p for p in [W.parents[1]/'cpor_mlj_v15_20261115/kit/aemo',W.parents[1]/'v15/aemo',W.parents[1]/'mlj-v15/aemo'] if p.is_dir()))
arms=['Reference','AdaptiveTwo','Full']
def score(p,y):return np.sqrt(np.mean((p-y)**2,axis=0)).mean()
hist=[]
for f in sorted(V.glob('*.npz')):
    block,_,_,bb,seed,*_=f.stem.split('_')
    with np.load(f) as a:
        ref=score(a['B'],a['Y'])
        hist.append(dict(bb=bb,run=int(seed),Reference=1.,**{k:score(a[k],a['Y'])/ref for k in arms[1:]}))
choices=pd.DataFrame(hist).groupby(['bb','run'])[arms].mean().idxmin(axis=1)
rows=[]
for (bb,run),pick in choices.items():
    with np.load(N/f'{bb}_{run}.npz') as a:
        for name,ix in [('all',slice(None)),('uncached_suffix',slice(4,None)),('day1',slice(0,48)),('day2',slice(48,None))]:
            r={k:score(a[k][ix],a['Y'][ix]) for k in arms}
            rows.append(dict(bb=bb,run=run,sample=name,pick=pick,**r,selected=r[pick],full_vs_adapt=100*(1-r['Full']/r['AdaptiveTwo']),full_vs_ref=100*(1-r['Full']/r['Reference']),selected_vs_ref=100*(1-r[pick]/r['Reference'])))
d=pd.DataFrame(rows);d.to_csv(W/'selection_sensitivity.csv',index=False)
print(choices.value_counts());print(d.groupby('sample')[['full_vs_adapt','full_vs_ref','selected_vs_ref']].mean())


