"""Recalculate the revised tables from the tagged, unrounded v17 records."""
from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np

here=Path(__file__).resolve().parent
prior=next(p for p in [here.parent/'v17',here.parent/'mlj-v17'] if p.is_dir())
for name,expected in json.loads((here/'FILES.json').read_text()).items():
    assert hashlib.sha256((here/name).read_bytes()).hexdigest()==expected,name
g=prior/'groups'
nodes=pd.read_csv(g/'node_geometry.csv.gz',float_precision='round_trip')
by_run=nodes.groupby(['dataset','protocol','backbone','run','part'])[['Full','Independent']].mean().groupby(level=[0,1,2,3]).mean()
by_run['saved']=by_run.Independent-by_run.Full
by_run['gain']=100*by_run.saved/by_run.Independent
ds=by_run.groupby(level=[0,1]).mean()
t=pd.read_csv(g/'timing.csv').pivot(index='dataset',columns='rule',values='mean_ms')
print('Main Table 6 (original units and mean release ms)')
print(ds.saved.unstack().join(t).to_string())
print('Table S32, Full versus Independent (%)')
print(ds.gain.unstack().to_string())
x=pd.read_csv(prior/'state/state_effects.csv').query("level=='overall'")
values=x[['full','independent_at_full_history','full_at_independent_history','independent']].mean().values
print('S35, rows = recorded history, columns = action, both in Full / Independent order')
print(values.reshape(2,2))
ff,fi,inf,ii=values
action=((fi-ff)+(ii-inf))/2
history=((inf-ff)+(ii-fi))/2
np.testing.assert_allclose([action,history,ii-ff],[.195,1.678,1.873],atol=.0005,rtol=0)
print('S35 symmetric action, history and total in MW:',action,history,ii-ff)
index=pd.read_csv(here/'training_coverage.csv')
assert len(index)==63 and (index.stages_without_identity>0).sum()==8
assert index[['matched_file_at_inventory','receipt_digest_only','stages_without_identity']].sum().tolist()==[331,12,32]
print('63 combinations; 55 with identified stages, 8 with missing stages. PASS')
