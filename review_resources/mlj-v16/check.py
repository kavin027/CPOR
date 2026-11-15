"""Verify the v16 evidence in extracted OR2 or a clone at its frozen tag."""
from pathlib import Path
import hashlib,json,sys,subprocess
import pandas as pd
import numpy as np
p=Path(__file__).resolve().parent
kit=next(a for a in [p.parent/'v15',p.parent/'mlj-v15'] if (a/'inputs.csv').is_file())
for name,h in json.loads((p/'FILES.json').read_text()).items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
def run(script,*args):
    subprocess.run([sys.executable,str(p/script),*map(str,args)],cwd=p,check=True)
run('groups/analyze.py','--check')
run('groups/check_example.py')
for name in ['certificate_scope','sensitivity']:
    run('integrity/'+name+'.py','--kit',kit,'--legacy',p/'legacy_inputs')
run('temporal/check.py')
for prefix in ['','lp_']:
    raw=pd.read_csv(p/f'representation/{prefix}streams.csv')
    assert len(raw)==54 and raw.breaches.sum()==0
    raw['protocol']=np.where(raw.part=='continuous','continuous','window')
    cols=['original','alternative','full']
    cell=raw.groupby(['dataset','protocol','backbone','run'])[cols].mean()
    gains=100*(1-cell.full/cell.alternative)
    a=gains.groupby(['dataset','protocol']).mean()
    b=pd.read_csv(p/f'representation/{prefix}summary.csv').set_index(['dataset','protocol']).full_vs_alt
    np.testing.assert_allclose(a.sort_index(),b.sort_index(),atol=1e-10)
print('PASS MLJ-V16: per-node comparisons, six-rule example, 189 certificates, provenance subset, 45 demand paths and both representation audits.')
