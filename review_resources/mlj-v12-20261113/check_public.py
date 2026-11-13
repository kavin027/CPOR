from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parent
for name,h in json.loads((p/'SHA256.json').read_text()).items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
for name in ['controls_new/check_controls.py','theory_new/check_theory.py']:
    subprocess.run([sys.executable,p/name],check=True)
d=pd.concat([pd.read_csv(p/'clock_new'/n) for n in ['stream_scores.csv','nnls_scores.csv']])
assert len(d)==540 and d.query("rule!='Direct'")[['prefixbad','envbad']].eq(0).all().all()
for method,expected in [('Cov',[563.72,619.53]),('ML',[563.02,618.94])]:
    a=d.query("target==@method and rule=='Full' and ref=='persistence'").groupby('block').overall.mean()
    np.testing.assert_allclose(a,expected,atol=.005)
for rule,expected in [('Direct',[626.87,638.87]),('Full',[536.78,592.86])]:
    a=d.query("mode=='linear' and ref=='persistence' and rule==@rule").groupby('block').overall.mean()
    np.testing.assert_allclose(a,expected,atol=.005)
print('MLJ V12 public summary checks passed. For release-level verification use the journal OR2.')
