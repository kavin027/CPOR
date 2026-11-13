"""Recalculate the headline effects from the shareable stream-level records."""
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
p=Path(__file__).resolve().parent
for name,expected in json.loads((p/'PUBLIC_SHA256.json').read_text()).items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
for block,n,expected in [('nov08',144,[2.28305739,.03490531]),('nov11',96,[1.53293162,-.02213700])]:
    t=pd.read_csv(p/(block+'.csv'))
    for ref,e in zip(['NNLS','Incumbent'],expected):
        sub=t[t.reference==ref].set_index(['bb','seed','arm']).sort_index()
        full=sub.xs('Full',level='arm'); two=sub.xs('AdaptiveTwo',level='arm')
        gain=(100*(1-full.overall/two.overall)).mean()
        assert len(full)==9 and np.isclose(gain,e,atol=1e-7)
        assert (sub.query("arm!='Direct'").prefix_violations==0).all() and all(full.n==n)
        print(block,ref,round(gain,5),'percent; protected prefix breaches:',0)
bounds=pd.read_csv(p/'exact_grid.csv').query("rule=='two'")
win=bounds.part.isin(['early','late'])
assert (int(bounds.loc[win,'strict_mean'].sum()),int(win.sum()))==(37,126)
assert (int(bounds.loc[~win,'strict_mean'].sum()),int((~win).sum()))==(14,63)
print('Fixed two-group separation: 37/126 and 14/63. PASS.')


arm=pd.read_csv(p/'arm_metrics.csv')
pair=pd.read_csv(p/'pair_metrics.csv')
assert len(arm)==8 and len(pair)==4
st=arm.query("reference=='Incumbent'")
assert (st.eligible_issues==0).all()
np.testing.assert_allclose(pair.query("reference=='Incumbent'").sort_values('block').top_lowers_se_pct,[50,47.762345679],rtol=1e-8)
print('Published diagnostic rows and reported values match; use the OR2 checker for the underlying release paths.')
