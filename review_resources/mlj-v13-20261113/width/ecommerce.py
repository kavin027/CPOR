"""Summarize issued ECommerce forecast changes and candidate eligibility on the actual Full ledger."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = F(40401,40000)


def generate(streams):
    for ctx in sorted((streams/'ECommerce').glob('*/*/*/context.npz')):
        bb,run,part=ctx.relative_to(streams/'ECommerce').parts[:3]
        if part not in ('early','late','continuous'):continue
        location=ctx.parent/('carry' if part=='continuous' else 'single')
        context_hash=json.loads((ctx.parent/'CONTEXT_READY.json').read_text())['sha256']
        assert hashlib.sha256(ctx.read_bytes()).hexdigest()==context_hash
        with np.load(ctx) as a:
            S,b,y=a['S'],a['b'][36:],a['truth'][36:]
        assert S.shape==(14,112) and np.linalg.matrix_rank(S)==14
        preds={}
        provenance={'context_sha256':context_hash}
        for name in ['PIR','PIRCoherent','PIRFull']:
            path=location/(name+'.npz')
            digest=json.loads(path.with_suffix('.json').read_text())['sha256']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
            provenance[name+'_sha256']=digest
            with np.load(path) as a:preds[name]=a['pred']
        raw,z,p=[preds[k] for k in ['PIR','PIRCoherent','PIRFull']]
        assert raw.shape==b.shape==z.shape==p.shape==y.shape
        outside=outside_issues=0
        a=[F(0)]*b.shape[1]
        for anchor, target, issued, truth in zip(b,z,p,y):
            crossed=0
            for i,(bi,zi,pi,yi) in enumerate(zip(anchor,target,issued,truth)):
                br,zr,pr,yr=map(lambda n:F.from_float(float(n)),(bi,zi,pi,yi))
                def fee(x):
                    return x*x-R*br*br+max(F(0),R*br-x)**2/(R-1)
                crossed+=int(fee(zr)>a[i])
                assert fee(pr)<=a[i]
                a[i]+=R*(br-yr)**2-(pr-yr)**2
            outside+=crossed
            outside_issues+=int(crossed>0)
        yield dict(bb=bb,run=run,part=part,issues=len(b),node_issues=b.size,
                   raw_abs_sum=np.abs(raw).sum(),reconcile_abs_change_sum=np.abs(z-raw).sum(),
                   protect_abs_change_sum=np.abs(p-z).sum(),candidate_outside=outside,
                   issues_outside=outside_issues,**provenance)


def summarize(fr):
    assert len(fr)==27 and not fr.duplicated(['bb','run','part']).any()
    n=fr.assign(protocol=fr.part.map({'early':'window','late':'window','continuous':'continuous'})).groupby('protocol').sum(numeric_only=True)
    out=pd.DataFrame(index=n.index)
    out['issues']=n.issues
    out['node_issues']=n.node_issues
    out['mean_absolute_revision_items']=n.reconcile_abs_change_sum/n.node_issues
    out['absolute_revision_pct_abs_raw']=100*n.reconcile_abs_change_sum/n.raw_abs_sum
    out['mean_absolute_protection_items']=n.protect_abs_change_sum/n.node_issues
    out['target_outside_full_node_pct']=100*n.candidate_outside/n.node_issues
    out['target_outside_full_issue_pct']=100*n.issues_outside/n.issues
    return out.reset_index()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--streams',type=Path,help='Original unified streams tree for regeneration')
    args=p.parse_args()
    if args.streams:
        rec=pd.DataFrame(generate(args.streams))
        rec.to_csv(HERE/'ecommerce_streams.csv',index=False)
        summarize(rec).to_csv(HERE/'ecommerce_summary.csv',index=False)
    actual=summarize(pd.read_csv(HERE/'ecommerce_streams.csv'))
    pd.testing.assert_frame_equal(actual,pd.read_csv(HERE/'ecommerce_summary.csv'),rtol=1e-12,atol=1e-10)
    print(actual.to_string(index=False))
