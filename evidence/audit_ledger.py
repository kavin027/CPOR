"""Independent exact audit of stored releases; no optimization or training.
Usage: python audit_ledger.py --project-root PATH
Requires the original cached C0_snapshots.npz inputs, listed in the source
RUN_DIAGNOSTICS.json. This source-only package does not redistribute them.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,hashlib
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--project-root',required=True,type=Path);args=ap.parse_args()
R=args.project_root;O=Path(__file__).resolve().parent
index=R/'output/kais_structural_revision_20261018/Online_Resource_3/RUN_DIAGNOSTICS.json'
rows=json.loads(index.read_text());ratio=F(40401,40000);eta=ratio-1
total=0;paths=[]
for row in rows:
    if row['method']!='C0':continue
    p=Path(row['metadata']).with_name('C0_snapshots.npz')
    # Original manifest contains absolute paths; relocate using its output suffix.
    if not p.exists():p=R/Path('output')/str(p).replace('\\','/').split('/output/',1)[1]
    with np.load(p,allow_pickle=False) as z:x={k:z[k] for k in z.files}
    inds=np.flatnonzero(x['scored']);assert len(inds)==72
    n=x['b'].shape[1];a=[F(0) for _ in range(n)]
    for t in inds:
        for i in range(n):
            saved=F(int(x['A_num'][t,i]),int(x['A_den'][t,i]));assert saved==a[i]
            b,pub,y=[F.from_float(float(x[k][t,i])) for k in ['b','published','y']]
            gamma=pub*pub-ratio*b*b+max(F(0),ratio*b-pub)**2/eta
            assert pub>=0 and gamma<=a[i]
            a[i]+=ratio*(b-y)**2-(pub-y)**2;assert a[i]>=0
            total+=1
    paths.append(dict(file=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),node_releases=len(inds)*n))
    print('checked',len(paths),row['dataset'],row['backbone'],flush=True)
assert len(paths)==66 and total==431568
result=dict(method='CPOR',archived_method_key='C0',paths=len(paths),node_releases=total,exact_R='40401/40000',account_mismatches=0,envelope_violations=0,prefix_violations=0,inputs=paths,scope='Stored releases only; independent from stored allowances; no new forecasts or fitting; coherence separately audited.')
(O/'LEDGER_AUDIT.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='inputs'}))
