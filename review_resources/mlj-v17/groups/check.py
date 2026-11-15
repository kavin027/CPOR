"""Check the standalone Full versus Independent evidence package."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from box import classify

ROOT = Path(__file__).resolve().parent
for name, expected in json.loads((ROOT/'sha256.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
for script in ['effects.py','analyze_geometry.py']:
    subprocess.run([sys.executable, str(ROOT/script), '--check'], check=True)
examples=pd.read_csv(ROOT/'examples/index.csv')
for row in examples.itertuples():
    with np.load(ROOT/'examples'/row.file) as m:
        label, low, high, latent = classify(m['S'],m['low'],m['high'],m['p'],m['w'])
    assert (label,int(low.sum()),int(high.sum()),int(latent.sum())) == (row.box,row.node_low,row.node_high,row.latent_witness_out), row.file
d=pd.read_csv(ROOT/'issue_geometry.csv.gz',float_precision='round_trip')
v=d.full_distance-d.matched_independent_distance
r=v/(1+d.matched_independent_distance)
assert (r>1e-6).sum()==31 and np.isclose(r.max(),2.0728030307582457e-6,rtol=1e-9,atol=0)
assert not d.box.eq('unresolved').any()
print(f"PASS: 189 path summaries, {len(d):,} issue records, {len(examples)} geometric certificates; objective reversals {(r>1e-6).sum()}, max normalized {r.max():.12g}, absolute {v.max():.12g}.")

