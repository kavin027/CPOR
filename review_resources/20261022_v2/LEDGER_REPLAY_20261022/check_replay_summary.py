"""Check delivered replay results and hashes only; does not reread NPZ inputs."""
from pathlib import Path
import csv,json,hashlib
A=Path(__file__).resolve().parent
r=list(csv.DictReader((A/'TRAJECTORY_REPLAY.csv').open()))
s=json.loads((A/'REPLAY_SUMMARY.json').read_text())
assert len(r)==s['trajectories']==66
assert sum(int(x['releases']) for x in r)==s['releases']==4752
assert sum(int(x['node_checks']) for x in r)==s['node_checks']==431568
assert len({(x['dataset'],x['backbone'],x['window'],x['run']) for x in r})==66
for x in r:
    assert int(x['node_checks'])==72*int(x['nodes'])
    assert all(int(x[k])==0 for k in ['envelope_violations','prefix_violations','balance_mismatches'])
for name,h in json.loads((A/'BUNDLE_HASHES.json').read_text()).items():
    assert hashlib.sha256((A/name).read_bytes()).hexdigest()==h
print(json.dumps(dict(passed=True,trajectories=66,node_checks=431568,scope='Checks delivered replay summary arithmetic and hashes only; run replay_nodewise_ledger.py with original NPZ inputs to repeat the exact ledger replay.'),indent=2))
