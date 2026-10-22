from pathlib import Path
import csv,json,hashlib
A=Path(__file__).resolve().parent
r=list(csv.DictReader((A/'RELEASE_DIAGNOSTICS.csv').open()));s=json.loads((A/'SUMMARY.json').read_text());src=json.loads((A/'SOURCE_INDEX.json').read_text())
keys={(x['dataset'],x['backbone'],x['window'],x['run']) for x in r};assert len(keys)==66 and len(src)==66 and len(r)==4752
for k in keys:assert {int(x['issue']) for x in r if (x['dataset'],x['backbone'],x['window'],x['run'])==k}==set(range(1,73))
assert sum(float(x['rho'])<1 for x in r)==s['contracted']==0
assert sum(float(x['max_absolute_move'])>0 for x in r)==s['any_stored_change']==0
assert sum(int(x['learning_failed']) for x in r)==s['learning_failed']==0
assert all(float(x['rho'])==1 and float(x['contraction_fraction'])==0 for x in r)
assert all(x['reconstruction_scaled_error']<1e-12 for x in src)
assert s['contraction_median_positive'] is None and s['contraction_p95_positive'] is None
for name,h in json.loads((A/'BUNDLE_HASHES.json').read_text()).items():assert hashlib.sha256((A/name).read_bytes()).hexdigest()==h
print(json.dumps({'passed':True,'trajectories':66,'releases':4752,'final_contractions':0,'recorded_changes':0,'scope':'Checks published diagnostic arithmetic and hashes; source NPZ audit requires archived inputs; excludes candidate repairs and protection-ledger replay.'},indent=2))
