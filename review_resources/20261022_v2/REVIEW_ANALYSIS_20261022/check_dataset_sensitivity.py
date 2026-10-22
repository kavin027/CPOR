from pathlib import Path
import csv,json,statistics,math,hashlib
A=Path(__file__).resolve().parent
rows=lambda n:list(csv.DictReader((A/n).open(encoding='utf-8-sig')))
for n,h in json.loads((A/'INPUT_HASHES.json').read_text()).items():assert hashlib.sha256((A/n).read_bytes()).hexdigest()==h
ordinary=rows('GROUP_RUN_PAIRED_GAINS.csv');four=rows('FOUR_ARM_RUN_GAINS.csv');expected=rows('LEAVE_ONE_DATASET_OUT.csv');summary=json.loads((A/'LEAVE_ONE_DATASET_OUT.json').read_text())
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11),(a,b)
for comp,record in summary.items():
 r=[x for x in (four if comp=='SquareFull' else ordinary) if x['comparator']==comp and (comp!='SquareFull' or x['method']=='CPOR')]
 datasets=sorted(set(x['dataset'] for x in r));assert len(datasets)==6
 dg={ds:statistics.mean(float(x['gain_pct']) for x in r if x['dataset']==ds) for ds in datasets}
 for ds in datasets:assert len([x for x in r if x['dataset']==ds])==6
 vals=[]
 for ds in datasets:
  v=statistics.mean(x for k,x in dg.items() if k!=ds);vals.append(v)
  e=[x for x in expected if x['comparator']==comp and x['omitted_dataset']==ds];assert len(e)==1
  near(e[0]['retained_mean_gain_pct'],v);near(e[0]['omitted_dataset_gain_pct'],dg[ds])
 near(record['full_mean'],statistics.mean(dg.values()));near(record['minimum'],min(vals));near(record['maximum'],max(vals))
 for ds,v in dg.items():near(record['dataset_means'][ds],v)
print(json.dumps({'passed':True,'comparisons':6,'omissions_checked':36,'datasets':6,'scope':'Descriptive dataset-composition sensitivity; not confidence intervals or independent validation.'},indent=2))
