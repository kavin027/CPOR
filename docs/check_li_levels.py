"""Rebuild Li level comparisons from retained node RMSEs; no model execution."""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import argparse,csv,json,math,re

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--records-dir',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--supplement',type=Path);a=ap.parse_args();root=a.records_dir
    leaf_counts={'Traffic':200,'Wiki2':150,'Labour':32,'M5':30}
    nodes=defaultdict(dict)
    for row in csv.DictReader((root/'RAW_NODE_RMSE.csv').open(encoding='utf-8-sig')):
        key=(row['dataset'],row['backbone'],row['window'],row['method'])
        i=int(row['node']);assert i not in nodes[key];nodes[key][i]=float(row['rmse'])
    assert len(nodes)==64
    windows={};groups=defaultdict(list)
    for key,v in nodes.items():
        ds,bb,w,method=key;n=len(v);assert set(v)==set(range(n));start=n-leaf_counts[ds]
        for scope,ids in [('system',[0]),('middle',range(1,start)),('bottom',range(start,n)),('all',range(n))]:
            val=mean(v[i] for i in ids);windows[key+(scope,)]=val;groups[(ds,bb,method,scope)].append(val)
    assert len(windows)==256
    records=[r for r in csv.DictReader((root/'RAW_RECOMPUTED_WINDOW_RMSE.csv').open(encoding='utf-8-sig')) if r['scope'] in ['all','system','middle','bottom']]
    assert len(records)==256
    for r in records:
        k=tuple(r[x] for x in ['dataset','backbone','window','method','scope'])
        assert math.isclose(windows[k],float(r['rmse']),rel_tol=1e-13,abs_tol=1e-10),k
    groups={k:mean(v) for k,v in groups.items()};out=[]
    for comparator in ['Factor','Diag']:
        for scope in ['system','middle','bottom']:
            for method in ['Cov','ML']:
                values=[100*(1-groups[(ds,bb,method,scope)]/v) for (ds,bb,m,s),v in groups.items() if m==comparator and s==scope]
                assert len(values)==8;out.append(mean(values))
    expected=list(csv.DictReader((root/'LI_LEVEL_REDUCTIONS.csv').open(encoding='utf-8-sig')))
    assert len(expected)==12
    for val,r in zip(out,expected):assert math.isclose(val,float(r['recomputed']),abs_tol=1e-10)
    if a.supplement:
        s=a.supplement.read_text(encoding='utf8').split('Relative to Factor, Cov and ML reduced system RMSE',1)[1].split('For Traffic/DLinear',1)[0]
        shown=[float(x) for x in re.findall(r'([-+]?\d+\.\d+)\\%',s)]
        assert len(shown)==12
        for val,display in zip(out,shown):assert abs(val-display)<=.005000001,(val,display)
    print(json.dumps({'passed':True,'window_level_values':256,'paired_level_comparisons':12,'middle':'all nonsystem nonbottom nodes; Labour 8 regional plus 16 sex-region nodes','scope':'Arithmetic on retained per-node errors, not a new forecast or raw-array replay.'},indent=2))

if __name__=='__main__':main()
