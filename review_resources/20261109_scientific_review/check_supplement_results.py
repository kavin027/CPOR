"""Check audited numerical records, optionally against the supplied Supplement TeX.

This checks recorded arithmetic, not neural training or release-policy execution.
"""
from pathlib import Path
from collections import defaultdict
from statistics import mean
from decimal import Decimal
import argparse,csv,re,math,json
NUM=re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?')
START={'tab:variation': 1, 'tab:sd_window_DLinear': 1, 'tab:sd_window_DUET': 1, 'tab:sd_continuous_carry_PatchTST': 1, 'tab:sd_continuous_carry_DLinear': 1, 'tab:current_sd_last': 1, 'tab:aemocomplete': 1, 'tab:aemonodes': 1, 'tab:order': 1, 'tab:mcgrad': 1, 'tab:pirpaired': 2, 'tab:pir_continuous': 2, 'tab:carry_details': 2, 'tab:layer_details': 2, 'tab:layer_continuous': 2, 'tab:envelope_sd': 2, 'tab:events': 2, 'tab:ext': 2, 'tab:kddres': 2, 'tab:li': 2, 'tab:uncertainty': 3, 'tab:piruncertainty': 3, 'tab:historical_time': 3}
def tables(tex):
    answer={}
    for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',tex,re.S):
        label=re.search(r'\\label\{([^}]+)\}',m.group()).group(1)
        if label not in START:continue
        tab=re.search(r'\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}',m.group(),re.S).group(1)
        vals=[];active=False
        for line in tab.splitlines():
            if r'\midrule' in line:active=True;continue
            if not active or '&' not in line or r'\multicolumn' in line:continue
            for cell in line.split('&')[START[label]:]:
                cell=cell.split(r'\\')[0]
                vals.extend(x.group() for x in NUM.finditer(cell))
        answer[label]=vals
    return answer

def close(shown,value):
    d=Decimal(shown.replace(',',''));tol=float(Decimal(5).scaleb(d.as_tuple().exponent-1))
    return abs(float(d)-float(value))<=tol+max(1,abs(float(value)))*1e-13
def li_gains(path):
    g=defaultdict(list)
    for r in csv.DictReader(Path(path).open(encoding='utf-8-sig')):g[(r['dataset'],r['backbone'],r['scope'],r['method'])].append(float(r['rmse']))
    g={k:mean(v) for k,v in g.items()};out=[]
    for ref in ['Factor','Diag']:
        for scope in ['system','middle','bottom']:
            for impl in ['Cov','ML']:
                effects=[100*(1-g[(d,b,s,impl)]/v) for (d,b,s,m),v in g.items() if s==scope and m==ref]
                assert len(effects)==8
                out.append(dict(comparator=ref,level=scope,implementation=impl,recomputed=mean(effects)))
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--supplement',type=Path);ap.add_argument('--records-dir',type=Path,default=Path(__file__).resolve().parent);args=ap.parse_args();root=args.records_dir
    rows=list(csv.DictReader((root/'TABLE_VALUES.csv').open(encoding='utf-8-sig')))
    assert len(rows)==2188
    for r in rows:assert close(r['displayed'],r['recomputed']),(r['table'],r['position'])
    if args.supplement:
        text=args.supplement.read_text(encoding='utf8');values=tables(text);group=defaultdict(list)
        for r in rows:group[r['table_label']].append(r)
        for k,seq in group.items():
            assert len(values[k])==len(seq),(k,len(values[k]),len(seq))
            for shown,r in zip(values[k],seq):assert float(shown.replace(',',''))==float(r['displayed'].replace(',','')),(k,r['position'],shown,r['displayed'])
        paragraph=text.split('Relative to Factor, Cov and ML reduced system RMSE',1)[1].split('For Traffic/DLinear',1)[0]
        actual=[float(x) for x in re.findall(r'([-+]?\d+\.\d+)\\%',paragraph)]
        assert len(actual)==12
    gains=li_gains(root/'LI_WINDOW_RMSE.csv');expected=list(csv.DictReader((root/'LI_LEVEL_REDUCTIONS.csv').open(encoding='utf-8-sig')))
    for i,(a,b) in enumerate(zip(gains,expected)):
        assert all(a[k]==b[k] for k in ['comparator','level','implementation'])
        assert math.isclose(a['recomputed'],float(b['recomputed']),abs_tol=1e-10)
        assert close(b['displayed'],a['recomputed'])
        if args.supplement:assert actual[i]==float(b['displayed'])
    print(json.dumps({'passed':True,'table_numeric_components':len(rows),'li_level_comparisons':len(gains),'supplement_tex_checked':bool(args.supplement),'scope':'Recorded table values and historical Li aggregation; not model retraining or publication optimization.'},indent=2))
if __name__=='__main__':main()
