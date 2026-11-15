"""Recalculate the fixed-two online vs Full procedure gain on 63 and 55 group-runs."""
import argparse,csv,math
from collections import defaultdict
from pathlib import Path
from statistics import mean
def rows(p):return list(csv.DictReader(p.open(encoding='utf8')))
def main():
    if not __debug__:raise RuntimeError('Run with assertions enabled (omit -O).')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit',type=Path,required=True)
    parser.add_argument('--legacy',type=Path,required=True)
    p=parser.parse_args()
    out=Path(__file__).resolve().parent
    allrows=rows(p.kit/'inputs.csv');subset=rows(out/'identified_combinations.csv')
    good={(s['dataset'],s['backbone'],s['run']) for s in subset if s['include_primary']=='1'}
    assert len(good)==55
    prior={(s['scope'],s['protocol']):float(s['gain_pct'])
           for s in rows(p.legacy/'MATCHED_SENSITIVITY.csv') if s['comparator']=='PIRTwoGroup'}
    for mode in ['window','continuous']:
        for scope in ['all','omit_8_group_runs']:
            comb=defaultdict(list)
            for r in allrows:
                g=(r['dataset'],r['backbone'],r['run'])
                if mode==r['protocol'] and (scope=='all' or g in good):
                    comb[g].append((float(r['full']),float(r['online'])))
            effects=defaultdict(list)
            for g,q in comb.items():
                assert len(q)==(2 if mode=='window' else 1)
                f,b=map(mean,zip(*q))
                effects[g[:2]].append(100*(1-f/b))
            assert len(effects)==21
            gain=mean(mean(q) for q in effects.values())
            assert math.isclose(gain,prior[scope,'continuous_carry' if mode=='continuous' else mode],abs_tol=1e-9)
            print(mode,scope,len(comb),gain)
    print('PASS fixed-rule effect using stream-level original-unit RMSE')
if __name__=='__main__':main()

