"""Verify the retrospective fixed-partition certificates; no optimizer required."""
import csv
import hashlib
import json
import sys
from fractions import Fraction
from math import isqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import validate

ROOT = Path(__file__).resolve().parent

def unpack(record, key):
    raw = record[key]
    width = raw.shape[-1]
    return np.array([int.from_bytes(v, "little", signed=True)
                     for v in raw.reshape(-1, width)], dtype=object).reshape(raw.shape[:-1])

def score(residual, exponent):
    T, n = residual.shape
    squares = (residual**2).sum(0)
    denominator = T * 2**(2*exponent)
    scale = 2**80
    low = [isqrt(int(s)*scale**2//denominator) for s in squares]
    high = [k+int(k*k*denominator < int(s)*scale**2) for k,s in zip(low,squares)]
    return Fraction(sum(low), n*scale), Fraction(sum(high), n*scale)

def check(row):
    folder = ROOT / 'paths'
    name = row['case']
    with np.load(folder / (name+'.npz'), allow_pickle=False) as rec:
        b,r,d = (unpack(rec,k) for k in ('cap','residual','direction'))
        T,n = r.shape
        e = int(rec['exponent'])
        def bounds(v):
            f = np.ldexp(v.astype(float), -e)
            return validate.dn(f), validate.up(f)
        model = SimpleNamespace(T=T,n=n,B=np.ldexp(b.astype(float),-e),r=rec['r_float'],
                                D=rec['D_float'],online=float(row['online']))
        inputs = b,b-r,d,e,bounds(d),bounds(r)
        path = rec['path']
        assert np.all((path>=0)&(path<=1)) and np.all(path[0]==0)
        anchors = rec['anchors']
        assert np.isfinite(anchors).all() and np.all(rec['dual']>=0)
        meta = [(int(i),int(t),int(j),float(s)) for i,t,j,s in rec['meta']]
        lower = validate.dual(model,inputs,meta,anchors,rec['dual'])
        upper,_ = validate.primal(model,inputs,path,replay=True)
        full = score(unpack(rec,'full_error'),int(rec['full_exponent']))
        group = score(unpack(rec,'group_error'),int(rec['group_exponent']))
        # S9 stores floor(2^144 * relaxed squared error / T), node by node.
        s9 = Fraction(sum(isqrt(int(k)) for k in unpack(rec,'s9_moments')),n*2**72)
    assert abs(lower-float(row['lower'])) <= 1e-9 * max(1,abs(lower))
    assert upper == float(row['upper']) and abs(float(s9)-float(row['s9'])) < 1e-10*max(1,float(s9))
    assert s9<=lower<=upper and upper<=float(group[1])+1e-8
    assert abs(float(full[1])-float(row['full']))<1e-10*max(1,float(full[1]))
    assert abs(float(group[0])-model.online)<1e-10*max(1,model.online)
    verdict = 'Full below class optimum' if full[1]<lower else 'feasible fixed path below Full' if upper<full[0] else 'unresolved'
    assert verdict == row['verdict']
    return (name, float(s9), lower, upper, float(group[1]),float(full[1]),verdict)

def main():
    if not __debug__: raise RuntimeError('Run with Python assertions enabled.')
    for n,h in json.loads((ROOT/'SHA256.json').read_text(encoding='utf-8-sig')).items():
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise ValueError('Hash mismatch: '+n)
    rows = list(csv.DictReader((ROOT/'inputs.csv').open(encoding='utf-8-sig')))
    assert len({r['case'] for r in rows})==len(rows)==189
    all_cases = '--all' in sys.argv
    chosen = rows if all_cases else [r for r in rows if r['case'] in ('Wiki2_DLinear_1_early','ECommerce_DLinear_1_early')]
    totals = {}
    for row in chosen:
        result = check(row)
        key = (row['protocol'],result[-1])
        totals[key] = totals.get(key,0)+1
        if len(chosen)==2: print('%s  S9 %.8f  optimum [%.8f, %.8f]  online %.8f  Full %.8f  %s'%result)
    if all_cases:
        assert totals=={('window','Full below class optimum'):94,
                        ('continuous','Full below class optimum'):45,
                        ('window','feasible fixed path below Full'):32,
                        ('continuous','feasible fixed path below Full'):18}
    else: assert len(totals)==2
    print('PASS',len(chosen),'path certificates',totals,flush=True)

if __name__=='__main__': main()

