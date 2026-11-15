"""Optional crosswalk from the four original forecast caches to a compact certificate."""
import argparse,csv,hashlib
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from grouped import partition
from validate import dyadic,exact_inputs
from verify import ROOT,unpack

def residual(p,y):
    u,a=dyadic(p);v,b=dyadic(y);e=max(a,b)
    return u*2**(e-a)-v*2**(e-b),e

def check(root,row):
    files={}
    for key in ('context','PIRCoherent','PIRFull','PIRTwoGroup'):
        f=root/row[key+'_archive']
        if hashlib.sha256(f.read_bytes()).hexdigest()!=row[key+'_sha256']:raise ValueError(f)
        files[key]=f
    with np.load(files['context']) as a,np.load(files['PIRCoherent']) as z:
        m=SimpleNamespace(S=a['S'],B=a['b'][36:],Y=a['truth'][36:].astype(float),Z=z['pred'])
        B,Y,D,e,*_=exact_inputs(m,a['ub'][36:],z['witness'],partition(m.S))
    with np.load(ROOT/row['download']) as w:
        assert e==int(w['exponent'])
        np.testing.assert_array_equal(B-Y,unpack(w,'residual'))
        np.testing.assert_array_equal(D,unpack(w,'direction'))
        cap=unpack(w,'cap')
        assert np.all(cap<=B) and np.all((cap==B)|(401*cap>=40000*np.abs(D).sum(2)))
        for key,arm in [('full','PIRFull'),('group','PIRTwoGroup')]:
            with np.load(files[arm]) as f:r,k=residual(f['pred'],m.Y)
            assert k==int(w[key+'_exponent'])
            np.testing.assert_array_equal(r,unpack(w,key+'_error'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path,help='original directory containing streams/')
    p.add_argument('--case',help='one case ID from inputs.csv; default is all 189')
    args=p.parse_args()
    rows=[r for r in csv.DictReader((ROOT/'inputs.csv').open()) if args.case is None or r['case']==args.case]
    if not rows:p.error('unknown case')
    for row in rows:check(args.archive,row)
    print('PASS original-input crosswalk:',len(rows))

