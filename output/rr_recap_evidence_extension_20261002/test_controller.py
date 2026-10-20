from fractions import Fraction
from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from fast_solver import SafeProjection
from controller import release,proposal,excess_bound
from endpoints import build
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'procast_joint_safe_20261001_v4'
def audit(x,out,delay):
    rr=(1+Fraction('.005'))**2;k=rr-1;bb=x['b'][36:];yy=x['truth'][36:];pp=out['pred'];checks=0
    for j in range(yy.shape[1]):
        B=Fraction(0);costs=[];gains=[];actual=Fraction(0)
        for t in range(len(yy)):
            if t-delay>=0:B+=costs[t-delay]+gains[t-delay]
            b,y,p=[Fraction.from_float(float(a)) for a in [bb[t,j],yy[t,j],pp[t,j]]]
            adv=max(Fraction(0),(rr*b-p)/k);h=(p-adv)**2-rr*(b-adv)**2
            assert h<=B;B-=h;costs.append(h);gain=rr*(b-y)**2-(p-y)**2;gains.append(gain);actual+=gain
            assert B>=0 and actual>=0;checks+=1
    assert out['witness'].min()>=0;np.testing.assert_allclose(out['pred'],out['witness']@x['S'].T,rtol=1e-8,atol=2e-6)
    np.testing.assert_array_equal(out['pred'][:delay],x['b'][36:36+delay])
    return checks
def main():
    rng=np.random.default_rng(351);S=np.array([[1,1],[1,0],[0,1.]])
    ybot=np.maximum(0,rng.normal(10,3,(150,2)));wide=pd.DataFrame(ybot@S.T,index=pd.date_range('2020-01-01',periods=150));raw=wide.to_numpy()[42:]+rng.normal(0,2,(108,3))
    total=0;causal=0
    for delay in [1,3,12]:
        x=build(wide,raw,S,42,delay);m=proposal(x['b'],x['z'],x['truth'],x['v'],delay)
        changed=wide.copy();cut=60;changed.iloc[42+cut:]+=300
        alt=build(changed,raw,S,42,delay);am=proposal(alt['b'],alt['z'],alt['truth'],alt['v'],delay)
        np.testing.assert_array_equal(x['z'][:cut+delay],alt['z'][:cut+delay]);np.testing.assert_array_equal(m[:cut+delay],am[:cut+delay])
        for geometry in ['full','segment','scalar']:
            out=release(x,m,geometry,delay);total+=audit(x,out,delay)
            altout=release(alt,am,geometry,delay)
            np.testing.assert_array_equal(out['pred'][:cut+delay-36],altout['pred'][:cut+delay-36]);causal+=1
    # Real-data daily candidate equivalence to the established expert bank.
    frame=pd.read_csv(OLD/'data/RH.csv',parse_dates=['ds']).set_index('ds');old=dict(np.load(OLD/'inputs/RH/42/endpoints.npz'))
    x=build(frame,old['raw'],old['S'],len(frame)-108)
    for name in ['b','z','v']:np.testing.assert_allclose(x[name],old[name],rtol=1e-9,atol=1e-7)
    r=dict(passed=True,synthetic_node_prefix_and_reservation_checks=total,delay_causality_checks=causal,daily_candidate_reproduced=True)
    (ROOT/'controller_tests.json').write_text(json.dumps(r,indent=2));print(r)
if __name__=='__main__':
    with threadpool_limits(1):main()
