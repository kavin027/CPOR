"""Change the release rule at each of the two observed histories without changing later history."""
from argparse import ArgumentParser
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

here = Path(__file__).resolve().parent
source = here.parent / 'temporal'
sys.path.insert(0, str(source))
from publishers import SafeProjection, S, ar, publish

def available(bal, pending, issue):
    while pending and pending[0][0] <= issue:
        _, b, p, y, _ = pending.pop(0)
        bal[:] = ar.balance(bal, b, p, y)
    return [credit - sum(q[-1][i] for q in pending) for i, credit in enumerate(bal)]

def replay(path):
    base, run = path.stem.split('_')
    with np.load(path) as src:
        d = dict(src)
    assert np.all(np.diff(d['ready_ns']) >= 0)
    arms = ('Full', 'Independent')
    balances = {h: [Fraction(0)] * 6 for h in arms}
    pending = {h: [] for h in arms}
    vs = d['v']
    solvers = {h: SafeProjection(S, vs) for h in arms}
    pred = {(h, a): [] for h in arms for a in arms}
    breaks = []
    for t, (b, ub, z, u, y) in enumerate(zip(d['B'], d['UB'], d['Z'], d['U'], d['Y'])):
        for h in arms:
            free = available(balances[h], pending[h], d['issue_ns'][t])
            assert min(free) >= 0
            for a in arms:
                p, witness, pq, _, _ = publish(a, solvers[h], vs, b, ub, z, u, free)
                pred[h, a].append(p.copy())
                if a == h:
                    bq = ar.exact(b)
                    pending[h].append((d['ready_ns'][t], bq, pq, ar.exact(y), ar.excess(bq, pq)))
            f = pred[h, 'Full'][-1]
            i = pred[h, 'Independent'][-1]
            over = np.maximum(f[1:] - np.maximum(ub, u), np.minimum(ub, u) - f[1:])
            sep = bool(np.any(over > 2*(2e-6+1e-8*np.abs(f[1:]))))
            breaks.append(dict(backbone=base,run=run,issue=t,history=h,outside_box=int(sep),
                               largest_excursion_MW=max(0.,float(over.max())),
                               weighted_distance_saved=np.sum(((i-z)**2-(f-z)**2)/vs)))
    values = {k: np.sqrt(np.mean((np.array(v)-d['Y'])**2,axis=0)) for k,v in pred.items()}
    for h in arms:
        if h in d: np.testing.assert_allclose(pred[h,h], d[h], atol=2e-6, rtol=1e-9)
    with np.load(source/'expected'/path.name) as saved:
        for h in arms:
            np.testing.assert_allclose(pred[h,h], saved[h], atol=2e-6, rtol=1e-9)
    ans = []
    for level,where in [('overall',slice(None)),('regions',slice(1,None)),('system',0)]:
        v = {k: np.mean(x[where]) for k,x in values.items()}
        ff,fi=v['Full','Full'],v['Full','Independent']
        ii,if_=v['Independent','Independent'],v['Independent','Full']
        action=((ii-if_)+(fi-ff))/2
        state=((ii-fi)+(if_-ff))/2
        np.testing.assert_allclose(action+state,ii-ff,atol=1e-10)
        ans.append(dict(backbone=base,run=run,level=level,full=ff,independent=ii,
                        independent_at_full_history=fi,full_at_independent_history=if_,
                        action_MW=action,history_MW=state,total_MW=ii-ff))
    return ans,breaks

def main():
    parser=ArgumentParser();parser.add_argument('--write',action='store_true')
    write=parser.parse_args().write
    rows=[];events=[]
    for p in sorted((source/'inputs').glob('*.npz')):
        data,issues=replay(p); rows.extend(data);events.extend(issues)
    assert len(rows)==27 and len(events)==1728
    for n, records in [('state_effects',rows),('same_state_events',events)]:
        frame=pd.DataFrame(records); dest=here/(n+'.csv')
        if write: frame.to_csv(dest,index=False)
        else: pd.testing.assert_frame_equal(pd.read_csv(dest,dtype={'run':str}),frame,rtol=1e-7,atol=2e-6,check_dtype=False)
    print(pd.DataFrame(rows).groupby('level')[['action_MW','history_MW','total_MW']].mean())
    print(pd.DataFrame(events).groupby('history').outside_box.agg(['sum','count']))

if __name__=='__main__':
    with threadpool_limits(1): main()
