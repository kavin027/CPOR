"""Compare scalar interval widths with the reference-only bound on the same issues."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = 40401 / 40000


def rows(root):
    for name, path in [('Nov08', 'clock/november_original'), ('Nov11', 'temporal/scored')]:
        folder = root / path
        lag = json.loads((folder / 'metadata.json').read_text())['k']
        assert lag == 3
        index = pd.read_csv(folder / 'stream_index.csv')
        assert len(index) == 9
        for ref, key in [('NNLS', 'B'), ('Incumbent', 'inc_B')]:
            widths, all_bounds, settled_bounds = [], [], []
            for stream in index.itertuples(index=False):
                f = folder / stream.record
                assert hashlib.sha256(f.read_bytes()).hexdigest() == stream.record_sha256
                with np.load(f) as a:
                    b, y = a[key], a['Y']
                    free = a[ref + '_Full_free']
                # At zero-based issue t, all previous dates end at t-1. With k=3
                # feedback the last settled date is t-3, inclusive.
                cumulative = np.vstack([np.zeros(b.shape[1]), np.cumsum((b-y)**2, axis=0)])
                prior = cumulative[np.arange(len(b))]
                returned = cumulative[np.maximum(np.arange(len(b))-lag+1, 0)]
                dx = np.sqrt(free * (R-1)/R)
                low = np.maximum(0, b-dx)
                high = np.where(free <= R*(R-1)*b*b, b+dx, np.sqrt(free+R*b*b))
                width = high - low
                complete_bound, available_bound = 2*np.sqrt(R*prior), 2*np.sqrt(R*returned)
                assert np.all(width >= 0) and np.all(width <= available_bound+1e-8)
                assert np.all(available_bound <= complete_bound+1e-8)
                widths.extend(width.ravel())
                all_bounds.extend(complete_bound.ravel())
                settled_bounds.extend(available_bound.ravel())
            yield dict(block=name,reference=ref,node_issues=len(widths),
                       width_median_mw=np.median(widths),
                       bound_all_prior_median_mw=np.median(all_bounds),
                       bound_returned_median_mw=np.median(settled_bounds))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path,help='Unpacked OR2 root')
    ap.add_argument('--write',action='store_true')
    args=ap.parse_args()
    computed=pd.DataFrame(rows(args.root))
    frozen=pd.read_csv(args.root/'mechanism/arm_metrics.csv').query("arm=='Full'")
    ordered=computed.merge(frozen,on=['block','reference'],validate='one_to_one',suffixes=('_new','_old'))
    np.testing.assert_allclose(ordered.width_median_mw_new,ordered.width_median_mw_old,atol=1e-9)
    assert (ordered.node_issues_new==ordered.node_issues_old).all()
    target=HERE/'widths.csv'
    if args.write:computed.to_csv(target,index=False)
    else:pd.testing.assert_frame_equal(computed,pd.read_csv(target),rtol=1e-12,atol=1e-10)
    print(computed.to_string(index=False))
