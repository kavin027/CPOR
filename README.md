# CPOR

**Cumulatively Protected Online Reconciliation**

Code for *Online Forecast Reconciliation under Nodewise Cumulative Error Constraints*
by Lei Zhang, Capital Normal University.

CPOR reconciles system and regional forecasts while enforcing a cumulative
error limit for every node relative to a prescribed reference. It constructs
protected candidates and learns shared combination weights from root-error
increments. The protection guarantee does not imply accuracy dominance.

## Quick start

Use Python 3.11. The numerical core runs on CPU; a GPU is not needed.

```bash
git clone https://github.com/kavin027/CPOR.git
cd CPOR
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python examples/synthetic.py
python cpor.py check-results --evidence evidence
```

The synthetic example exercises candidate protection, six scored weight fits,
exact envelope checks, feedback settlement and numerical coherence checks.
The aggregate check requires only the Python standard library. It reconstructs
the paper's six-dataset average reductions: 28.4743323683% relative to adapted
ProCAST and 0.3575532491% relative to the protected squared-risk variant.
These arithmetic checks are not regeneration of predictions.

## Using the algorithm

```python
from cpor import CPOR
model = CPOR(S, v)
result = model.issue(index, b, ub, cov_target, experts, previous_label,
                     calibration=False, reset=False)
prediction = result["pred"]
```

- `S`: nonnegative aggregation matrix, shape `(n, m)`.
- `v`: fixed positive node variance weights, shape `(n,)`.
- `b = S @ ub`: coherent nonnegative reference; `ub` has shape `(m,)`.
- `cov_target`: causal Cov candidate, shape `(n,)`.
- `experts`: nine causal raw expert forecasts, shape `(n, 9)`.
- `previous_label`: previous issue's outcome, returned before this release.
- `calibration=True`: use the protected Cov candidate and archive the record.
- `reset=True`: reset live cumulative error and remaining allowance, retaining
  the archived calibration states; use once at the start of scoring.

The supplied core uses the paper's fixed RMSE allowance of 0.5%, eleven
candidate columns, regularization and solver budget. After the final label
returns, call `model.close(final_label)` once, then set
`model._actor.pending = None` if continuing with a manually settled state.
`examples/synthetic.py` gives a complete sequence.
Cov construction and the nine expert rules are in the archived source modules;
the synthetic example supplies simple causal candidates to exercise the interface.

## Source layout and reproducibility

`cpor.py` is the public interface. `output/` contains the frozen numerical
source dependencies under their original relative paths; despite the directory
name, this release includes no prediction caches. The historical keys `C0`
and `RootFull` refer to CPOR, and `RR-RECAP` is its previous public name.
Other experimental arms in those files are historical controls, not the
method reported as CPOR. Numerical sources are preserved byte for byte.
`SOURCE_SHA256.json` identifies them.

`evidence/` contains group-level results, the arithmetic checker, an exact
ledger auditor and numerical settings. `evidence/ENVIRONMENT.json` records the
historical research environment, not an installable dependency lock.
`requirements.txt` uses NumPy 1.26.4 to satisfy SciPy 1.16.3's dependency
requirement; the historical environment recorded NumPy 1.25.2. Functional
validation of the public environment does not establish bitwise equivalence
of all benchmark outputs under changed dependencies.

Original observations, neural prediction caches, snapshots and trained
checkpoints are not included. Full empirical replay needs these inputs and
their applicable source permissions. The cached replay entry point is:

```bash
python cpor.py replay --project-root /path/to/research-checkout --cell 0 --out /path/to/research-checkout/output/cpor_replay_new
python cpor.py audit-ledger --evidence evidence --project-root /path/to/research-checkout
```

Those commands require the archived research directory layout. They cannot
run from this source-only checkout alone. The release does not include raw-data
backbone training or claim to reproduce the original protocols of external
papers. The aggregate CSVs retain original run identifiers for traceability.

## Contact

Lei Zhang — lei.zhang@cnu.edu.cn
