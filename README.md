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
- `cov_target`: the paper's **coherent, not-yet-protected** target
  $z_t^{\mathrm{Cov}}$, shape `(n,)`; it is not the raw mixture
  $a_t^{\mathrm{Cov}}$ or the protected column $c_t^{\mathrm{Cov}}$.
  `issue()` applies the protection projection and release guard to construct
  $c_t^{\mathrm{Cov}}$, the first column of the protected candidate matrix.
- `experts`: the nine raw rules **after their separate nonnegative
  reconciliation**, shape `(n, 9)`, before protection. `issue()` protects these
  nine columns and appends the reference as column eleven.
- `previous_label`: previous issue's outcome, returned before this release.
- `calibration=True`: use the protected Cov candidate and archive the record.
- `reset=True`: reset live cumulative error and remaining allowance, retaining
  the archived calibration states; use once at the start of scoring.

The supplied core uses the paper's fixed RMSE allowance of 0.5%, eleven
candidate columns, regularization and solver budget. After the final label
returns, call `model.close(final_label)` once, then set
`model._actor.pending = None` if continuing with a manually settled state.
`examples/synthetic.py` gives a complete sequence.
The raw rules are implemented by
[`expert_bank`](output/rr_recap_evidence_extension_20261002/frozen_core.py).
In the [Cov module](output/rr_recap_error_dependence_20261003/method.py),
`moment` builds the regularized expert-error second-moment matrices,
`solve_weights` computes nodewise combination weights, and `targets` forms
the raw mixture and reconciles it. `targets` returns `raw_target` and coherent
`target`.
[`make_contexts`](output/self_funded_policy_learning_20261013/calibration_contexts.py)
prepares calibration Cov targets and reconciles each expert before calling the
actor. It reuses the saved coherent scoring targets.
[`make_bank`](output/paired_root_actual_credit_20261013/numerical_v2/run.py)
protects those inputs using the actor's current remaining allowances.
These are archived experiment functions; `targets` assumes the original
36-observation calibration boundary. The synthetic example uses already
coherent constructed inputs, not the paper's raw expert-generation pipeline.

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

## Resource access

See [RESOURCE_INDEX.md](RESOURCE_INDEX.md) for the relationship between this
public checkout, Review Evidence 20261020, the historical Online Resources
1–3, saved-path audits and backbone training. The numerical source was released
in commit `dfe569351ff38e1d7ed568d0b490c91b3745b5cf`; later documentation
clarifications do not change that numerical implementation.

## Contact

Lei Zhang — lei.zhang@cnu.edu.cn
