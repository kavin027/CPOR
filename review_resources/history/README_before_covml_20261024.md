# CPOR

**Cumulative Protection for Online Reconciliation**

Code for *CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits*
by Lei Zhang, Capital Normal University.

CPOR reconciles system and regional forecasts while enforcing a cumulative
error limit for every node relative to a prescribed reference. The framework
supports Cov, ML and Root as alternative prediction selectors, each maintaining
its own remaining allowance. The exact protection structure and its online
maintenance are the shared contribution; the selectors are not three separate
algorithmic novelty claims. The protection guarantee does not imply accuracy dominance.

## Quick start

Use Python 3.11. The numerical core runs on CPU; a GPU is not needed.

```bash
git clone https://github.com/kavin027/CPOR.git
cd CPOR
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python examples/synthetic.py
python review_resources/20261022_v2/CURRENT_RESULTS_20261022/check_current_results.py
```

The synthetic example exercises candidate protection, six scored weight fits,
exact envelope checks, feedback settlement and numerical coherence checks.
The current aggregate check requires only the Python standard library. It verifies
the increased-budget DLinear results paired with the retained PatchTST streams.

## Current manuscript evidence (22 October 2026, E1 v2)

Use [the E1 resource index](review_resources/20261022_v2/README.md) for the current
results, dataset-composition sensitivity, final-guard diagnostics and nodewise replay.
The [complete E1 archive](review_resources/CPOR_Evidence_20261022_v2.zip) and
[archive SHA256](review_resources/EVIDENCE_RECEIPT.json) are supplied together.
The index distinguishes four commands runnable with included files from ledger replay
requiring archived NPZ inputs. No forecast/observation caches are published.

Current aggregate reductions are 17.04% versus adapted ProCAST, 11.27% versus Ada-Y,
and about 0.433% for root/full versus squared/full. CPOR-Root has no aggregate advantage
over protected Cov-only or ML-Pol; these are descriptive comparisons, not population
generalization guarantees. The author-run ledger replay passed 431,568 node-date
checks; it is not independent external replication.

## Historical manuscript evidence (21 October 2026)

The [historical resource index](review_resources/20261021/README.md), `evidence/`,
`python evidence/check_results.py` and `python cpor.py check-results --evidence evidence`
refer to the original 500-update DLinear streams. Their 28.474332% ProCAST reduction
and 0.357553% root/square reduction must not be substituted for current results.
The prior code and results remain unchanged for provenance and budget sensitivity.

## Framework names and implementation entry points

**The public `CPOR` class implements the CPOR-Root selector in the current manuscript.**
The class and numerical source are unchanged. Cov and ML are archived experiment
paths, not additional classes with the same API.

| Manuscript implementation | Existing result key | Entry points |
|---|---|---|
| CPOR-Cov | `CovFullFix` | [`method.targets`](output/rr_recap_error_dependence_20261003/method.py), then [`precision_release.release_repaired`](output/rr_recap_pooling_20261003/precision_release.py); the archived M5 path uses [`method.release`](output/rr_recap_error_dependence_20261003/method.py) instead |
| CPOR-ML | `ProtectedMLPol` | [`common.controls`](output/c0_seed_validation_20261016/common.py) builds the ML-Pol target, reconciles it, and calls `common.protect` |
| CPOR-Root | `CPOR` / `C0` | [`cpor.CPOR`](cpor.py), delegating to [`Actor`](output/paired_root_actual_credit_20261013/numerical_v2/run.py) |

All three share a reference rule and allowance ratio, not a shared live balance.
Root uses the pre-protection Cov target and protects it under Root's own state;
it does not take CPOR-Cov's published forecast as its first candidate.
Archived numerical paths and calibration conventions are preserved, including
the M5 Cov path distinction. Do not substitute one wrapper and assume that its
stored trajectory must remain identical.

The current manuscript's E2 (`CPOR_Paired_Analysis_20261022_v2.zip`) and E3
(`CPOR_Framework_Analysis_20261023.zip`) are submission attachments. E2 provides
paired dataset-level analysis. E3 provides the three-implementation summaries,
descriptor analysis, timing receipts and `SOURCE_MAP.json`. They are not claimed
to be part of the historical E1 repository commit. The accompanying submission
package's `CURRENT_RESOURCE_INDEX.md` and `EVIDENCE_TABLE_MAP.csv` map current
table numbers to these retained files.

## Using the Root implementation

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
and `RootFull` refer to CPOR-Root, and `RR-RECAP` is its previous public name.
The current four-arm actor and square-risk kernel are linked in the resource
index; other two-step arms remain historical controls. Numerical sources are preserved byte for byte.
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
