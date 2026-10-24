# CPOR

**Cumulative Protection for Online Reconciliation**

Code and results for *CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits*, Lei Zhang, Capital Normal University.

CPOR protects each node's cumulative forecast error relative to a prescribed causal reference while maintaining aggregation consistency. The current manuscript evaluates **CPOR-Cov and CPOR-ML** as two implementations of this common protection framework.

## Current manuscript: Cov/ML, 24 October 2026

Start with [the current resource index](review_resources/20261024_covml/README.md). The [versioned result ZIP](review_resources/20261024_covml/CPOR_CovML_Results_Minor_20261024.zip) contains the current tables and standard-library checker; [its receipt](review_resources/20261024_covml/RECEIPT.json) gives the SHA256.

```bash
git clone https://github.com/kavin027/CPOR.git
cd CPOR
python -m zipfile -e review_resources/20261024_covml/CPOR_CovML_Results_Minor_20261024.zip review_resources/20261024_covml/extracted
python review_resources/20261024_covml/extracted/check_results.py
```

This checks aggregate arithmetic and versions. [REPRODUCTION.md](review_resources/20261024_covml/REPRODUCTION.md) distinguishes it from exact auditing of stored trajectories and regeneration from cached backbone forecasts. Those operations need additional hash-identified archived inputs. The [numerical environment](review_resources/20261024_covml/NUMERICAL_ENVIRONMENT.json) describes the complete-grid timing replay.

The current result attachment covers six datasets, two backbones and three training runs. Cov and ML improve mean-node RMSE over NNLS by 6.82% and 6.84%, and over adapted ProCAST by 17.07% and 17.12%. The retrospective nonnegative-versus-all-real comparison gives mean gains of 1.969% and 1.957%, with window- and layer-level exceptions. Tables and source maps in the ZIP define these comparison scopes. Protection may increase error relative to unprotected forecasts.

## Current implementation entry points

| Implementation | Archived result key | Source entry |
|---|---|---|
| Cov | `CovFullFix` | `method.targets` / context builders generate a second-moment target; `precision_release.release_repaired` or the archived M5 `method.release` path protects it. |
| ML | `ProtectedMLPol` | `common.controls` generates the adapted ML-Pol target, reconciles it, and calls `common.protect`. |

The result ZIP's `source_map.json`, `archived_source/` and `timing_full/source_closure/` map these functions and their exact versions. Each implementation maintains its own allowance. Preserve its numerical path and calibration convention for replay.

## Historical Root API and earlier evidence

The root-level `cpor.CPOR` class, `examples/synthetic.py`, and root-level `requirements.txt` support the **historical Root implementation**; they are not the current Cov/ML quick start. They remain unchanged for provenance.

- [22 October E1 resources](review_resources/20261022_v2/README.md) retain the Root-era results and checks.
- [21 October resources](review_resources/20261021/README.md) and root-level `evidence/` retain the earlier 500-update DLinear results.
- E2/E3 and three-implementation descriptions in historical records refer to earlier manuscripts, not the current tables.
- [Prior README, preserved verbatim](review_resources/history/README_before_covml_20261024.md) contains historical API instructions; its original relative links are repository-root-relative.

No observation/prediction caches or neural checkpoints are published in the current aggregate attachment. The paper and its resource index state the scope of available checks.
