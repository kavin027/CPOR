# CPOR

**Cumulative Protection for Online Reconciliation**

Code and results for CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits, Lei Zhang, Capital Normal University.

CPOR derives the exact one-step protection set for nodewise cumulative squared-error limits relative to a causal coherent reference. Cov and ML provide two target-generation implementations.

## Current manuscript and numerical resources

Use the [AEMO temporal-validation revision and OR2 v6](review_resources/20261107_aemo/README.md). The main text has 28 pages; the supplement has 20. [Receipts](review_resources/20261107_aemo/RECEIPT.json) identify the current files. [The previous frozen manuscript resources](review_resources/20261107_pir/README.md) are retained unchanged.

    git clone https://github.com/kavin027/CPOR.git
    cd CPOR
    python -m zipfile -e review_resources/20261107_aemo/Numerical_Resources_20261107_v6.zip review_resources/20261107_aemo/extracted
    cd review_resources/20261107_aemo/extracted
    python aemo_october/check_current.py
    python aemo_october/check_package.py
    python pir_grouped/check_grouped.py
    python pir_matched/check_pir.py
    python projection_order/check_addendum.py

The first checker uses the standard library; exact AEMO stored-path replay needs NumPy. It verifies all 53,568 monthly node-date checks and archived rational allowances, and rejects corruption. These checks do not retrain forecasting models.

On identical coherent PIR candidates across the original six datasets, full projection improves RMSE by 2.09% over common shrinkage and 1.88% over a two-group rule. A separate frozen new-outcome AEMO month gives 3.82% and 2.26% gains without allowance resets, with system and average regional gains on both backbones and retained regional/weekly reversals. The resource README states the variable, prior eligibility inspection, feedback and training-budget boundaries.

## Current implementation sources

The result archives preserve Cov/ML implementation source maps and drivers. Cov combines raw experts using error second moments; ML combines them using sequential weights. Both apply nonnegative reconciliation and nodewise cumulative protection. The addendum's direct-protection alternative preserves the same reference, target sequence, metric, allowance and feedback, with an independent allowance path.

## Historical resources

- [24 October Cov/ML resources](review_resources/20261024_covml/README.md) retain their original archive and table numbering.
- [22 October resources](review_resources/20261022_v2/README.md) retain earlier Root results.
- [21 October resources](review_resources/20261021/README.md) retain the earlier training-budget results.
- [Previous README](review_resources/history/README_before_pir_20261107.md) preserves the prior navigation.

The root cpor.py, examples/synthetic.py and root requirements.txt belong to the historical Root API. Current Cov/ML entry points and release drivers are indexed in the current resource archive.
