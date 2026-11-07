# CPOR

**Cumulative Protection for Online Reconciliation**

Code and results for CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits, Lei Zhang, Capital Normal University.

CPOR derives the exact one-step protection set for nodewise cumulative squared-error limits relative to a causal coherent reference. Cov and ML supply two target-generation implementations.

## Current manuscript and numerical resources

Use the [6 November grouped-PIR manuscript resources and 7 November evidence addendum](review_resources/20261107_pir/README.md). This is the current entry point for Table 8 and Tables S20/S21. [File receipts](review_resources/20261107_pir/RECEIPT.json) identify both the frozen OR2 v3 and the v4 resource containing the separate addendum.

    git clone https://github.com/kavin027/CPOR.git
    cd CPOR
    python -m zipfile -e review_resources/20261107_pir/Numerical_Resources_20261107_v4.zip review_resources/20261107_pir/extracted
    cd review_resources/20261107_pir/extracted
    python pir_grouped/check_grouped.py
    python pir_matched/check_pir.py
    python projection_order/check_addendum.py

The checks reproduce numerical summaries. Input-level replay requires the archived prediction and observation arrays and the project environment. The resource index describes these levels.

The frozen manuscript reports 2.09% lower RMSE for full projection versus common shrinkage and 1.88% versus a fixed two-group rule on identical PIR candidates. All three satisfy the cumulative limits. The new addendum evaluates direct versus two-stage protection of Cov/ML targets and adds dataset-cluster uncertainty for Table 8. It is available for review before any manuscript revision.

## Current implementation sources

The result archives preserve Cov/ML implementation source maps and drivers. Cov combines raw experts using error second moments; ML combines them using sequential weights. Both apply nonnegative reconciliation and nodewise cumulative protection. The addendum's direct-protection alternative preserves the same reference, target sequence, metric, allowance and feedback, with an independent allowance path.

## Historical resources

- [24 October Cov/ML resources](review_resources/20261024_covml/README.md) retain their original archive and table numbering.
- [22 October resources](review_resources/20261022_v2/README.md) retain earlier Root results.
- [21 October resources](review_resources/20261021/README.md) retain the earlier training-budget results.
- [Previous README](review_resources/history/README_before_pir_20261107.md) preserves the prior navigation.

The root cpor.py, examples/synthetic.py and root requirements.txt belong to the historical Root API. Current Cov/ML entry points and release drivers are indexed in the current resource archive.
