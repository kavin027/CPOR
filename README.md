# CPOR

**Cumulative Protection for Online Reconciliation**

Code and results for CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits, Lei Zhang, Capital Normal University.

CPOR derives the exact one-step protection set for nodewise cumulative squared-error limits relative to a causal coherent reference. Cov and ML provide two target-generation implementations.

## Current manuscript and numerical resources

Use the [AEMO three-run revision and OR2 v9](review_resources/20261107_aemo_three_runs/README.md). The clean main text has 30 pages and the supplement 21 pages. [Receipts](review_resources/20261107_aemo_three_runs/FINAL_RECEIPT.json) pin all files. [The original frozen submission](review_resources/20261107_submission_final/README.md) remains unchanged.

    git clone https://github.com/kavin027/CPOR.git
    cd CPOR
    python -m zipfile -e review_resources/20261107_aemo_three_runs/Numerical_Resources_20261107_v9.zip review_resources/20261107_aemo_three_runs/extracted
    cd review_resources/20261107_aemo_three_runs/extracted
    python aemo_three_runs/check_current.py
    python aemo_three_runs/check_results.py --self-test

The new archive distinguishes same-target protection gains from complete-procedure accuracy, retains protection costs and local reversals, and preserves all original six-benchmark records. See the fixed version's README for scope and exact commands.

## Current implementation sources

The result archives preserve Cov/ML implementation source maps and drivers. Cov combines raw experts using error second moments; ML combines them using sequential weights. Both apply nonnegative reconciliation and nodewise cumulative protection. The addendum's direct-protection alternative preserves the same reference, target sequence, metric, allowance and feedback, with an independent allowance path.

## Historical resources

- [Reviewed v8 table-note revision](review_resources/20261107_table_notes/README.md) remains unchanged.


- [Prior v7 local-freeze and evidence-scope resources](review_resources/20261107_aemo_minor/README.md) are preserved unchanged.


- [Prior AEMO v6 resources](review_resources/20261107_aemo/README.md) preserve the original 28-page manuscript evidence.


- [24 October Cov/ML resources](review_resources/20261024_covml/README.md) retain their original archive and table numbering.
- [22 October resources](review_resources/20261022_v2/README.md) retain earlier Root results.
- [21 October resources](review_resources/20261021/README.md) retain the earlier training-budget results.
- [Previous README](review_resources/history/README_before_pir_20261107.md) preserves the prior navigation.

The root cpor.py, examples/synthetic.py and root requirements.txt belong to the historical Root API. Current Cov/ML entry points and release drivers are indexed in the current resource archive.
