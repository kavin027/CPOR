# CPOR

**Cumulative Protection for Online Reconciliation**

Code and results for CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits, Lei Zhang, Capital Normal University.

CPOR derives the exact one-step protection set for nodewise cumulative squared-error limits relative to a causal coherent reference. Cov and ML provide two target-generation implementations.

## Current manuscript and numerical resources

Use the [Labour early-window update / OR2 v10](review_resources/20261108_labour_early/README.md) and [file receipts](review_resources/20261108_labour_early/FINAL_RECEIPT.json).

The study uses seven datasets: six equally weighted benchmarks and a separately summarized AEMO month. The six benchmarks now have two windows each,24 configurations and72 streams. The Labour early addition is retrospective. AEMO, Figure1 and all prior frozen packages remain unchanged. Current matched-PIR gains are1.94% versus common shrinkage and1.75% versus two-group shrinkage, with heterogeneous protection costs and negative cases retained.

The resource entry provides the numerical archive, exact table mapping and check commands. No experiment was added during manuscript revision.

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
