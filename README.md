# CPOR

**Cumulative Protection for Online Reconciliation**

Code and results for CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits, Lei Zhang, Capital Normal University.

CPOR derives the exact one-step protection set for nodewise cumulative squared-error limits relative to a causal coherent reference. Cov and ML provide two target-generation implementations.

## Current manuscript and numerical resources

Use the [dataset-scope and source-link revision](review_resources/20261107_scope_links/README.md) and [file receipts](review_resources/20261107_scope_links/FINAL_RECEIPT.json).

The study uses seven datasets: six benchmarks with equal-weight aggregate results, plus AEMO with a separately summarized continuous month and three training runs per backbone. The four principal manuscript sections now state this scope consistently. References [28]/[29] display the official AEMO URLs. All experimental values, statistical scopes, Figure 1 and OR2 v9 are unchanged.

The resource entry links the unchanged numerical archive and its check commands. Earlier frozen manuscripts remain preserved.

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
