# CPOR

**Cumulative Protection for Online Reconciliation**

Code and results for CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits, Lei Zhang, Capital Normal University.

CPOR derives the exact one-step protection set for nodewise cumulative squared-error limits relative to a causal coherent reference. Cov and ML provide two target-generation implementations.

## Current manuscript and numerical resources

Use the [seven-dataset organization revision](review_resources/20261107_seven_datasets/README.md): clean main text 29 pages, supplement 21 pages, one-page cover letter. [Receipts](review_resources/20261107_seven_datasets/FINAL_RECEIPT.json) pin the current files and unchanged OR2 v9.

Datasets are introduced together in Section 5.1; forecasting protocols are in Section 5.2. Results, table cells, statistical scope and Figure 1 are unchanged. The earlier AEMO three-run and original frozen submissions remain available.

The current resource entry links the bytewise-identical numerical archive and its aggregate/stored-trajectory check commands.

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
