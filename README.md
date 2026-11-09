# CPOR

**Cumulative Protection for Online Reconciliation**

Code and results for CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits, Lei Zhang, Capital Normal University.

CPOR derives the exact one-step protection set for nodewise cumulative squared-error limits relative to a causal coherent reference. Cov and ML provide two target-generation implementations.

## Current manuscript and numerical resources

The [Scientific Review document index](review_resources/20261109_scientific_review/README.md)
maps the latest PDFs and the preceding Scientific Main PDFs to unchanged OR2 v11
records. The [fixed experiment specifications](docs/README.md) cover raw experts,
training boundaries and numerical acceptance. Historical archive labels are
preserved; the Li-level mapping note identifies the middle-node aggregation.

Use the [seven-dataset,three-backbone update / OR2 v11](review_resources/20261108_unified/README.md), [aggregate numeric archive](review_resources/20261108_unified/Numerical_Resources_20261108_v11.zip) and [file receipts](review_resources/20261108_unified/FINAL_RECEIPT.json).

PatchTST,DLinear and DUET each have three training runs on seven datasets. Uniform72-step early/late windows and144-step continuous tests use common forecasts,dates and feedback within each comparison. The primary grid has42 configurations,126 window streams and63 continuous streams. AEMO's1488-step month is separate and covers all three backbones.

On identical coherent PIR candidates,full projection improves RMSE over common shrinkage by3.57% in windows and4.34% continuously; gains over two-group shrinkage are2.72% and3.28%. Complete-procedure rankings against unprotected PIR vary: DUET losses,system-level reversals and protection costs are retained. All protected paths satisfy their cumulative limits. The extension is retrospective; no new experiment was run during manuscript revision.

The public archive checks aggregate arithmetic. Full stored-release replay inputs accompany the complete OR2 v11 submission package. [The previous v10](review_resources/20261108_labour_early/README.md) remains unchanged.

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
