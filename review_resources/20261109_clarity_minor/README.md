# CPOR document update — 9 November 2026

[DOCUMENT_VERSION.json](DOCUMENT_VERSION.json) identifies the current clean manuscript (24 pages), supplement (17 pages), cover letter and editable sources by filename, size and SHA-256. It also maps the preceding Full Clarity PDFs to the same OR2 v11 records.

This is a wording-only revision. S5 now distinguishes two-window aggregation from RMSE over all 144 continuous observations. The main paper links to this map. No results, experiments, methods or figures changed.

## Numerical records

The frozen [OR2 v11 index](../20261108_unified/README.md), [aggregate archive](../20261108_unified/Numerical_Resources_20261108_v11.zip), [table map](../20261108_unified/CURRENT_VERSION.json) and [8 November receipt](../20261108_unified/FINAL_RECEIPT.json) remain unchanged. The older receipt identifies the older Unified PDFs; it does not identify the newer 17-page supplement.

Extract the aggregate archive and run:

    python current/check_tables.py

The complete submission payload remains `CPOR_OR2_v11.zip`, with the unchanged hash in DOCUMENT_VERSION.json. Stored-trajectory checking additionally needs that payload:

    python current/check_tables.py --trajectories

The first command rebuilds aggregate results. The second also checks stored releases. Neither retrains a backbone or recomputes a release policy. The public map lists file identities; the manuscript PDFs and complete prediction payload accompany the submission.
