# CPOR Labour early-window update — 8 November 2026

[OR2 v10 numerical resources](Numerical_Resources_20261108_v10.zip) and [exact file receipts](FINAL_RECEIPT.json) accompany the clean 29-page manuscript,22-page supplement and one-page cover letter. The manuscript/sources are identified by hashes in the receipt; this folder hosts their numerical evidence.

The completed retrospective Labour early experiment covers December2008–November2014. It adds two backbone configurations and six streams. Current primary summaries cover two windows for all six benchmarks:24 configurations and72 streams over three training runs per backbone. Window RMSE is averaged within each dataset/backbone/run before paired ratios; all six datasets retain equal weights. AEMO's uninterrupted October results remain separate and unchanged.

Current matched-PIR gains are1.94% over common shrinkage and1.75% over two-group shrinkage. The mean RMSE change relative to direct coherent PIR release is-0.07%; dataset-level costs and benefits remain. The Cov/ML target protection costs average23.24%/19.84%. ECommerce reversals, near ties with matched MinT, and layer-specific losses are retained. Early Labour is a retrospective addition, not a new untouched validation.

After extracting the ZIP (use a short destination path on Windows):

    python benchmark_update/check_current.py
    python labour_early/check_bundle.py
    python aemo_three_runs/check_current.py
    python aemo_three_runs/check_results.py --self-test

These commands passed in a clean extraction. They rebuild aggregates and uncertainty or check stored predictions; they do not retrain neural models or rerun projection optimizers. The early-window checker covers246 results and138 protected trajectories, with566,352 exact envelope and566,352 exact prefix checks. AEMO retains its independent stored-trajectory checker. Primary Cov/ML benchmark counts are456,192 node–date checks per implementation after the addition.

CURRENT_VERSION.json inside the archive maps each current table to unrounded records. Original v9 records and earlier registrations remain. DUET, COSA/robust/MCGrad, same-state geometry, projection order and dedicated timing retain their original window coverage. Historical timings are not labelled as measurements on the added early window. Figure1 PDF and editable SVG are unchanged. No experiment was added during this manuscript revision.

Previous frozen version: [dataset-scope/source-link edition](../20261107_scope_links/README.md).
