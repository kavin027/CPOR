CPOR Current Results, version 1 — 22 October 2026

Main accuracy data: three increased-budget DLinear training runs and three
retained PatchTST runs, six datasets, 22 configurations per training run.
Old 500-update results are retained separately for budget sensitivity.

After extraction, run from the parent of CURRENT_RESULTS_20261022/:
python CURRENT_RESULTS_20261022/check_current_results.py
Python standard library only; no observations, training or network required.

The check covers the aggregate statistics supporting main Tables 5–12 and
supplementary Tables S2–S8 and S13, including the old-budget side of Table 12.
Configuration RMSEs, group means/SDs, paired gains, four-arm means, regional/
system RMSE, violation frequencies/final p95 and protection costs are checked.
It verifies included-file hashes. It does not regenerate neural forecasts,
rerun online strategies, or replay exact rational protection ledgers.

The historical public R1 archive (commit 4baa19e...) uses old predictions.
Its checker is not the current-table checker. Supplement Table S14 and
historical allowance curves remain old-budget results.

SOURCE_INDEX.json and derived/INPUT_HASHES.json retain original research
paths as provenance; those paths are not needed for aggregate checking.
Raw observations, prediction caches and checkpoints are not distributed here.
