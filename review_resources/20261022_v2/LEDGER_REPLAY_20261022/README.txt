CPOR nodewise exact-ledger replay, 22 October 2026

Actual local replay: 66 current CPOR trajectories, 4752 scored releases,
431568 node-date checks; no envelope violation, prefix violation or
archived pre-release balance mismatch. R = 40401/40000 exactly.

To rerun the exact replay (requires NumPy and the indexed original NPZ files):
python LEDGER_REPLAY_20261022/replay_nodewise_ledger.py
For relocated source files, append --workspace /path/to/IJEPES_RR_RECAP
The script verifies original input hashes before using each file. No
optimizer or original forecast implementation is imported. Fractions
represent the exact stored binary64 inputs, not decimal-rounded values.

To check only the delivered summary and file hashes (standard library):
python LEDGER_REPLAY_20261022/check_replay_summary.py
This second command is NOT a prediction-level replay.

SOURCE_INDEX.json identifies archives; these contain predictions and
observations and remain subject to their source access terms. They are
not included in this bundle. The bundle contains scripts, hashes and
aggregate per-trajectory results only. RMSEs match the current CPOR result
index. This is an author-run separate implementation, not an independent
external replication or raw-data retraining. Coherence, candidate repair
events, other methods, and the historical MinT-reference experiment are
outside this replay. No performance experiment was rerun.

CANDIDATE_OPERATION_SCOPE.json counts 47520 prescribed scoring-candidate
insets, NOT 47520 nonzero changes or repairs. Fixed inset settings are not
failure triggers. Historical candidate repair counts were not retained.
