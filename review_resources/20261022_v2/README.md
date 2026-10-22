# CPOR evidence E1 — 20261022-v2

Evidence for the increased-budget DLinear manuscript, with unchanged PatchTST streams.
This directory supersedes the 21 October results **for the current tables only**.
Historical inputs, code and results are preserved. The CPOR method is unchanged.

## Run from this directory

| Command (prefix with `python`) | Inputs / dependency | Output / scope | Works with this download alone? |
|---|---|---|---|
| `CURRENT_RESULTS_20261022/check_current_results.py` | Included CSV/JSON; standard library | Current table means, SDs, paired gains, levels, protection frequencies, costs and hashes | Yes |
| `REVIEW_ANALYSIS_20261022/check_dataset_sensitivity.py` | Included paired summaries; standard library | Leave-one-dataset sensitivity ranges, not CIs | Yes |
| `GUARD_AUDIT_20261022/check_guard_diagnostics.py` | Included diagnostics; standard library | Final-release displacement counts and hashes | Yes |
| `LEDGER_REPLAY_20261022/check_replay_summary.py` | Included replay results; standard library | Recorded replay counts and hashes, **not** a new replay | Yes |
| `LEDGER_REPLAY_20261022/replay_nodewise_ledger.py --workspace /path/to/IJEPES_RR_RECAP` | NumPy plus indexed original NPZ files | Exact rational CPOR nodewise envelope/prefix/allowance replay | **No** — see INPUT_ACCESS.md |

The first four commands have been checked in a fresh extracted archive. The fifth
was run by the author against the archived inputs: all 431,568 node-date checks
passed across 66 trajectories. No third-party independent replication is claimed.
This replay excludes coherence, candidate repair events, other methods and retraining.

The current paired means are 17.04% against adapted ProCAST and 11.27% against Ada-Y.
Root/full versus squared/full improves by about 0.433%. CPOR has no aggregate advantage
over protected Cov-only or ML-Pol. Historical 28.47% / 0.358% summaries are not current.

## Files and version

The accompanying archive is `CPOR_Evidence_20261022_v2.zip`, attachment E1.
`MANIFEST_SHA256.json` identifies every payload file. `EVIDENCE_RECEIPT.json`
beside that ZIP provides its complete SHA256. This version identifier remains
fixed. The manuscript links to the repository commit containing these files.
The original aggregate ZIP, `CPOR_Current_Results_20261022_v1.zip`, is also retained;
its SHA256 is 67fd73706497921f9721eb09613ba3b74debc823f34b13a07cfb28d28623ff85.
Its extracted directory is included here without changing its contents.

`SOURCE_INDEX.json` inside the ledger and guard folders records NPZ source paths and
hashes. These source inputs are not bundled. Paths are provenance, not download URLs.
There are no observations, per-date forecasts or neural checkpoints in this release.
Diagnostics record numerical displacements and event counts, not target values.
