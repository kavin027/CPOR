# Current Cov/ML verification levels

The result ZIP belongs to the 24 October 2026 Cov/ML manuscript. Root results remain historical and are not used by the current tables.

| Level | Entry point / command | Required inputs | What is verified |
|---|---|---|---|
| Table arithmetic and versions | From the extracted ZIP: `python check_results.py` | Included CSVs, receipts and source/hash manifests; Python standard library | Main and supplementary aggregate results, paired percentages, timing means/P95 and file identity. Does not independently replay forecasts or prove the stored audit assertions. |
| Stored-trajectory audit | In the archived project: `python output/cpor_envelope_comparison_20261024/verify_trajectories.py` | Hash-identified original endpoint caches, completed comparison NPZ outputs and JSON receipts, NumPy; retain original project paths | Exact rational envelope and prefix tests on 1,726,272 node-date records, tolerance-based coherence, and agreement with archived predictions. Each record receives both envelope and prefix tests. |
| Forecast regeneration and timing | In a fresh mirrored project output directory: `python output/cpor_timing_full_20261024/run_timing.py` | Cached backbone forecasts and historical observations, original layout and numerical environment, indexed archived targets/predictions for comparison | Regenerates experts and Cov/ML targets and republishes them; compares each against the archive. Does not retrain a backbone. Existing DONE receipts are resume markers: use an empty timing-output directory to rerun, and never overwrite frozen evidence. |

The last two rows require archived inputs not included in this ZIP. Inspection copies of their scripts are in `matched_envelope/` and `timing_full/`; source closures preserve project-relative paths. They are not turnkey commands in an arbitrary extracted folder. `timing_full/verify_replay.py` separately compares stored replay arrays with original archives without importing the forecasting algorithm. It also requires those arrays.

`NUMERICAL_ENVIRONMENT.json` specifies the timing environment: Python 3.11.0, NumPy 1.25.2, SciPy 1.16.3, CVXPY 1.6.5, Clarabel 0.11.1 and scikit-learn 1.3.0; one numerical thread. Exact array layout and reconstruction batching also matter for bitwise replay and are documented in `timing_full/AMENDMENT.md`. Installing these versions alone is not evidence of replay agreement.

The current attachment contains aggregate data, source and receipts, not observation/prediction caches or neural checkpoints. Cached-input access remains subject to the original source terms.
