# Machine Learning v16 evidence, 15 November 2026

In extracted Online Resource 2, install requirements.txt in a Python 3.11 environment and run python check_results.py. For only the new evidence run python v16/check.py. In a clone of the public tag mlj-v16-20261115, run python review_resources/mlj-v16/check.py after installing review_resources/mlj-v15/requirements-aemo.txt and the numerical dependencies in groups/requirements.txt.

| Question | Inputs and check |
|---|---|
| Which protected rules retain accuracy at each node? | groups/records/node_rmse.csv.gz, groups/summary, groups/analyze.py --check. groups/check_example.py regenerates all six releases for an included example. For all 189 full-dimensional forward replays the original archives listed in v15/inputs.csv are needed. The README gives sourcing and commands. |
| Which fixed directions are ruled out? | integrity/certificate_margins.csv, differences, ../v15/paths (mlj-v15/paths on GitHub), integrity/certificate_scope.py. Includes outward bounds under alternate direction conventions. |
| What survives incomplete training provenance? | integrity/identified_combinations.csv, integrity/sensitivity.py, groups/summary/verified_subset.csv. |
| What if latent representations change? | representation/streams.csv and lp_streams.csv; check.py rebuilds both sets of summaries. Replaying the 54 paths for each sensitivity uses the authorized original stream archive. |
| Do the demand results survive a strong reference? | temporal/REGISTRATION.txt, source_records.csv, clock.csv, predictions, records, temporal/check.py. Regenerates all 45 publication paths. Regional and system results and publication times are in temporal/summary.csv. The source receipt identifies the AEMO input files. |
| What can future validation establish? | temporal/future_runner.py and REGISTRATION.txt. The 17–23 November evaluation is registered, with no results in this edition. |
| Which external configurations were evaluated? | baseline_check/README.md and its two audit tables. |

The new comparisons reuse trained forecast streams. The compact public files support score reconstruction, fixed-class certificate verification, the six-rule example and demand release replay. They do not replace missing original training checkpoints. Historical timing measurements are included as records. Hardware-specific execution time is not an equality target. For historical results and replay instructions see ../v15/README.md (mlj-v15/README.md on GitHub).
