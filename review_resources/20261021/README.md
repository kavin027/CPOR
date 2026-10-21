# Current review resources — v2, 21 October 2026

This directory is the resource attachment named in Supplement S8. It is also
included in `CPOR_Review_Resources_v2.zip`. The manuscript pins the Git commit.

| Paper item | Existing result files / exact source entry |
| --- | --- |
| Table 5 and 28.474% paired reduction | `results/table5/PAIRED_GAINS.csv`, `check_table5.py`; full six-method configuration and group RMSEs in the same folder |
| Table 7 protected combination/learning controls | `results/table7/PAIRED_GAINS.csv`, `check_table7.py` |
| Table 8 and Tables S6-S8 | `results/table8/CELL_METRICS.csv`, `GROUP_EXPERIMENT_METRICS.csv`, `PAIRED_GAINS.csv`, `check_table8.py` — 264 trajectories, 12 groups, 3 runs, 4 arms |
| Root and square kernels | repository `output/paired_root_actual_credit_20261013/numerical_v2/conic_learning.py::fit,dual_lower`; `output/c0_matched_ablation_20261014/square_learning.py::fit,polish` |
| Directional/full actor and completion driver | `output/c0_matched_ablation_20261014/experiment.py`; `output/cpor_ablation_completion_20261020/worker.py`; `FOUR_ARM_RUN_INDEX.csv` maps every saved arm |
| Training/calibration and all 66 CPOR paths | `CONFIG_INDEX_v1/configurations.csv`, `fitting_stages.csv`, source and archive hash files |
| Ada-Y/PIR run records | `EXTERNAL_RUN_INDEX.csv` and 132 `external_receipts/*.json` |
| ProCAST comparator identity | `PROCAST_BRANCH.md` and fixed research source locators |

From the repository root:

```bash
python evidence/check_results.py
```

The checker uses only the Python standard library and rebuilds the current
Table 5, 7 and 8 aggregates from their saved configuration RMSEs. It verifies
paired percentages before averaging and hashes of the mapped source files.
It does not regenerate RMSE from predictions or re-run solvers. CSVs and
individual table checkers are copied byte-for-byte from the completed studies;
we have not created replacement observations or new performance results.
Internal identifiers 42/1/2 map to training runs 1/2/3.

The old `evidence/ablation_group_metrics.csv` and its summary concern a single
historical five-dataset comparison. They are retained for provenance and are
not the current four-arm table or the default checker target.

Frozen drivers retain their original research paths and require the archived
inputs/registrations and training dependencies for execution. They are a
source-to-result map, not a complete raw-data retraining release. The source
closure for the CPOR numerical core remains available through `cpor.py`.
No cached predictions, observed data or checkpoint binaries are published.
Input-level replay remains subject to data-source access conditions.

`CONFIG_INDEX_v1` explicitly distinguishes original/recovered receipts from
32 reconstructed historical fitting stages. A historical reconstruction is
not a contemporaneous training receipt. No experiments were run for this
resource revision; only existing hashes and aggregate arithmetic were checked.
