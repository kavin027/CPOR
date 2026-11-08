# Current unified results / OR2 v11

Download and extract [the aggregate numerical resource](Numerical_Resources_20261108_v11.zip), then run `python current/check_tables.py`. [CURRENT_VERSION.json](CURRENT_VERSION.json) maps tables to records, and [FINAL_RECEIPT.json](FINAL_RECEIPT.json) records the submitted document and full-OR2 hashes.

The main manuscript is24 pages,the supplement16 pages and the cover letter1 page. Figure1 and the previous frozen version are preserved.

Current main results use independent early/late72-step windows and144-step continuous tests on RH,ECommerce,Traffic,Wiki2,Labour,M5,AEMO. Each has36 calibration records. There are126 window prediction streams and63 continuous streams. AEMO's1488-step month is separate, now covering all three backbones and three runs. These are retrospective extensions.

## Checks

    python current/check_tables.py

This rebuilds window/run aggregation,mean/sampleSD,paired percentages,Table8,costs and seven-dataset uncertainty using unrounded records. It checks reported compliance counts; it does not replay trajectories by default.

    python current/check_tables.py --trajectories

This additionally extracts the supplied numeric payload into a new `trajectory_check` directory and checks stored binary-rational envelopes,prefixes and displayed RMSEs. It does not fit a backbone or rerun a release policy. Install NumPy and pandas; choose a fresh `--audit-dir` when repeating this optional command.

The complete OR2 ZIP includes `CPOR_Unified_Seven_Datasets_Three_Backbones_20261108.zip` and the unchanged `historical/CPOR_OR2_v10.zip`. The public aggregate resource keeps only current records and commands; the large prediction payload accompanies the submission. Its hash and exact scope appear in CURRENT_VERSION.json.

## Interpretation

Table8 supplies identical coherent candidates to alternative protected release rules. Complete-procedure comparisons against unprotected PIR/ProCAST additionally change target generation and protection requirements. DUET-PIR reversals,system losses and protection costs are retained. Training SDs are not independent-dataset uncertainty. A continuous prefix limit is not a rolling/late-half contract.

Window RMSE is averaged within each run before paired percentages. Continuous RMSE uses all144 releases directly. Costs have positive sign for error increases. Main-paper cost medians use63 run-level contrasts; group-mean medians in older research reports are distinct descriptive summaries.

Historical extra baselines,domain/order controls and dedicated timing are retained in v10 with their original scope. The current aggregate summaries do not silently pool them into the new21-group grid. The eight experiment-source files document the extension and reference the original project/archive layout; they are not a standalone raw-data retraining release.
