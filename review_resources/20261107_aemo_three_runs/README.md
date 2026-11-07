# CPOR AEMO three-run revision — OR2 v9

Current clean manuscript: 30 pages; Supplement/Online Resource 1: 21 pages; one-page cover letter. [FINAL_RECEIPT.json](FINAL_RECEIPT.json) identifies the exact documents and numerical archive. The manuscript revision adds no new experiment, preserves Figure 1, and keeps [the original frozen submission](../20261107_submission_final/README.md) unchanged.

## Current numerical package

[Numerical_Resources_20261107_v9.zip](Numerical_Resources_20261107_v9.zip)

    python -m zipfile -e review_resources/20261107_aemo_three_runs/Numerical_Resources_20261107_v9.zip review_resources/20261107_aemo_three_runs/extracted
    cd review_resources/20261107_aemo_three_runs/extracted
    python aemo_three_runs/check_current.py
    python aemo_three_runs/check_results.py --self-test

Both checks require NumPy plus Python's standard library. The first rebuilds current RMSE and paired percentages from stored predictions; the second recomputes 1,232,064 exact nodewise envelope checks and the same number of prefix checks across 138 protected trajectories. Coherence is checked within the stated floating-point tolerances. These checks do not retrain a model or re-execute an optimizer.

## Current evidence and table mapping

- Main Table 14(a): aemo_three_runs/THREE_RUN_RMSE.csv. Tables show three-run mean RMSE and sample SD.
- Main Table 14(b): aemo_three_runs/PAIRED_GAINS.csv. Percentages are calculated within runs before averaging.
- Supplement S24: protocol/selection records and BACKBONE_FITS.csv.
- S25: node and level RMSE, RUN_SCOPE_RMSE.csv and THREE_RUN_RMSE.csv.
- S26: AEMO_WEEKLY.csv; segments preserve month-long balances.
- S27: PROTECTION_CHECKS.csv, CPOR_COST_AND_EVENTS.csv and per-release diagnostic records.
- S28: complete-procedure results for all applicable evaluated methods.
- CURRENT_VERSION.json maps the numerical archive. Original six-benchmark data/checks and the original one-run AEMO folder remain unchanged.

At equal protection and identical Cov/ML targets, full projection improves RMSE over common shrinkage by 9.0191%/9.0981%, and over all-real protection by 14.0459%/14.3198%. The corresponding protection costs are 49.2376%/51.1424%. With Cov input supplied to protected MinT, complete-procedure margins narrow to 0.4256%/1.3484%. Regional losses and ML's near tie with a fixed persistence expert on DLinear are retained. Larger differences against external complete procedures include candidate-generation effects.

The added method coverage and training repeats are retrospective after the original single-run temporal validation. The study retains all five AEMO regions, finalized sequential feedback, a 0.5% allowance and 1,488 uninterrupted October releases. The three runs share one month and are not independent new periods. Historical data-vintage arrival times are not reconstructed. The original pre-acquisition outcome/PIR registration remains in the archive.
