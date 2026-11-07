# CPOR dataset-scope and source-link revision

The clean manuscript has 29 pages, Supplement/Online Resource 1 has 21 pages, and the cover letter has one page. [FINAL_RECEIPT.json](FINAL_RECEIPT.json) identifies exact delivery files and the unchanged numerical archive.

The abstract, introduction, results overview and conclusion state that seven datasets are used: six benchmarks and AEMO demand. The six benchmark datasets receive equal weights in their aggregate results and dataset-level tests. AEMO is summarized over one continuous month, averaging three training runs for each backbone; its results remain outside those six-benchmark summaries. All numerical results, table cells and statistical scopes are unchanged.

References [28] and [29] now print visible, clickable official URLs. The prior version embedded the URLs in linked titles. Reference numbering is unchanged.

- [28: AEMO operational demand and underlying-demand definition](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/operational-demand-data)
- [29: AEMO solar forecasting system and measured rooftop-PV estimated actuals](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/nem-forecasting-and-planning/operational-forecasting/solar-and-wind-energy-forecasting/australian-solar-energy-forecasting-system)

Figure 1 PDF/SVG and all previous frozen packages are preserved. No new experiment, training, retuning or numerical analysis is added.

## Unchanged numerical evidence

Use [OR2 v9](../20261107_aemo_three_runs/Numerical_Resources_20261107_v9.zip), whose exact hash is recorded in the receipt. Main Table 14 and Supplement Tables S24--S28 keep their original numerical mapping.

    python -m zipfile -e review_resources/20261107_aemo_three_runs/Numerical_Resources_20261107_v9.zip extracted_or2_v9
    cd extracted_or2_v9
    python aemo_three_runs/check_current.py
    python aemo_three_runs/check_results.py --self-test

The earlier successful aggregate and saved-trajectory checks apply to this bytewise-identical archive. This writing-only revision does not present those as new experiments.

[Previous seven-dataset organization revision](../20261107_seven_datasets/README.md) and [original frozen submission](../20261107_submission_final/README.md) remain unchanged.
