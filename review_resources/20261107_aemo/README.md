# Current manuscript resources: AEMO continuous-month validation

This OR2 v6 accompanies the 7 November 2026 AEMO revision: 28 main pages and 20 supplementary pages. The previous frozen manuscript and OR2 v5 remain [available unchanged](../20261107_pir/README.md).

- [Numerical_Resources_20261107_v6.zip](Numerical_Resources_20261107_v6.zip) — current complete OR2; retains all prior results.
- [RECEIPT.json](RECEIPT.json) — archive hash, upload-file hashes and verification scope.

## Run the checks

From the repository root:

    python -m zipfile -e review_resources/20261107_aemo/Numerical_Resources_20261107_v6.zip review_resources/20261107_aemo/extracted
    cd review_resources/20261107_aemo/extracted
    python aemo_october/check_current.py
    python aemo_october/check_package.py
    python pir_matched/check_pir.py
    python pir_grouped/check_grouped.py
    python projection_order/check_addendum.py

The first command checks aggregate arithmetic, all node/scope summaries, segment effects and timing summaries with the Python standard library. The second additionally needs NumPy. It replays all 53,568 AEMO node-date envelope/prefix checks and 53,568 archived exact rational allowances, checks floating coherence, and rejects a deliberately invalid first release. No neural fitting or optimizer replay is performed by these checkers. Twelve old and new checkers passed in a clean extraction.

## Paper-to-evidence map

| Paper location | OR2 records | Check |
|---|---|---|
| Sections 5.6/6.7, Table 14 | aemo_october/effects.csv, node_rmse.csv | aemo_october/check_current.py |
| S17 / Table S24 | aemo_october/PROTOCOL.md, REGISTRATION.json, AMENDMENT_01.json, AMENDMENT_02.json, models/*/*READY.json | source and file hashes; frozen temporal boundaries |
| Table S25 | aemo_october/node_effects.csv and node_rmse.csv | aemo_october/check_current.py |
| Table S26 | aemo_october/effects.csv, segment rows | aemo_october/check_current.py |
| Table S27 | aemo_october/risk.csv, timing.csv, publication_steps.csv | both AEMO checkers |
| Exact monthly protection | aemo_october/results/*/trajectories.npz and allowances.csv | aemo_october/check_package.py |
| Table 8 / S20 / S21 | pir_matched/ and pir_grouped/ | retained checks listed above |
| Projection order / S22 and Table 8 uncertainty / S23 | projection_order/ | projection_order/check_addendum.py |
| Li / S19 and prior KDD comparisons | li_factor/, li_preprint/, recent/, kdd_residual/ | 00_CURRENT_INDEX.txt |

## What the new month measures

The outcome follows AEMO's official underlying-demand definition: estimated operational consumption plus measurement-based rooftop PV generation. All NSW, QLD, SA, TAS and VIC observations remain, with their system sum. The five regional outcomes are nonnegative. Original signed operational-demand components remain unchanged; there is no clipping, interpolation or region deletion.

The replacement outcome and full protocol were frozen before acquiring its components, after an eligibility-only inspection of related TOTALDEMAND. The prior failed registration remains under aemo_october/prior_totaldemand/. Finalized observations enter under the registered sequential-feedback assumption; this does not reconstruct original release delays or data vintages.

October has 1,488 consecutive half-hours per node. Common, two-group and full-set publication each use their own zero-initialized allowance with no monthly sub-resets. Relative to common/two-group shrinkage, full projection reduces mean-node RMSE by 3.82%/2.26% averaged across the two backbones. System and average regional errors improve on both. NSW/PatchTST and a weekly segment retain losses. Full projection lowers PatchTST error by 5.28% relative to the same direct candidate, but raises DLinear error by 1.73%. All protected paths satisfy the cumulative limits.

The two models share the month's outcomes. PatchTST reaches its registered 500-update cap; DLinear stops at 1,783 below its 10,000 cap. This adds one temporal dataset to the evidence; it does not change the original six-dataset averages or their statistical tests.

## Raw sources and scope

All 520 original public-source URLs and hashes are in aemo_october/SOURCE_RECEIPT.json. Derived component/outcome series, release trajectories and quality summaries are included; raw-source ZIPs and neural checkpoints remain separate. Rebuilding model predictions requires those training/acquisition inputs and the preserved source environment. The exact-path checker uses the included stored predictions and observations.

Original AEMO definitions: [operational and underlying demand](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/operational-demand-data) and [rooftop PV estimates](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/nem-forecasting-and-planning/operational-forecasting/solar-and-wind-energy-forecasting/australian-solar-energy-forecasting-system).
