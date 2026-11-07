# Current PIR release-rule resources

The current manuscript is the 6 November 2026 grouped-PIR clean version: 26 main pages and 15 supplementary pages. Cov and ML remain the two implementations of CPOR.

| Artifact | Scope |
|---|---|
| [Numerical_Resources_20261106_v3.zip](Numerical_Resources_20261106_v3.zip) | Exact OR2 supplied with that frozen manuscript, including Table 8 and Tables S20/S21 |
| [Numerical_Resources_20261107_v4.zip](Numerical_Resources_20261107_v4.zip) | Same records plus a separately labelled retrospective projection-order and uncertainty addendum |
| [Projection_Order_Addendum_20261107.zip](Projection_Order_Addendum_20261107.zip) | The addendum alone |
| [RECEIPT.json](RECEIPT.json) | File SHA256 values and verification scope |
| [ADDENDUM.md](ADDENDUM.md) | New results and interpretation; these have not yet been inserted into the manuscript |

## Checks

Python 3 and its standard library suffice for the aggregate checks. From the repository root:

    python -m zipfile -e review_resources/20261107_pir/Numerical_Resources_20261107_v4.zip review_resources/20261107_pir/extracted
    cd review_resources/20261107_pir/extracted
    python pir_grouped/check_grouped.py
    python pir_matched/check_pir.py
    python projection_order/check_addendum.py

The first two commands check the frozen manuscript's PIR summaries. The third checks the subsequent addendum. Ten scripts passed after a fresh extraction; other commands are in the archive's 00_CURRENT_INDEX.txt.

| Paper result | Full-precision records inside OR2 | Check from extracted OR2 root |
|---|---|---|
| Table 7 / S16 | three_way/coherence_only.csv and kdd_residual/ | python three_way/check_aggregate.py and python kdd_residual/check.py |
| Table 8, DUET / S17 | recent/duet/window_RMSE.csv | python recent/check_recent.py |
| Table 8, PIR / S20 | pir_matched/window_rmse.csv and window_audit.csv | python pir_matched/check_pir.py |
| Table 8, two-group / S21 | pir_grouped/stream_metrics.csv and release_times.csv | python pir_grouped/check_grouped.py |
| Li / S19 | li_factor/windows.csv and li_preprint/windows.csv | python li_factor/check_current.py and python li_preprint/check_li.py |
| New projection-order addendum | projection_order/cell_rmse.csv, paired_rmse.csv and cell_audit.csv | python projection_order/check_addendum.py |
| New Table 8 interval analysis | projection_order/table8_dataset_effects.csv and table8_run_effects.csv | python projection_order/check_addendum.py |

Table 8 compares identical coherent PIR candidates and independent causal allowance paths. Full projection improves RMSE by 2.09% relative to common shrinkage and 1.88% relative to a fixed two-group rule. All three release rules meet the cumulative limits. Mean publication times are 1.86, 2.52 and 5.74 ms, respectively. Seven full-versus-two-group reversals all occur on ECommerce.

## Reproduction levels

The portable scripts reconstruct reported arithmetic, empirical intervals and archived count totals. The new addendum includes target/source hashes, exact input identities, reconstruction and replay drivers. Input-level protection replay needs the archived references, targets, observations, aggregation matrices and project environment. Backbone retraining is a further operation. The archives retain the input-access and source records described in the [previous reproduction notes](../20261024_covml/REPRODUCTION.md).

The root cpor.py and historical Root examples belong to an earlier implementation. For current Cov/ML code versions, follow source_map.json, archived_source/ and timing_full/source_closure/ inside OR2. New matched release drivers are pir_matched/, pir_grouped/ and projection_order/.

## Prior versions

The [24 October release](../20261024_covml/README.md) remains available for provenance. Its table numbering belongs to that manuscript, while the mapping above belongs to the grouped-PIR manuscript.
