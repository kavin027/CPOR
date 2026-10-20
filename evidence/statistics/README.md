# Conditional resampling provenance

`RESAMPLING_CONFIG.json` records the actual archived settings and source hash.
`block_sensitivity.csv` is the existing saved result, copied without rerunning.
Its development/SquareFull rows round to the ranges in Supplementary S3.7.
Historical `confirmation` rows identify M5; that identifier is not a claim
that the subsequent analyses constitute untouched confirmation.

The original driver is `Online_Resource_2/research_scripts/analyze_revision.py`.
It computes 2,000 circular-block replicates for each length 6, 12 and 24,
reinitializing `numpy.random.default_rng(20261016)` for each length. Dataset/
window indices are shared across methods and backbones. Window RMSEs are
averaged before the paired ratio, then group reductions are equally averaged.
The ranges are conditional descriptive percentiles, not retraining uncertainty.
Saved error streams are required to rerun this calculation; see the resource index.
