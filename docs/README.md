# Fixed experiment and numerical specifications

This documentation defines the executed Cov/ML configurations for the
seven-dataset, three-backbone manuscript. It accompanies the scientific
Supplement without adding a second set of manuscript configuration tables.

| Specification | Location |
|---|---|
| Nine raw experts, observation windows, ridge features and variance floor | [EXPERIMENT_SETTINGS.md](EXPERIMENT_SETTINGS.md#raw-candidate-bank) |
| Cov coefficients, backbone losses, budgets, selection and normalization | [EXPERIMENT_SETTINGS.md](EXPERIMENT_SETTINGS.md#backbone-settings) and its final settings section |
| Calibration folds and historical support | [EXPERIMENT_SETTINGS.md](EXPERIMENT_SETTINGS.md#settings-consolidated-from-the-main-text-9-november-2026) |
| Actual training/validation boundaries and checkpoint/source provenance | [training_stage_metadata.csv](provenance/training_stage_metadata.csv), linked by [training_stream_trace.csv](provenance/training_stream_trace.csv) |
| Rational conversion, coherence residual, objective gap, retries, coordinate repair and reference fallback | [NUMERICAL_IMPLEMENTATION.md](NUMERICAL_IMPLEMENTATION.md) |
| AEMO fields, units, timestamp rules and train/support/calibration/score boundaries | [DATA_AND_REPRODUCIBILITY.md](DATA_AND_REPRODUCIBILITY.md#aemo-data-provenance) |

The exact implementation paths are indexed inside the unchanged OR2 v11
archives. The public [v11 resource](../review_resources/20261108_unified/README.md)
contains current drivers in `current/experiment_sources/`, source hashes and
provenance in `current/provenance/`; the full numerical archive retains historical
source closure and stored forecasts. These files preserve the source snapshots
used by the experiments; this documentation revision changes no model or release
rule. Local provenance paths and receipt hashes identify the original runs;
they are not extra package-relative executables or new retraining receipts.

The [current document and evidence index](../review_resources/20261109_scientific_review/README.md)
maps document hashes, the result-check addendum and archived input versions.
