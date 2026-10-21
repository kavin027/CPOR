CPOR-implementation-index-v1-20261021

configurations.csv: 66 trajectories (22 dataset/backbone/window configurations,
three experiments). Each row gives calibration/scoring boundaries and hashes
for the archived raw predictions and CPOR snapshots.
fitting_stages.csv: 228 backbone fitting stages. Dates are inclusive.
For five non-M5 datasets, three 12-observation calibration folds and one
72-observation scoring fold use separate fits. At each fit, the last 36
pre-prediction observations are validation; preceding observations are gradient
training. For M5 one fit precedes the 108-observation support segment and
remains fixed through 36 calibration and 72 scoring observations. Cov/expert
statistics start at calibration, not at the earlier ProCAST support segment.
Within each 12-step block, the backbone does not refit; its input is refreshed
at block boundaries. Postprocessors receive returned labels sequentially.

Evidence is labelled per stage: original receipt, recovered original receipt,
or reconstructed historical schedule. A reconstruction is not execution proof.
Later experiments are retrospective evaluations on the same dates, not new
untouched confirmation datasets. Experiment IDs 1/2/3 correspond to internal
random-state identifiers 42/1/2, retained only for locating original archives.

source_index.json and source/: exact local source snapshots for the operations
documented in Supplement S7. Historical directory names are provenance locators,
not method names. These snippets are a source index, not a complete dependency
closure. Public numerical core: github.com/kavin027/CPOR at
dfe569351ff38e1d7ed568d0b490c91b3745b5cf; documentation:
2183163b5ab4cfb953c9669646bd1f901594e307.

Public code supports aggregate arithmetic checks and a synthetic example.
Input-level replay and rational audits additionally require the retained
prediction/observation archives under applicable source access terms. Hashes
identify those inputs but do not grant access. No raw observations, cached
predictions, neural checkpoints or private input archives are included here.
The historical Online Resources 1-3 are retained research archives, not asserted
to be public downloads. A complete raw-data backbone-retraining release is not
claimed. This revision performed no training or experiment replay.
