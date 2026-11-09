# CPOR experiment settings

This file records implementation settings moved out of the scientific Supplement. It does not alter any experiment or the frozen model-selection rules. Window and continuous protocols, comparison definitions and statistical units remain in the paper.

## Raw candidate bank

All windows and lags below are observation steps. There are nine raw candidates.

| Rule | Setting |
|---|---|
| Raw forecast | supplied backbone forecast |
| Four bias corrections | subtract mean returned base-forecast error over 3, 7, 14 or 28 observations; use available history, zero correction before feedback |
| Persistence | last observation |
| Recent mean | preceding 7 observations |
| Lagged value | observation 7 steps earlier |
| Ridge correction | fit after 12 returned errors, retain at most36; ridge penalty10, unpenalized intercept |

Ridge predicts base-forecast error divided by sqrt(v_i). Features are the current forecast, last observation, preceding7-observation mean, within-block position/12 and sine/cosine of the timestamp's calendar weekday. Per-node StandardScaler uses fitting-window mean and population SD, with scale1 for a constant feature. The predicted residual is returned to original units before subtraction. Before fitting starts, use the raw forecast.

The population-variance scale uses every available dataset observation strictly before the corresponding calibration start, with population ddof=0. Continuous tests use the early calibration boundary and retain this scale. Its floor is max(1e-12,1e-6 times median of variances above1e-12); use1e-12 if none exceed it. Observation lags can precede calibration. Error/feature histories start during calibration and continue causally.

Source: `output/rr_recap_evidence_extension_20261002/frozen_core.py::expert_bank` in the preserved experiment-source tree; current driver references are in OR2 v11.

## Backbone settings

| Setting | PatchTST | DLinear | DUET |
|---|---|---|---|
| Input/horizon | 24/12 | 24/12 | 24/12 |
| Learning rate | 1e-4 | 1e-4 | 5e-4 |
| Training objective | MAE | MAE | MAE plus authors' expert-balancing loss, coefficient1 |
| Maximum budget | 500 updates | 10000 updates | 100 epochs |
| Early stopping | 50 validation checks | 50 validation checks | 5 epochs |
| Validation | 36 observations | 36 observations | last20% of the earlier fitting span (chronological validation) |
| Scaling | internal RevIN | identity | authors' normalization |

DUET uses one encoder layer, model width64, FF width128, two experts, one selected expert, one attention head and batch16. Validation uses MAE alone. Eleven original DUET windows were reused; other runs cover remaining initializations, Labour early and AEMO. Reused DLinear first-run fits that stopped below the earlier5000 cap are preserved. The audit verified84 reused window streams,258 primary training stages and323 records including support fits. The 45 unique smaller-cap fits all stopped below5000; this is provenance, not an extra experimental treatment.

Exact stage boundaries and checkpoint identities are retained in the frozen training records. Some legacy receipts lack an original checkpoint hash; the later metadata audit records current hashes and reuse links without inventing historical authentication.

## External-method adaptation settings

- COSA: 12-step forecast, context from10 completed block means; standardize on36 calibration observations; warm up3 calibration blocks; authors' gated output layer/adaptive step size;3 Adam steps per update. Adapt only from returned block prefixes; complete the block update after its last observation.
- Residual corrector: at most144 pre-score forecast/observation pairs; Huber loss; last24 for validation. Early ECommerce has48 dates, split36 training/12 validation. Select lowest validation loss over10 epochs. iTransformer has2 layers,8 heads and widths256. Subtract0.05 times the residual forecast in original units; do not revise a block from its own unreturned observations.
- MCGrad regression extension: raw forecast, node id and within-block position features; at most144 returned dates, last24 for validation (12 for short ECommerce). Fixed and causal block-refit variants remain distinct.
- Li Separate+eRidge: authors' Algorithm1 candidates including cross-branch forecasts; Diag or one-factor covariance, RiskPortfolios2.1.7 executed source. Start from36 calibration dates, chronological80/20 validation, refit only on available history. The upper ridge-grid value is1e5. Labour's matrix basis difference and all historical windows remain in the preserved source/results.

These settings document the executed adaptations. They do not turn fixed-stream experiments into exact reproductions of the original papers' different protocols.

## Settings consolidated from the main text, 9 November 2026

Cov uses diagonal-shrinkage coefficient rho=0.1, ridge coefficient tau=1e-6,
and positive moment-scale floor gamma_min=1e-12. Its error history contains
at most28 returned observations. The specified allowance epsilon=0.005 is
encoded exactly as R=40401/40000; this is the same experiment, not a new value.

For PatchTST and DLinear, the36 calibration forecasts on RH, ECommerce,
Traffic, Wiki2 and Labour are generated in three12-observation chronological
folds. Each uses only earlier fitting data. M5 and AEMO models are fitted
before the historical-support forecasts. DUET uses the authors' fixed
configuration and chronological80/20 training/validation split before the
generated forecasts; Labour has a separate fit before late calibration.
All adapters use one fixed initialization per base-forecast stream.

The M5 source archive was pinned to Git blob
`925e92ee03cdb83c54a1dbbc12260a3b1939753c` in the archived input records.
No dataset, prediction, parameter choice or computation was changed by this move.
