# Numerical protection implementation

These are code-level numerical specifications, not additional scientific claims or new experimental settings.

## Projection and acceptance

- Intervals are contracted toward the reference by a fixed1e-6 fraction before numerical projection. Exact envelope certification still uses the original protection bounds.
- Up to5 Clarabel attempts, at most500 iterations each. For primal objective F and numerical dual bound D, accept a gap in [-1e-7(1+F),2e-6(1+F)]. Unaccepted QP attempts may use SLSQP; exhausted projection attempts stop the trajectory.
- Coherence uses u>=0 and absolute residual <=2e-6+1e-8*abs(Su). This is a tolerance check, separate from exact rational envelope/prefix checking.
- Coordinate repair clips to the contracted intervals, then moves a failing coordinate by up to16 representable floating-point steps toward its reference. Keep the representation only when coherence and the exact envelope check pass. Otherwise jointly contract forecast and representation; the reference is the final release-guard fallback.
- After coordinate repair, Cov records the objective gap; ML additionally requires its acceptance threshold. M5 Cov instead uses joint contraction. The historical M5 repair sensitivity remains in `ARCHIVED_IMPLEMENTATION_EXTRACTS.tex` and the unchanged stored results: mean RMSE reduction4.62e-6%, max absolute change2.93e-5% over6 paired backbone/run averages.

## State handling

The protection allowance resets only at the declared accounting start (and at the matched control's release73). Error histories, observations, ridge features and Cov/ML combination states are retained. Continuous and reset-control paths share early-window variance weights and the reference rule. Every rule settles its own actual released forecast.

Cov's temporary worst-case reservation is restored when the observation returns before the next release; reservation and restoration cancel. It therefore gives the same allowance update as direct settlement. This bookkeeping identity is archived with the implementation extracts, not presented as a new theorem.

## Source entry points

See the archived `precision_release.py::release_repaired`, `run_experiment.py::one_action`, `guard_v8.py::guard` and the current `protocol_engine.py` mapped in OR2 v11. Experiment and source hashes remain unchanged. No code behavior was changed by this documentation move.

## Historical direct/two-stage replay record

```latex
The original 66 streams are retained for this historical projection-order analysis. Where the raw target was absent from the release archive, we recovered it from the stored expert forecasts and the unchanged causal weighting rule. Reconciliation reproduced every archived coherent target within numerical tolerance. Recomputing the two-stage releases reproduced all 132 stored forecast
sequences exactly. Each implementation and order passed 431,568 exact node--date envelope and prefix checks, with zero violations and coherence within the study's numerical tolerances.

```

The external one-run protected-target audits each covered143856 node dates; these implementation audit counts were moved out of the scientific Supplement.

## Exact state representation

The stored reference, released forecast and observation are binary floating-point
values. Each is converted exactly to a rational before updating the protection
allowance. The specified decimal epsilon is also encoded rationally. The exact
allowance starts at zero and is identical to the certification allowance;
interval construction and numerical optimization use floating-point arithmetic.
Projection optimality uses numerical diagnostics. These storage/solver details
are separate from the mathematical protection result and the tolerance-based
aggregation check. See ARCHIVED_MAIN_IMPLEMENTATION_EXTRACTS.tex for the former
main-paper text.
