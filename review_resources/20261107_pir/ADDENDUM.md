# Projection order and matched-PIR uncertainty, 7 November 2026

This retrospective analysis reuses the six datasets, both backbones, three stored runs and all 66 streams. It is incorporated in the 7 November 2026 manuscript, Section 6.4 and Tables S22/S23. The previous manuscript is archived separately.

## Projection order

Within each of Cov and ML, the two policies use the same causal target sequence before reconciliation, reference, variance weights, allowance, feedback and implementation-specific numerical routines. Only the target sent to the protection projection changes. Each policy settles its own actual publications.

Positive percentages below favor direct projection of the raw target over protection after nonnegative reconciliation. Windows are averaged within dataset, backbone and run before the paired ratio; the six datasets have equal weight.

| Dataset | Cov direct gain (%) | ML direct gain (%) |
|---|---:|---:|
| ECommerce | +0.000000 | -0.000000 |
| Labour | +0.000000 | -0.000000 |
| M5 | +0.000000 | -0.000000 |
| RH | -0.035494 | -0.017580 |
| Traffic | -0.000336 | -0.000000 |
| Wiki2 | +0.052765 | +0.083512 |

Mean direct gains are 0.002822% for Cov and 0.010989% for ML. At displayed RMSE precision four datasets are virtually tied. Wiki2 favors direct projection, RH favors the two-stage path, and Cov/Traffic marginally favors the two-stage path. On labelled system totals, mean direct gains are -0.013598% (Cov) and -0.043914% (ML). Existing observations do not identify a generally better order.

Every raw target was reconciled and matched the archived coherent target exactly. All 132 two-stage release replays were bitwise identical to their archived controls. Each of the four implementation-by-policy combinations passed 431,568 exact envelope/prefix node-date checks and stored-coherence checks, with zero violations over 5,994 node sequences. These counts are author-run audit records. The portable script checks aggregate arithmetic and totals; it does not independently reconstruct the original arrays.

## Existing Table 8 uncertainty

The analysis was fixed to the two matched full-versus-common and full-versus-two-group contrasts. It enumerates all 46,656 cluster bootstrap resamples of the six equally weighted dataset effects, and all 64 dataset sign assignments. Reported intervals are unadjusted empirical 95% intervals. Holm uses this two-comparison retrospective family only.

| Contrast | Mean gain (%) | Empirical 95% interval (%) | Raw p | Holm p |
|---|---:|---|---:|---:|
| Full vs common shrinkage | 2.092290 | [1.238297, 2.789895] | 0.03125 | 0.0625 |
| Full vs two-group shrinkage | 1.875751 | [1.115860, 2.464800] | 0.03125 | 0.0625 |

Full-vs-two-group run means are 1.550497%, 2.034730%, 2.042026%. Removing one dataset at a time gives 1.679420% to 2.230039%; the corresponding full-vs-common range is 1.866154% to 2.488091%. Cluster resampling assumes exchangeable benchmark datasets, and six selected benchmarks limit inference. This is retrospective evidence.

## Check and replay

Run python projection_order/check_addendum.py from the extracted OR2 v4 root. It requires only Python 3. The script recomputes all paired values, layer summaries and uncertainty from the bundled full-precision records. Source and target receipts identify archived inputs. Replaying stored-value envelope checks requires those prediction/observation arrays and the original project environment. The addendum distributes numerical summaries and drivers, with the input-access boundaries of the frozen OR2 retained.
