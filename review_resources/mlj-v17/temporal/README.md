# 13–14 November first-report replay

Independent gives each of the five latent regional corrections a coefficient in [0,1]. Full projects onto the complete protected coherent set. Both receive the same reconciled PIR target, latest complete persistence reference and 0.5% cumulative RMSE allowance. The new comparison was specified after this period had already been analysed for other rules. All nine streams refer to the same 96 half-hours.

Run `python verify.py` from this directory with Python 3.11 and the packages in `kernel/requirements-aemo.txt`. A fresh Python 3.11 environment can use `python verify.py --install` to create a local virtual environment, install those packages and run the same checks. The public receipt verifies file digests, recomputes the availability clock and the five region-plus-system observations from `source_records.csv`, independently reconstructs exact delayed allowances from stored releases, then generates all Full and Independent forecasts from the saved PIR inputs. Regenerated releases are compared numerically, and the stored floats are audited as exact binary rational values; aggregation checks use absolute tolerance 0.000002 MW and relative tolerance 1e-8. Model training and PIR inference precede these saved inputs.

## Results

Mean RMSE (MW), averaging nine within-stream RMSE values. Regional RMSE averages the five regional node RMSEs. Overall weights all six nodes equally.

|Rule|Overall|Regions|System|
|---|---:|---:|---:|
|Reference|578.99|359.46|1676.62|
|Direct|627.26|419.24|1667.35|
|Common|568.29|353.07|1644.36|
|AdaptiveTwo|559.91|348.34|1617.75|
|Independent|556.67|346.46|1607.69|
|Full|554.79|345.34|1602.05|

The mean within-stream percentage reductions of Full over Independent were 0.339% overall, 0.325% regionally and 0.354% at the system. Full improved eight of nine overall scores and 39 of 54 node–stream scores. For DLinear run 2 it increased overall RMSE by 0.108% and QLD RMSE by 0.359%. Tasmania improved in three of nine runs. Each arm passed 5,184 exact envelope and 5,184 prefix checks. Direct had 3,646 violating prefixes and all 54 node streams violated at least once. The saved Full outputs exactly match the previous 9 streams on the verification machine.

Full versus direct publication, mean of the three within-run changes per backbone (positive values favor Full):

|Backbone|Overall (%)|Regions (%)|System (%)|
|---|---:|---:|---:|
|DLinear|28.00|27.94|28.05|
|PatchTST|14.66|25.25|−0.73|
|DUET|−21.91|−10.06|−38.00|

The across-run mean system reduction versus Direct is −3.56%. This averages normalized comparisons, while the MW columns above average unnormalized errors. Direct was more accurate on all three DUET runs at all three displayed levels. As no historical residual is available for a pending release, its worst-case excess is reserved until the complete label arrives. All source vectors were generated 30.07–31.20 minutes after interval end, so the persistence reference at issue t is the observation for t−3. The evidence is one retrospective two-day block.

## Timing and files

On the original single-process, one-library-thread paired replay, mean and P95 publication times (ms) were 5.407 and 11.126 for Full, and 4.802 and 10.121 for Independent. Each arm went first on 432 releases and second on 432. Timings include publication, representation and exact envelope checks, exclude settlement and input/output, and cover 864 calls per arm. Both solvers saw a synthetic non-scored warm-up before the loop. The machine was shared; elapsed times vary on rerun. Prior methods other than Full were not retimed.

* `scores.csv`, `comparisons.csv`, `by_backbone.csv`: absolute RMSE and counts, within-stream gains and within-backbone summary.
* `timings.csv`, `timing_summary_ms.csv`: publication order and individual wall-clock times, with pooled summaries.
* `inputs/*.npz`: frozen reconciled target (Z, U), reference (B, UB), observations Y, aggregation matrix S, variance weights v, first-report ready and issue times, original release sequences. `expected/*.npz` preserves the new Full and Independent releases, their nonnegative latent witnesses and exact free balances before each release. For example, `Fraction(str(archive['Independent_free'][t,i]))` gives the exact available credit. The nominal AEMO dates are NEM UTC+10; numeric nanosecond arrays are naive NEM clock encodings, compared directly.
* `PROTOCOL.md` and `TIMELINE.md`: v17 retrospective analysis lock and evidence dates, with earlier registration preserved at `inputs/REGISTRATION.txt`.
* `kernel/run_groups.py` is the frozen v16 algorithm; `publishers.py` supplies the exact free credit after pending reservations and exposes the latent witness already used by the acceptance check.
* `mean_rmse_MW.csv` and `mean_paired_percent.csv`: all six nodes in native MW and all within-stream relative reductions.

