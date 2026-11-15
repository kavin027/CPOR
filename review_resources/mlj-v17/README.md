# CPOR evidence for Machine Learning, v17, 15 November 2026

From extracted Online Resource 2, use Python 3.11, install `python -m pip install -r requirements.txt`, then run `python v17/check.py`. From a checkout of the public tag, install `review_resources/mlj-v17/temporal/kernel/requirements-aemo.txt` and run `python review_resources/mlj-v17/check.py`. This release uses the adjoining v15 and v16 directories for inputs and recorded direction differences. An uninterrupted end-to-end historical and current check is available in OR2 as `python check_results.py`.

|Question|Inputs and check|
|---|---|
|Which corrections can interpolation reproduce?|integrity/geometry.py uses exact arithmetic for the worked example and the nonunique representation. groups/check.py reconstructs all group summaries from node/issue records and tests ten representative releases using original matrices. It does not re-solve classifications for all 18,144 issues.|
|How large are the benchmark effects?|groups/effects.py and groups/analyze_geometry.py rebuild the paired original-unit contrasts, dataset-level uncertainty, node distributions and geometric summaries. Timing values are recorded wall-clock measurements.|
|Can hindsight coefficients recover Full?|integrity/independent.py recomputes all 189 primal/dual comparisons and error-transfer margins directly from hashed v15 arrays and v16 direction differences with integer arithmetic.|
|Did each published November path meet its contract?|temporal/verify.py rebuilds the first-report clock, audits the exact stored releases, regenerates the Full and Independent paths and scores each arm using its own reserved credit. See temporal/TIMELINE.md for acquisition and registration. Both rules use the same archived PIR candidate.|
|How much of the temporal difference arises at the same inherited state?|state/compare.py recomputes both actions at both actual histories and matches both diagonal paths against the released predictions.|

The 13–14 November dates constitute one chronological holdout with nine prediction streams. The Independent replay on that block is retrospective. The 55 combinations with identified training stages and the eight partially identified combinations are indexed in v16. Publication replays start at the saved candidate. They do not recover a missing training checkpoint or measure out-of-domain performance.
