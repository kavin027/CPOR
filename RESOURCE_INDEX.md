# Resource navigation — 24 October 2026

The current increased-budget DLinear results are in
[E1, version 20261022-v2](review_resources/20261022_v2/README.md).
Use its four summary checkers for included data; ledger replay additionally
requires the indexed NPZ inputs. `evidence/check_results.py` belongs to the
historical 500-update results, not the current manuscript tables.

The [README implementation map](README.md#framework-names-and-implementation-entry-points)
distinguishes CPOR-Cov, CPOR-ML and CPOR-Root. The public `CPOR` class is Root.
E2 paired analysis and E3 framework analysis are named submission attachments;
see the submission package's `CURRENT_RESOURCE_INDEX.md` for their exact
versions, hashes and commands. They are not part of the pinned E1 archive.

The following index is retained for provenance. Its table numbers refer to
the 21 October manuscript, and its uses of "current" are historical.

---

# Historical resource index — 21 October 2026

See [review_resources/20261021/README.md](review_resources/20261021/README.md)
for the historical Table 5/7/8 CSVs and checkers, four-arm source/driver mapping,
all 66 CPOR configuration boundaries and 132 Ada-Y/PIR runtime receipts.
`python evidence/check_results.py` checks the historical 500-update aggregates, not the earlier
single-run 10-group four-arm comparison. The old checker is preserved as
`evidence/check_results_historical.py` and is not evidence for current Table 8.

The original numerical core remains unchanged from dfe5693. Added four-arm
sources are byte-identical to the recorded experiments and are hash-mapped
in `review_resources/20261021/SOURCE_AND_RESULTS_MAP.json`. The current
manuscript pins this resource revision's commit. Repository code, synthetic
example, aggregate tables and source hashes are public; saved-path ledger
audits and optimizer replay additionally require retained prediction/observation
archives under source access conditions. Online Resources 1-3 are retained
research archives, not asserted public downloads. Email the corresponding
author identifying dataset, backbone, window, run and requested audit.

No model training, new forecasts or raw-data replay is performed by the
standard-library aggregate checker. The original numerical API and optional
cached-input replay entry remain documented in README.md.
