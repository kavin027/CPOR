# Resource index — manuscript minor revision, 20 October 2026

Public repository: https://github.com/kavin027/CPOR

The numerical source is unchanged from commit
`dfe569351ff38e1d7ed568d0b490c91b3745b5cf`; `SOURCE_SHA256.json` pins its
21 dependency files. Current documentation adds interface and access details.
The manuscript delivery records its full repository commit in
`CPOR_REPOSITORY_RELEASE.json`; `CPOR_Public_Code_20261020.zip` is the
matching Git archive.

| Check | Stable file or entry point | What it verifies | Additional inputs / access |
| --- | --- | --- | --- |
| Method and synthetic sequence | `cpor.py`, `examples/synthetic.py` | Protected candidate construction, weight fitting, exact envelope and final balance checks on synthetic data | Public dependencies only |
| Aggregate arithmetic | `evidence/check_results.py`, `paired_group_seed_gains.csv`, `ablation_group_metrics.csv`, `ablation_summary.json` | Reported group averages and four-arm table arithmetic | None; no prediction regeneration |
| Conditional intervals | `evidence/statistics/RESAMPLING_CONFIG.json`, `block_sensitivity.csv` | Recorded resampling settings and saved ranges | Original saved errors and `Online_Resource_2/research_scripts/analyze_revision.py` needed for rerunning |
| Saved-path ledger | `evidence/audit_ledger.py`; `Review_Evidence_20261020/LEDGER_AUDIT.json` in the manuscript package | Exact envelope and prefix balances, 66 paths / 431,568 node releases | `Online_Resource_3/RUN_DIAGNOSTICS.json` plus indexed `C0_snapshots.npz` files; conditional author access |
| Cached-input replay | `python cpor.py replay --project-root <research-root> --cell 0 --out <fresh-output>` | One existing configuration's optimizer outputs | Cached endpoints, raw expert bank, Cov target and saved CPOR output in archived research layout; conditional author access |
| Backbone training | Data-source references and `Review_Evidence_20261020/historical_cache_provenance.csv` | Acquisition/fitting provenance, not a runnable training release | Original datasets, training environment and model history; no complete retraining pipeline or checkpoints are included |

## Review Evidence and Online Resources

`Review_Evidence_20261020` is the source-and-aggregate attachment inside the
manuscript submission package. It contains the same numerical dependencies
plus node maps, aggregation matrices, training boundaries and audit summaries.
Its `SHA256.json` identifies the delivered files. It is not a forecast-cache
release and is broader than the public aggregate examples.

Online Resources 1–3 are retained research archives, not public downloads or
attachments bundled into this source-only release:

- Resource 1: cached inputs, original method outputs, source and provenance.
- Resource 2: controls, saved-path analyses and the original resampling driver.
- Resource 3: repeated-experiment archives, metric rebuilding and replay tools.

To request input-level review, email Lei Zhang at lei.zhang@cnu.edu.cn,
identifying the table/audit, dataset, backbone, window and experiment number.
The author must check source permissions before providing the relevant inputs
and hash manifest, or arranging an authorized access route. Transfer remains
conditional; no unrestricted redistribution or approved controlled-access
service is asserted. Original model checkpoints are not bundled.

## Commands

```bash
python examples/synthetic.py
python cpor.py check-results --evidence evidence
```

Only after authorized inputs are restored in their research layout:

```bash
python cpor.py audit-ledger --evidence evidence --project-root /path/to/research-checkout
python cpor.py replay --project-root /path/to/research-checkout --cell 0 --out /path/to/research-checkout/output/cpor_replay_new
```

This documentation revision adds no training, new predictions or statistical reruns.
