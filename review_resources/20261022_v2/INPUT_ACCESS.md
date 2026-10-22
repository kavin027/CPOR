# Input access for the current saved-trajectory replay

**The download does not contain the NPZ inputs needed to rerun the ledger.**

`LEDGER_REPLAY_20261022/SOURCE_INDEX.json` identifies 66 trajectory pairs by dataset,
backbone, evaluation window and run: `path` is the saved CPOR prediction archive;
`snapshot_path` holds reference values, observations, publications and exact
pre-release allowance numerators/denominators. Each has a SHA256. The list, including
reused first-run models, is authoritative; do not substitute old 500-update DLinear files.

For a relocated project, `--workspace /new/root/IJEPES_RR_RECAP` maps each suffix
after `/IJEPES_RR_RECAP/` to that root. The script verifies hashes before reading data.
Accepted banks and final diagnostics are archived; historical candidate-repair event
logs were not retained and cannot be supplied or inferred from accepted banks.

To request the indexed inputs, contact Lei Zhang at lei.zhang@cnu.edu.cn and identify
the relevant dataset/backbone/window/run and this E1 version. Transmission depends on
the applicable source terms or permission. No approved controlled-access service or
unrestricted redistribution is asserted. Original acquisition routes recorded in the
research provenance are:

- RH/ECommerce: https://github.com/HellOwhatAs/ProCAST (datasets/RH/data.pkl;
  datasets/e-commerce/data.csv and category.json).
- Traffic/Wiki2/Labour: https://nixtlaverse.nixtla.io/datasetsforecast/hierarchical.html
  and the hierarchical-data archive referenced there.
- M5: https://github.com/Nixtla/m5-forecasts, subject to original data/competition terms.

Source availability and anonymization do not by themselves authorize redistribution.
These acquisition routes do not reproduce the archived neural forecasts byte for byte.
Exact saved-stream replay requires the indexed saved inputs; raw-data retraining is
a different operation and is not provided by the new ledger script.
