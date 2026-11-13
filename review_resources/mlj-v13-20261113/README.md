# CPOR · MLJ-V13-20261113
This revision adds a hindsight-comparator proposition and proof, calibrates the reference-only width bound against the November data, and measures how often the reconciled ECommerce candidate actually left the available intervals. TABLE_INDEX.csv maps the old table labels to the current manuscript. The preceding [v12 experimental records](../mlj-v12-20261113/) remain available.

From this directory run \`python width/ecommerce.py\` and \`python tplot/group_horizon.py --check\` to recompute the ECommerce table and verify the analytical curve. With the journal attachment ESM_2.zip extracted, \`python width/check_widths.py PATH_TO_OR2\` verifies the interval calculation against both blocks' stored streams. Running \`python check_results.py\` at the root of the extracted OR2 checks the current and inherited results. FILES.json identifies each submission attachment by SHA-256.

The prospective December protocol and its NNLS reference are unchanged. The proposed additional linear-fill analysis is explicitly unregistered. December outcomes are pending.
