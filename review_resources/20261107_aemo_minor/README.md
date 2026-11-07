# OR2 v7: AEMO minor revision

This version accompanies the 29-page main paper and 20-page supplement. It clarifies the author-recorded freeze, file-hash inventory, ideal direct-projection objective and reproduction coverage. No experiment, numerical result, saved trajectory, original registration or checker changes.

- [Numerical_Resources_20261107_v7.zip](Numerical_Resources_20261107_v7.zip) — complete current numerical archive.
- [RECEIPT.json](RECEIPT.json) — current artifact and manuscript hashes.
- [Previous frozen AEMO resources, v6](../20261107_aemo/README.md) — preserved unchanged.

## Check the current archive

    python -m zipfile -e review_resources/20261107_aemo_minor/Numerical_Resources_20261107_v7.zip review_resources/20261107_aemo_minor/extracted
    cd review_resources/20261107_aemo_minor/extracted
    python aemo_october/check_current.py
    python aemo_october/check_package.py
    python pir_matched/check_pir.py
    python pir_grouped/check_grouped.py
    python projection_order/check_addendum.py

The aggregate checker uses the standard library; the AEMO saved-path checker additionally needs NumPy. It recomputes exact envelopes, cumulative prefix errors and all 53,568 stored rational allowance states, checks floating coherence and rejects a corrupted release. These commands do not rerun projection solvers or train neural models. The six-benchmark aggregate checks and source maps retain their original input-access scope. Twelve portable checks passed after fresh extraction.

## Freeze and hash inventory

The AEMO pre-acquisition freeze is author-recorded, with local timestamps, hashes and amendment records. It is not a third-party preregistration or externally certified timestamp. Its 86 entries comprise 84 Python implementation/dependency files, a directory inventory and a protocol. The separate acquisition manifest identifies 520 raw data ZIPs and their post-download hashes. Prior related TOTALDEMAND inspection, its failed domain eligibility and both local registrations are retained unchanged.

## Paper map

- Main Eq. (24) / Section 6.4 and Supplement S15 explain direct versus two-stage projection. The saved numerical comparison is still projection_order/.
- Main Sections 5.6/6.7 and Table 14 map to Supplement S17 and Tables S24--S27; the complete records remain in aemo_october/.
- Table 8, S20/S21 and the S23 uncertainty are unchanged; see pir_matched/, pir_grouped/ and projection_order/.
- Start aemo_minor/EVIDENCE_SCOPE.md for the clarified checking levels and interpretation; the original registrations and source receipts retain their original bytes.

The same matched results remain: AEMO mean-node gains of 3.82% versus common shrinkage and 2.26% versus two-group shrinkage, under finalized sequential feedback and the NNLS/0.5% contract. Regional/weekly losses and the DLinear protection cost remain reported. Similar direct/two-stage accuracy is an empirical result on the evaluated streams.
