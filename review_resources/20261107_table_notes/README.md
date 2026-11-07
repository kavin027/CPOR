# OR2 v8 — three table notes

Current manuscript: 29 pages; Supplement: 20 pages; cover letter: one page. This revision makes the notes in main Table 14 and Supplement S23/S26 self-contained. No experiments, numerical results, saved arrays, original registrations or checkers change.

- [Numerical_Resources_20261107_v8.zip](Numerical_Resources_20261107_v8.zip) — current complete OR2.
- [RECEIPT.json](RECEIPT.json) — hashes for uniquely named v8 upload files.
- [Previous v7 resources](../20261107_aemo_minor/README.md) — local-freeze/evidence-scope revision, unchanged.
- [Original AEMO v6](../20261107_aemo/README.md) — unchanged.

## Identify the manuscript

Use CPOR_Manuscript_v8_Clean.pdf, CPOR_Supplement_v8_Clean.pdf and CPOR_Cover_Letter_v8.pdf. The 28-page main/v6 resources referenced in the latest review are earlier than the already-delivered 29-page v7 revision. The receipt identifies the current files; dates in renamed PDF filenames alone do not establish manuscript identity.

## Run the checks

    python -m zipfile -e review_resources/20261107_table_notes/Numerical_Resources_20261107_v8.zip review_resources/20261107_table_notes/extracted
    cd review_resources/20261107_table_notes/extracted
    python aemo_october/check_current.py
    python aemo_october/check_package.py
    python pir_matched/check_pir.py
    python pir_grouped/check_grouped.py
    python projection_order/check_addendum.py

The first AEMO checker uses the standard library; saved-path protection replay additionally needs NumPy. It recomputes 53,568 envelope/prefix checks and 53,568 rational allowance states from stored arrays, verifies floating coherence and rejects corruption. This does not execute the projection optimizers or retrain neural models. All twelve old/new checks passed in a clean extraction.

The table_notes/README.md inside OR2 maps the three additions. All result locations and the local-freeze/reproduction-scope explanation in aemo_minor/EVIDENCE_SCOPE.md remain unchanged. The new table notes specify the one-month, one-run-per-backbone AEMO unit; six-dataset uncertainty unit and unadjusted intervals with a two-contrast Holm family; and positive reduction versus positive cost directions.
