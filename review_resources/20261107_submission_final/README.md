# CPOR submission final — 7 November 2026

Final proofread documents: CPOR_Manuscript_Submission_Final.pdf (29 pages), CPOR_Supplement_Submission_Final.pdf (20 pages), CPOR_Cover_Letter_Submission_Final.pdf (one page). Their exact hashes and the unchanged numerical archive are in [FINAL_RECEIPT.json](FINAL_RECEIPT.json).

The sole scientific-text proofreading change from the reviewed v8 Clean is replacing “immutable registration” with “archived local freeze record” in S17.3. References32/33 already contained official AEMO hyperlinks, verified as active PDF URI annotations. The final fixed resource link and PDF subject metadata are updated. Figure1, every numerical result and original frozen files remain unchanged.

## Numerical evidence

[Numerical_Resources_20261107_v8.zip](Numerical_Resources_20261107_v8.zip) is byte-for-byte the previously verified OR2 v8. No experiment or training is repeated in this final proofreading step. Earlier validation conclusions apply to the same file hash.

    python -m zipfile -e review_resources/20261107_submission_final/Numerical_Resources_20261107_v8.zip review_resources/20261107_submission_final/extracted
    cd review_resources/20261107_submission_final/extracted
    python aemo_october/check_current.py
    python aemo_october/check_package.py
    python pir_matched/check_pir.py
    python pir_grouped/check_grouped.py
    python projection_order/check_addendum.py

The first checker uses the standard library; saved-path checking additionally needs NumPy. It verifies protection from stored references, candidates, releases and observations. Optimizer re-execution and neural training are distinct operations; their scope remains documented in the archive.

## Prior immutable versions

- [Reviewed v8 table-note documents/resources](../20261107_table_notes/README.md).
- [v7 local-freeze/evidence-scope revision](../20261107_aemo_minor/README.md).
- [Original AEMO v6](../20261107_aemo/README.md).

The final proofreading step does not change the empirical conclusions or enlarge their scope. Main Sections5.6/6.7 and Table14 still map to S17 and S24--S27; projection order and uncertainty remain S15/S16, with the same recorded effects and limitations.
