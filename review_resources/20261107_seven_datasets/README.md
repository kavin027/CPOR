# CPOR seven-dataset organization revision

The clean manuscript has 29 pages, Supplement/Online Resource 1 has 21 pages, and the cover letter has one page. [FINAL_RECEIPT.json](FINAL_RECEIPT.json) identifies the exact files and the unchanged numerical archive.

## Structure and unchanged evidence

- Section 5.1 introduces all seven datasets and defines AEMO underlying demand alongside the other data.
- Section 5.2 contains the forecasting protocol, including the AEMO continuous month and no-reset rule. The former Section 5.6 is removed.
- Section 5.5 keeps the original statistical rules. The six benchmark summaries/tests and AEMO monthly summaries remain distinct.
- Results/Conclusion text and every table cell are unchanged. No experiment or numerical reanalysis is added.
- Figure 1 PDF/SVG and both earlier frozen submissions are preserved.
- Reference entries are unchanged; moving the AEMO citations changes first-citation order. [CITATION_NUMBER_MAP.json](CITATION_NUMBER_MAP.json) records this numbering change.

## Numerical resources (unchanged OR2 v9)

Use [Numerical_Resources_20261107_v9.zip](../20261107_aemo_three_runs/Numerical_Resources_20261107_v9.zip). Its exact SHA-256 is in the receipt. Older section-number pointers describe the previous manuscript; the current dataset/protocol locations are 5.1/5.2. Main Table 14 and Supplement Tables S24--S28 retain their numerical mapping.

    python -m zipfile -e review_resources/20261107_aemo_three_runs/Numerical_Resources_20261107_v9.zip extracted_or2_v9
    cd extracted_or2_v9
    python aemo_three_runs/check_current.py
    python aemo_three_runs/check_results.py --self-test

The archive is bytewise identical to the previously verified v9. Earlier aggregate and exact stored-trajectory checks therefore apply to the same records; this revision does not describe those as new experiments.

[Previous AEMO three-run manuscript/resources](../20261107_aemo_three_runs/README.md) and [original frozen submission](../20261107_submission_final/README.md) remain unchanged.
