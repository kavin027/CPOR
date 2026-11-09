# Current Information Sciences minor-revision documents — 9 November 2026

This is a document-format and metadata update, not a new scientific experiment version.
The preceding IS and KAIS releases remain preserved. [Exact current file hashes](DOCUMENT_VERSION.json)
separate the present PDFs and sources from the fixed scientific implementation at
[scientific-review-20261109](https://github.com/kavin027/CPOR/tree/scientific-review-20261109).

| Role | Current file |
|---|---|
| Main paper, 37 pages | [Clean manuscript](CPOR_IS_Minor_Manuscript_Clean_20261109.pdf) |
| Supplement, 13 pages | [Clean supplement](CPOR_IS_Minor_Supplement_Clean_20261109.pdf) |
| IS letter, 1 page | [Cover letter](CPOR_IS_Minor_Cover_Letter_Clean_20261109.pdf) |
| Editorial Manager sources, flat | [Manuscript LaTeX ZIP](CPOR_IS_Minor_Manuscript_LaTeX_20261109.zip) |
| All editable documents and figures | [Editable project ZIP](CPOR_IS_Minor_Editable_Project_20261109.zip) |
| Numerical evidence, unchanged | [Compact OR2 v11](../20261109_compact_submission/CPOR_OR2_Compact_v11_20261109.zip) |

The PDF update removes 15 duplicated paragraph-heading periods and uses standard CRediT roles.
Tables, equations, results, negative findings and the author's Figure 1 are unchanged.
The original imageGen model version cannot be recovered from the retained provenance and is not invented.

The compact OR2 supports unrounded table arithmetic, saved node-to-level aggregation,
and recorded protection-check summaries. Extract it and run `python check_all.py`
with NumPy and pandas. It excludes the complete raw prediction/observation inputs for
full release replay and backbone retraining. Those are separate levels of verification.
See [resource scope](RESOURCE_SCOPE.md), [revision decisions](REVIEW_RESPONSE.md), and
[the integrity report](INTEGRITY_REPORT.json). No new model training or release experiment.
