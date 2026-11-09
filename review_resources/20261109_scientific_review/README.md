# Scientific Review — current documents and evidence, 9 November 2026

Fixed tag: `scientific-review-20261109`.

`DOCUMENT_VERSION.json` maps the latest 24-page main paper, 13-page Supplement,
unchanged cover letter, editable sources, and result-check addendum to OR2 v11.
It also identifies the preceding Scientific Main PDFs, which remain unchanged.

## Current numerical checks

Download [the compact current checks](CPOR_OR2_v11_Scientific_Review_Addendum_20261109.zip),
extract them, then run:

    python check_supplement_results.py --supplement /path/to/supplement/main.tex
    python check_li_levels.py --supplement /path/to/supplement/main.tex

The first checks 2,188 retained table components and the twelve Li percentages.
The second independently rebuilds 256 window/level errors and twelve group
comparisons from retained per-node RMSEs. They do not fit models or execute a
forecast-release rule. `RAW_NODE_RMSE.csv` contains errors, not observations.
The unchanged [OR2 v11 records](../20261108_unified/README.md) carry the main
grid; its full submission payload is required for stored-forecast replay.

## Li intermediate-level scope

Current intermediate-level RMSE includes every observed nonsystem, nonbottom
node. Labour therefore includes eight regions plus sixteen sex-region series.
The historical v10 `LI_WINDOWS.csv` used only the sixteen sex-region nodes
for Labour early while retaining all intermediates elsewhere. This changes
eight method/window entries and four group percentages. The original archive
is preserved; it must not be mixed with the uniform all-intermediate definition.

| Comparison | Uniform all-intermediate result | Mixed historical label |
|---|---:|---:|
| Cov vs Factor | 4.38% | 4.25% |
| ML vs Factor | 4.32% | 4.19% |
| Cov vs Diag | 3.95% | 3.85% |
| ML vs Diag | 3.88% | 3.78% |

Original stored-array recomputation supports all twelve paper percentages.
The raw-array audit is explicitly same-family/provisional; its warning is
about inconsistent historical evidence labels. `LI_SCOPE_RESOLUTION.json`,
the per-window difference table and the two Labour node definitions make
this distinction executable. No prediction, experiment or RMSE measurement
was changed by this clarification.

## Fixed experiment specifications

[Code documentation](../../docs/README.md) defines the experts, variance floor,
Cov coefficients, backbone settings and training boundaries, AEMO fields/time
boundaries, coherence residual, exact rational conversion, gap acceptance and
failure handling. These remain in the code documents, outside the scientific
Supplement. The supplied source bundle includes the same documents.
