# Machine Learning · MLJ-R2-20261113
**CPOR: Cumulative Protection for Online Reconciliation with Nodewise Error Limits**  
Lei Zhang, Capital Normal University

The submission PDFs and OR2 are identified by SHA-256 in FILES.json. The author-supplied manuscript and OR2 are separate journal uploads. The previous [R1 materials](../mlj-r1-20261113/) remain available.

| Verification | Input and command |
|---|---|
| Recalculate the two November contrasts and count the fixed-partition separations | Run `python check_public.py` in this directory (NumPy and pandas). The inputs include every model run, reference and release rule, plus the outward-rounded bound records. |
| Inspect legacy training provenance | STAGE_MAP.csv identifies 32 reconstructed fitting boundaries. AFFECTED_STREAMS.csv maps their downstream uses and table numbers. MATCHED_SENSITIVITY.csv gives the paired gains after omitting the affected runs. In the journal's Online Resource 2, `python legacy/check_legacy.py` checks these rows against the fitting, stream, adapter, certificate and paired-result records. |
| November rules and underlying conditional protection | From the extracted OR2 root run `python check_results.py`. It recomputes nodewise prefix errors and available allowances under pending reservations using the stored releases. It also checks the fixed-group inputs and reported solver diagnostics. |
| Mechanism of the strong-reference results | The public arm_metrics.csv and pair_metrics.csv give the summaries used in Supplementary Note S12, Table S29. In OR2, `python mechanism/check_mechanism.py` derives them from the dated releases, reference predictions and returned observations. |
| Regenerate historical model predictions | Requires the upstream data, preprocessing and identified model weights. The 32 legacy stages have reconstructed schedules but no retained checkpoint identities; saved comparisons do not restore those identities. |

The two November blocks have disjoint scoring dates in the same market. Their nine runs use common observations. December is a registered future test with no outcomes in this release.

