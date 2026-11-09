# Current IS documents and numerical evidence scope

The current IS files are identified by DOCUMENT_VERSION.json. The fixed scientific implementation and configuration version remains scientific-review-20261109. The numerical archive is the byte-identical CPOR_OR2_Compact_v11_20261109.zip, not a new experiment version.

The compact archive supports recomputing unrounded result summaries, aggregating saved node errors into reported levels, and checking stored protection-check summaries. Extract it and run `python check_all.py` with NumPy and pandas. It does not contain complete raw forecast/observation inputs for independently rerunning every release or retraining the backbones. Full stored-input replay depends on separately retained numerical inputs.

These levels of checking are distinct: mathematics; source/configuration review; aggregate arithmetic; node-to-level aggregation; full stored-trajectory replay; raw-data retraining. This revision repeats the existing arithmetic and stored-record checks only. No new model fit, forecast run or outcome-dependent selection was performed.

The earlier IS document map remains preserved and should not be mistaken for the present PDF hashes. Source and Figure 1 hashes are checked against the preceding IS release. The current bundle contains the current PDFs and both the flat manuscript source ZIP and complete editable project ZIP.
