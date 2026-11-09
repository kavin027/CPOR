# Numerical evidence and document scope

The fixed scientific implementation remains scientific-review-20261109, and all underlying experiment records remain unified-v11-20261108. This version displays previously saved Coherent PIR results and adds descriptive paired arithmetic; it does not train or execute a forecasting policy.

The compact OR2 now additionally contains presentation/COHERENT_PIR_TABLES.csv (42 three-run means/SDs), presentation/COHERENT_PIR_EFFECTS.csv (12 paired contrasts) and check_coherent_pir.py. All prior numeric records are byte-identical. Extract the ONE current compact OR2 and run `python check_all.py` with NumPy and pandas; this runs old and new arithmetic checks together.

Checking saved arithmetic, node-to-level aggregation and recorded protection-check summaries is distinct from independently replaying raw forecast streams or retraining backbones. Complete raw replay/training inputs remain excluded. DOCUMENT_VERSION.json maps this document version separately from the scientific-code and numeric-input versions.
