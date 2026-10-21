# ProCAST branch identification

The comparator is the adapted learned-oblique implementation, not the
orthogonal NNLS branch. The frozen author class `_TorchPinv` is called with
`masked=False`, `max_iter=100000`, float64 arithmetic and a mapping reused
across issues. Each fit uses up to 72 preceding forecast/label pairs with total
weight 0.5, plus the current raw forecast as its own target with weight 0.5;
the current true label is excluded. The nonnegative latent forecast is mapped
back through S. The original routines are `torch_oblique` and
`tntnn_orthogonal` in the author's `reconciliare_opt.py`.

Frozen research locators: `output/rr_recap_evidence_extension_20261002/procast_author.py`
(`_TorchPinv`, `torch_oblique`, `tntnn_orthogonal`),
`output/rr_recap_strong_baseline_20261006/run.py::run`, and
`output/c0_seed_validation_20261016/gpu.py::procast`.
Our NNLS column solves the same Euclidean cone projection problem as the
orthogonal branch, with SciPy NNLS/lsq_linear instead of TNT-NN. It is not a
claim of bit-identical author-code replication. The manuscript's 28.474%
comparison concerns the complete adapted learned-oblique procedure under
the shared forecasting protocol, not the author's original experiment.

Sources:
https://ojs.aaai.org/index.php/AAAI/article/view/39961/43922
https://github.com/HellOwhatAs/ProCAST/blob/main/reconciliare_opt.py
