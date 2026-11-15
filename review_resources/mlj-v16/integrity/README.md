# Certificate scope, training provenance and candidate parameters

From an extracted v16 OR2 archive, use CPython 3.11 and NumPy 1.25.2 and run:

    python v16/integrity/certificate_scope.py --kit v15 --legacy legacy
    python v16/integrity/sensitivity.py --kit v15 --legacy legacy

Adjust the path to the scripts if this folder is placed elsewhere. The first command reconstructs all verified intervals from v15's public trajectory inputs, and independently computes per-trajectory numerical and representation-transfer margins using the shipped small direction differences. It does not call an optimizer or read neural checkpoints. The second repeats the two matched release comparisons using the unrounded stream scores. A full clean copy of v15 OR2 and the small difference arrays are sufficient for these commands.

For holders of the original retained forecast archive, append --source PATH_TO_DIRECTORY_CONTAINING_streams to the first command. It then matches the context and Coherent PIR file hashes and regenerates each direction discrepancy from the aggregation matrix and nonnegative witnesses as dyadic integers. --build is a maintainer's option for generating new files; omit it during verification.

## What is certified

At fixed issue t, e_b=S u_b-b and e_z=S u_z-z are exactly determined by the archived arrays. The closed map uses D2=z-b-D1. The literal second-group sum uses D2+e_z-e_b with the same first-group direction. Each sidecar in differences/ stores both endpoint residual arrays as lossless signed integer mantissas with a common binary exponent. Their column order follows the archived node order. The per-trajectory masses, cut anchors, directions and outward arithmetic are those supplied in v15.

certificate_margins.csv reports the signed strict-lower margin L-Full_upper and reverse-witness margin Full_lower-U in original RMSE units. It also reports the optimum enclosure width and the Full score bracket width. Each positive margin is rounded down. The literal-direction lower bound recomputes the affine minorants for the objective and every historical contract row after changing the direction. For a release reconstructed wholly from witnesses, the physical-transfer deduction bounds the offset e_b in all past losses, the current envelope and the RMSE objective. The stored contract reference is held fixed. If a rule issues the stored anchor as a fallback it remains covered. The upper/reverse witnesses certify the closed map only. The enclosure width is distinct from the price of changing the geometric representation. This audit does not select new NNLS solutions or new groups.

Training provenance: identified_combinations.csv lists the 63 dataset/backbone/training-run combinations. The primary provenance-qualified comparison uses the 55 with include_primary=1, omitting both windows of the 8 affected combinations. They retain 110 windows and 55 continuous paths. A separate full-grid report covers every archived prediction stream. The filter addresses one known identity gap. It supplies no assertion that all networks have been retrained or that independent training exactly regenerates any particular cache.

On the identified subset the closed certificate strictly separates 84 of 110 windows and 41 of 55 continuous paths. In 26 and 14 respectively a closed-path upper witness beats Full. The signed separation remains positive in every original positive case under both literal directions and the full witness-offset deduction, including on the full grid (94 windows, 45 continuous). Fixed two-group online vs Full gains match the original archive to the displayed precision. Exact node-limit checks, tolerance-based aggregation and whole-trajectory performance comparisons are separate levels of verification.

The files parameter_sources.json and parameter_source.txt identify the actual functions and their configuration lines. The three TeX fragments document the proof and parameters. Full history files for the 32 reconstruction-only stages are unavailable. For the right level of reproduction use:
- Recalculate result tables: OR2 check_results.py (and the v16 commands above).
- Regenerate a publication sequence from frozen inputs: v15/reproduce.py --aemo (includes exact prefix tests in both rule paths); other caches identified in the legacy source map have restricted or separately requested full replay inputs.
- Check a fixed-class certificate: v15/reproduce.py or v15/reproduce.py --all, plus the v16 representation check above.
- Reproduce a supplied forecast from existing checkpoints: only when the required predictors, fitted adapters, data versions and input clock are available.
- Reproduce training: preserve the provenance flags in legacy/STAGE_MAP.csv. Downstream agreement does not identify missing weights.

