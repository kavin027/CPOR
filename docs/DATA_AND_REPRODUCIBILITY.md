# Data processing and reproduction entry points

## AEMO data provenance

Underlying demand equals operational demand plus measurement-based rooftop PV generation, in MW. Preserve all five regions and observations; the system is their sum. Read operational demand from OPERATIONAL_DEMAND/ACTUAL, field OPERATIONAL_DEMAND; PV from ROOFTOP/ACTUAL, field POWER, TYPE=MEASUREMENT. The retained PV reports have quality indicators0.7 and1; neither group is filtered out. Latest LASTCHANGED wins; conflicting values tied at the latest timestamp fail completeness.

The five-region data cover2928 half-hours from1 September2026 00:30 through1 November2026 00:00, fixed AEST UTC+10. Source archive metadata comprises520 ZIPs containing3594 CSV reports; August trading-day reports complete the September boundary. The operational component has37 negative values,30 in October. Original PV estimates make the resulting underlying demand nonnegative; no clipping or region deletion was used. Regional October minima are6206,5445,1072,910 and4313 MW.

The earlier signed TOTALDEMAND attempt had34 negative SA October values and was not scored. Original registrations remain preserved. Replacement-outcome protocol and86 file hashes (84 Python files, inventory, protocol) were locally saved7 November2026 03:26:17.918634 UTC before component acquisition; related electricity fields had already been inspected. Retain those provenance records without relabeling later extensions as untouched confirmations.

Monthly boundaries: train/validate before28 September00:30; support28 September00:30–30 September06:00 (108); calibration30 September06:30–1 October00:00 (36); score1 October00:30–1 November00:00 (1488). Short-window adapters instead use216 support errors plus36 calibration observations. Every protected monthly path starts at zero once and carries allowance for all1488 releases.

## Resources and checking

The scientific Supplement contains additional results and the derivation needed to understand the robust-map control. Source code, settings and numerical procedures are documented alongside this file. Frozen OR2 v11 holds current unrounded records and stored-prediction inputs; its v10 payload retains historical experiments and registrations. The supplementary-check addendum supplies the verified empirical cells and corrected Li layer summaries.

From an extracted OR2 root:

    python current/check_tables.py
    python current/check_tables.py --trajectories

The first command rebuilds aggregates; the optional second command also needs the complete cached numerical payload and checks stored releases. Neither command retrains a backbone or executes a new forecast-release policy.

The retained-results checker accompanying this version uses stable LaTeX table labels, so changing Supplement numbering does not change the empirical data mapping. Training, numerical acceptance and data-processing settings moved out of the Supplement are preserved verbatim in ARCHIVED_IMPLEMENTATION_EXTRACTS.tex and in the older frozen manuscript. File versions and hashes are supplied with the submission; no original frozen registration or numerical payload is overwritten.
