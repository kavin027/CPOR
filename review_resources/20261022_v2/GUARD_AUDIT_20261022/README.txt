Final release-guard audit, 22 October 2026
66 current CPOR trajectories, 4752 scored releases.
Run: python GUARD_AUDIT_20261022/check_guard_diagnostics.py
No observations or forecasts are contained in this diagnostic bundle.
The checker verifies exported diagnostic arithmetic and file hashes, not the protection ledger or the original NPZ files. SOURCE_INDEX.json records original source/snapshot hashes; reconstruction agreement is floating-point and scaled, not an exact certificate.
audit_archived_sources.py.txt records the analysis used on local NPZ archives; it requires the original workspace structure and inputs. Candidate projection repair events were not counted. Fixed candidate insets are outside the final guard statistic. Every release still undergoes a guard check even when no change occurs.
