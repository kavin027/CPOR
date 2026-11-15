# Evidence chronology

All times in the following table are UTC on 15 November 2026. Source observation times and file-generation times in the data are in NEM time (UTC+10). File metadata describes this analysis machine and can be changed by copying. The original receipt contains AEMO URLs and content hashes but did not log HTTP completion timestamps.

|Stage|Time (UTC)|Evidence|
|---|---|---|
|Original local registration saved|02:53:45.227|Original v16 temporal/REGISTRATION.txt modification time; SHA256 6d8fe62eadcc6e2e84298afff144fb4453e0f58e1f0d88d934a00919f5e3c8a6|
|Public registration commit created|02:54:44|Repository commit 0f053d157044550391cd065b6afe7ca27dfba501, git committer date. Exact push-completion second was not recorded.|
|New local acquisition on this run|02:55:27.891–02:55:48.318|Range of last-write timestamps across the 254 original source/*.zip files. First file PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_202611121430_20261112143009.zip; last file PUBLIC_ROOFTOP_PV_ACTUAL_MEASUREMENT_20261115013000_0000000548622383.zip.|
|Extraction complete|02:55:48.540|Final writes: source_records.csv .516, clock.csv .526, observed.csv .540.|
|Original scoring run|02:58:03.295–02:59:14.006|Original score.log creation/last write; stream_scores.csv written at 02:59:13.947.|
|Retrospective Independent comparison recorded|04:21:55.347|v17 PROTOCOL.md modification time; SHA256 c2550ca14b966b8e3effc4bd6cdddb43dd908c1a197bffe85cc9aacc9d6234f6.|
|Matched Full/Independent results recorded|04:26:29.964|Original v17 scores.csv modification time.|

The scored intervals had elapsed before the first registration. Three full composite observations and five component records for a fourth interval were already present in older cached source ZIPs. None of the 96 dated observations was in the preceding extracted series. This evidence supports time separation and the reported sequence of acquisition; it does not establish that no person ever saw a label. The narrowly defined prospective 17–23 November block and the December registration have separate evidential status.

