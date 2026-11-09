# Fresh Li hierarchical percentage audit

Scope: Supplement S8.4, 12 percentages. Same-family; provisional. Stored raw arrays were found for all 16 windows and all four methods. No new experiment or manuscript edit.

Formula: average the two unrounded window RMSEs within each dataset/backbone group, compute 100×(1−CPOR/Li), then equally average the eight group percentages. Node RMSE is over 72 forecasts; level RMSE is the mean of node RMSEs.

Middle = all internal non-root nodes. Labour nodes 1..24 comprise 8 regions and 16 sex-region nodes. The archive uses only sex-region nodes 9..24 in Labour early, while the other windows use all intermediates. This is confirmed directly from raw arrays and calculation code.

|Level|CPOR|Li reference|Paper %|Archive %|Public %|Raw all-intermediate %|Public−archive pp|
|---|---|---|---:|---:|---:|---:|---:|
|system|Cov|Factor|4.77|4.769243766897009|4.769243766897002|4.769243766897002|-7.105427357601002e-15|
|system|ML|Factor|4.58|4.5790806096968915|4.57908060969689|4.57908060969689|-1.7763568394002505e-15|
|system|Cov|Diag|4.86|4.8558827004041|4.855882700404096|4.855882700404096|-4.440892098500626e-15|
|system|ML|Diag|4.67|4.6718124291698615|4.671812429169865|4.671812429169865|3.552713678800501e-15|
|middle|Cov|Factor|4.38|4.253153880386588|4.384590362081903|4.384590362081903|0.13143648169531552|
|middle|ML|Factor|4.32|4.187202353359265|4.320893196661544|4.320893196661544|0.1336908433022792|
|middle|Cov|Diag|3.95|3.8476815536216513|3.9457440168589013|3.9457440168589013|0.09806246323724999|
|middle|ML|Diag|3.88|3.781557285824609|3.8818525679146023|3.8818525679146023|0.10029528208999317|
|bottom|Cov|Factor|5.52|5.519264789008589|5.519264789008588|5.519264789008588|-8.881784197001252e-16|
|bottom|ML|Factor|5.51|5.506671755531456|5.506671755531453|5.506671755531453|-2.6645352591003757e-15|
|bottom|Cov|Diag|5.02|5.018853960245851|5.018853960245853|5.018853960245853|1.7763568394002505e-15|
|bottom|ML|Diag|5.01|5.006104513229053|5.006104513229054|5.006104513229054|8.881784197001252e-16|

All 12 paper percentages agree with raw-array recomputation to displayed precision. Verdict WARN concerns inconsistent archive evidence, not an identified paper-number error. Public input agrees exactly with raw recomputation for all 256 common records.

Intermediate-level without a named restriction most naturally denotes all non-root, non-leaf internal nodes here, matching the 14 original windows and the summing matrices. Because Labour has both region and sex-region internal levels, an explicit all-intermediate-node definition would remove residual language ambiguity. Sex-region-only is a valid separately named level but is not interchangeable with all intermediate nodes.

Four-method Labour window values for both middle definitions are in LABOUR_MIDDLE_DEFINITIONS_FOUR_METHODS.csv. Complete window-level differences, node identities, hashes and limitations are in the JSON and CSV siblings.

Archive window-to-group maximum absolute difference: 0.0. Labour early archive vs sex-region-only raw maximum absolute difference: 0.0.
