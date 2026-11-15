from pathlib import Path
import json,hashlib,csv
import pandas as pd
W=Path(__file__).resolve().parent;O=W.parents[1]
old=O/'cpor_mlj_major_20261113/evidence/temporal'
oldfiles={r['name']:r['sha256'] for r in json.loads((old/'receipts.json').read_text())}
new=json.loads((W/'receipts.json').read_text())
common=[i for i in new if i['name'] in oldfiles]
assert all(i['sha256']==oldfiles[i['name']] for i in common)
src=pd.read_csv(W/'source_records.csv',parse_dates=['ds','issued'])
prev=pd.read_csv(old/'source_records.csv',parse_dates=['ds','issued'])
score=src[src.ds>=pd.Timestamp('2026-11-13 00:30')]; oldend=prev.ds.max()
oldscore=score[score.source.isin(oldfiles)]
f=score.sort_values(['issued','source']).drop_duplicates(['feed','ds','region'])
both=f[f.source.isin(oldfiles)].groupby('ds').size()
res=dict(previous_last_extracted_interval=str(oldend),overlapping_archived_files=len(common),
        previously_retained_component_rows_for_these_scoring_dates=int((prev.ds>=score.ds.min()).sum()),
        newly_scored_dates_with_any_first_report_already_in_previous_source_zips=len(both),
        newly_scored_dates_with_all_ten_first_reports_already_in_previous_source_zips=int((both==10).sum()),
        previously_cached_scoring_date_keys={str(d):int(n) for d,n in both.items()},
        scored_half_hours=96,context_half_hours=24,coverage='source-receipt overlap check against immediately preceding AEMO historical extension; no claim about what an author may ever have viewed')
(W/'EXPOSURE_AUDIT.json').write_text(json.dumps(res,indent=2));print(res)
