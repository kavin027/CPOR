from pathlib import Path
from urllib.parse import urljoin
import requests, pandas as pd, numpy as np, zipfile, io, csv, json, hashlib, re, time
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha((H/'REGISTRATION.txt').read_bytes())=='6d8fe62eadcc6e2e84298afff144fb4453e0f58e1f0d88d934a00919f5e3c8a6'
regions=['NSW1','QLD1','SA1','TAS1','VIC1']
idx=pd.date_range('2026-11-12 12:30','2026-11-15 00:00',freq='30min'); assert len(idx)==120
def get(url):
    fail=None
    for i in range(4):
        try:
            sess=requests.Session();sess.trust_env=i<2
            target=url.replace("://nemweb.","://www.nemweb.") if i%2 else url
            r=sess.get(target,timeout=(8,25));r.raise_for_status();return r
        except requests.RequestException as e:fail=e;time.sleep(.4)
    raise fail
def file(job):
    feed,url=job
    path=H/'source'/url.rsplit('/',1)[1];path.parent.mkdir(exist_ok=True)
    if not path.exists():path.write_bytes(get(url).content)
    rows=[]
    with zipfile.ZipFile(path) as f:
        for m in f.namelist():
            if not m.lower().endswith('.csv'):continue
            for r in csv.reader(io.StringIO(f.read(m).decode('utf-8-sig'))):
                if r[0]=='C' and len(r)>8:issued=pd.Timestamp(r[5]+' '+r[6])
                if r[0]=='I':cols=r[4:]
                if r[0]!='D':continue
                v=dict(zip(cols,r[4:]));reg=v.get('REGIONID')
                if reg not in regions or (feed=='pv' and v.get('TYPE')!='MEASUREMENT'):continue
                ts=pd.Timestamp(v['INTERVAL_DATETIME'])
                if ts in idx:rows.append((feed,ts,reg,float(v['POWER' if feed=='pv' else 'OPERATIONAL_DEMAND']),issued,v.get('QI',''),path.name))
    return dict(name=path.name,url=url,sha256=sha(path.read_bytes())),rows
def main():
    jobs=[]
    for feed,sub in [('op','Operational_Demand/ACTUAL_HH/'),('pv','ROOFTOP_PV/ACTUAL/')]:
        base='https://nemweb.com.au/Reports/Current/'+sub; listing=get(base).text
        for u in re.findall(r'href=["\x27]([^"\x27]+)',listing,re.I):
            name=u.split('/')[-1];dt=re.search(r'_(202611\d{6})(?:\d{2})?_',name)
            if not dt or not name.lower().endswith('.zip') or feed=='pv' and 'MEASUREMENT' not in name:continue
            if '202611121100'<=dt[1]<='202611150200':jobs.append((feed,urljoin(base,u)))
    print('selected files',len(jobs),flush=True)
    with ThreadPoolExecutor(12) as ex:out=list(ex.map(file,sorted(set(jobs))))
    (H/'receipts.json').write_text(json.dumps([x[0] for x in out],indent=2))
    d=pd.DataFrame([r for _,rows in out for r in rows],columns=['feed','ds','region','MW','issued','qi','source']).sort_values(['issued','source'])
    first=d.drop_duplicates(['feed','ds','region']);assert len(first)==len(idx)*10,len(first)
    def pivot(k):return first[first.feed==k].pivot(index='ds',columns='region',values='MW').loc[idx,regions]
    y=pivot('pv')+pivot('op')
    assert np.isfinite(y).all().all() and (y>=0).all().all()
    ready=first.groupby('ds').issued.max().reindex(idx);lag=(ready-idx).dt.total_seconds()/60
    print('lag',lag.min(),lag.max(),flush=True)
    d.to_csv(H/'source_records.csv',index=False)
    pd.DataFrame({'ds':idx,'last_component_issued':ready.values,'lag_minutes':lag.values}).to_csv(H/'clock.csv',index=False)
    assert (lag>=0).all()
    y.insert(0,'SYSTEM',y.sum(axis=1));y.rename_axis('ds').to_csv(H/'observed.csv')
    print('PASS',y.shape,sha((H/'observed.csv').read_bytes()))
if __name__=='__main__':main()

