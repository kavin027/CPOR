"""Prepare the separately registered 17–23 November evaluation. Does not schedule or backdate a run."""
import argparse, ast, hashlib, json, shutil, subprocess, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORK=HERE/'prospective_20261117'
NEM=timezone(timedelta(hours=10))
FIRST=datetime(2026,11,16,12,30,tzinfo=NEM)
LAST=datetime(2026,11,24,0,0,tzinfo=NEM)
FROZEN='6d8fe62eadcc6e2e84298afff144fb4453e0f58e1f0d88d934a00919f5e3c8a6'
def transform(s,a,b):
    if s.count(a)!=1:raise ValueError('Frozen implementation signature changed: '+a)
    return s.replace(a,b)
def materialize():
    assert hashlib.sha256((HERE/'REGISTRATION.txt').read_bytes()).hexdigest()==FROZEN
    WORK.mkdir(exist_ok=True)
    shutil.copy2(HERE/'REGISTRATION.txt',WORK/'REGISTRATION.txt')
    a=(HERE/'acquire.py').read_text('utf-8-sig').replace('2026-11-12 12:30','2026-11-16 12:30').replace('2026-11-15 00:00','2026-11-24 00:00').replace('len(idx)==120','len(idx)==360').replace('202611121100','202611161100').replace('202611150200','202611240400')
    a=transform(a,"    print('selected files',len(jobs),flush=True)",'''    manifest=H/'collected_urls.json'
    if manifest.exists():jobs.extend((feed,url) for feed,url in json.loads(manifest.read_text()))
    jobs=sorted(set(jobs))
    manifest.write_text(json.dumps(jobs,indent=2))
    print('selected files',len(jobs),flush=True)''')
    a=transform(a,'    d=pd.DataFrame([r for _,rows in out for r in rows]',"""    if __import__('sys').argv[-1]=='--capture':
        print('Partial archive capture; no scores.'); return
    d=pd.DataFrame([r for _,rows in out for r in rows]""")
    i=transform((HERE/'inference.py').read_text('utf-8-sig'),"O=W.parents[1]; N=", "O=next(p for p in W.parents if p.name=='output'); N=")
    s=transform((HERE/'score_new.py').read_text('utf-8-sig'), "O=W.parents[1]", "O=next(p for p in W.parents if p.name=='output')")
    s=transform(s,"assert np.array_equal(np.arange(24,24+len(Y))-last,np.full(len(Y),3))","assert np.all(last<np.arange(24,24+len(Y))) and len(Y)==336")
    c=(HERE/'check.py').read_text('utf-8-sig').replace('120*2*5','360*2*5').replace('len(issue)==96','len(issue)==336').replace('for cut in [0,4]:','for cut in [0]:')
    for f,v in [('acquire.py',a),('inference.py',i),('score_new.py',s),('check.py',c)]:
        ast.parse(v)
        (WORK/f).write_text(v,'utf8')
    shutil.copytree(HERE/'kernel',WORK/'kernel',dirs_exist_ok=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','capture','score'])
    args=p.parse_args()
    materialize()
    if args.stage=='prepare':print('Prepared registered code; no target access.');return
    now=datetime.now(NEM)
    if now < FIRST:raise SystemExit('Registered context does not yet exist.')
    if args.stage=='score' and now < LAST+timedelta(minutes=35):raise SystemExit('Await final first reports; no scoring.')
    if args.stage=='capture':subprocess.run([sys.executable,str(WORK/'acquire.py'),'--capture'],check=True);return
    for fn in ['acquire.py','inference.py','score_new.py','check.py']:subprocess.run([sys.executable,str(WORK/fn)],check=True)
if __name__=='__main__':main()
