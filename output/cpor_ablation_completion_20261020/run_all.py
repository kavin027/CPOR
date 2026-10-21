from pathlib import Path
import subprocess,sys,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
H=Path(__file__).resolve().parent;R=H.parents[1]
def one(job,gate=False):
 ci,seed,arm=job;log=H/'logs'/f'{"gate_" if gate else ""}{ci}_{seed}_{arm}.log';log.parent.mkdir(exist_ok=True)
 with log.open('a',encoding='utf-8') as f:p=subprocess.run([sys.executable,str(H/'worker.py'),str(ci),str(seed),arm]+(['gate'] if gate else []),cwd=R,stdout=f,stderr=subprocess.STDOUT)
 return dict(job=job,exit=p.returncode,log=str(log))
if __name__=='__main__':
 reg=json.loads((H/'REGISTRATION.json').read_text());start=time.time()
 gates=[one([0,42,arm],True) for arm in ['SquareDirectional','RootDirectional']]
 (H/'GATE.json').write_text(json.dumps(gates,indent=2));assert all(x['exit']==0 for x in gates),gates
 print('GATE_PASSED',flush=True);events=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(one,j) for j in reg['jobs']]):
   result=f.result();events.append(result);(H/'PROCESS_EVENTS.json').write_text(json.dumps(events,indent=2))
   print('FINISHED',len(events),'/',len(reg['jobs']),result['job'],'exit',result['exit'],'seconds',round(time.time()-start),flush=True)
 assert all(x['exit']==0 for x in events),[x for x in events if x['exit']]
 (H/'RUN_STATUS.json').write_text(json.dumps(dict(completed=len(events),required=96,seconds=time.time()-start,passed=True),indent=2))
