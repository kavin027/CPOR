"""One-command replay. Requires the numerical environment specified under kernel/."""
import argparse,hashlib,json,subprocess,sys,os
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError

p=Path(__file__).resolve().parent
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--install',action='store_true',help='Create a local .venv and install pinned dependencies')
    args=parser.parse_args()
    for name,h in json.loads((p/'SHA256.json').read_text(encoding='utf-8')).items():
        if hashlib.sha256((p/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed input: '+name)
    exe=sys.executable
    if args.install:
        v=p/'.venv'
        if not v.exists():subprocess.run([exe,'-m','venv',str(v)],check=True)
        exe=str(v/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))
        subprocess.run([exe,'-m','pip','install','-r',str(p/'kernel/requirements-aemo.txt')],check=True)
    env={**os.environ,**dict.fromkeys(['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'],'1')}
    for sub in ('audit_states.py','replay.py'):
        subprocess.run([exe,str(p/sub)],cwd=p,env=env,check=True)

