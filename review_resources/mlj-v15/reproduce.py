"""One-command reproduction in an isolated Python 3.11 environment."""
import argparse,os,sys,subprocess,venv
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--all',action='store_true',help='check all 189 trajectories')
    p.add_argument('--aemo',action='store_true',help='also regenerate November forecasts')
    args=p.parse_args()
    if sys.version_info[:2] != (3,11):
        p.error('Please run with CPython 3.11 (64-bit).')
    root=Path(__file__).resolve().parent
    envdir=root/'.venv'
    python=envdir/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():venv.EnvBuilder(with_pip=True).create(envdir)
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONNOUSERSITE='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    env.pop('PYTHONPATH',None)
    req=['-r',str(root/'requirements-aemo.txt')] if args.aemo else ['numpy==1.25.2']
    subprocess.run([str(python),'-m','pip','install','--only-binary=:all:',*req],cwd=root,env=env,check=True)
    subprocess.run([str(python),str(root/'verify.py'),*(['--all'] if args.all else [])],cwd=root,env=env,check=True)
    if args.aemo:subprocess.run([str(python),str(root/'replay_aemo.py')],cwd=root,env=env,check=True)

if __name__=='__main__':main()

