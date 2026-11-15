"""Verify the new MLJ evidence from a clean extraction or the public release tree."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os, csv, hashlib, json, subprocess, sys

base = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(name, *args):
    script = base / name
    subprocess.run([sys.executable,str(script),*map(str,args)],cwd=script.parent,env=env,check=True)

def previous(n):
    return next(p for p in [base.parent/f'v{n}',base.parent/f'mlj-v{n}'] if p.is_dir())

if __name__ == '__main__':
    if not __debug__: raise RuntimeError('Run without -O.')
    for name, value in json.loads((base/'FILES.json').read_text()).items():
        assert digest(base/name)==value, name
    run('integrity/geometry.py')
    run('groups/check.py')
    run('temporal/verify.py')
    run('state/compare.py')
    with TemporaryDirectory(dir=base) as directory:
        temp=Path(directory)
        run('integrity/independent.py','--kit',previous(15),'--audit',previous(16)/'integrity','--out',temp)
        for name in ['exact_margins.csv','summary.csv']:
            with (temp/name).open(newline='') as regenerated,(base/'integrity'/name).open(newline='') as archived:
                assert list(csv.DictReader(regenerated))==list(csv.DictReader(archived)),name
    print('MLJ v17 PASS: both actual delayed publication paths, all 189 fixed-class certificates, 18,144 benchmark issues and paired contrasts.')
