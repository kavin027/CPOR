"""Run the independent exact replay using only the submitted OR2 and NumPy.

python v17/integrity/reviewer.py
or python reviewer.py --or2 ESM_2.zip --out ./replayed
"""
import argparse,csv,hashlib,json,sys,tempfile,zipfile
from pathlib import Path
import independent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    here=Path(__file__).resolve().parent
    p.add_argument('--or2',type=Path,help='Optional unextracted submitted ZIP')
    p.add_argument('--out',type=Path,default=here/'replayed')
    a=p.parse_args()
    if __debug__ is False:raise RuntimeError('run without -O')
    a.out.mkdir(parents=True,exist_ok=True)
    def run(root):
        sys.argv=['independent','--kit',str(root/'v15'),'--audit',str(root/'v16'/'integrity'),'--out',str(a.out)]
        independent.main()
    if a.or2 is None:run(here.parents[1])
    else:
        with tempfile.TemporaryDirectory(dir=a.out) as temp,zipfile.ZipFile(a.or2) as z:
            base=Path(temp)
            required=['v15/inputs.csv','v16/integrity/certificate_margins.csv']
            for entry in required:
                dest=base/entry;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(entry))
            cases=independent.rows(base/'v15/inputs.csv')
            for r in cases:
                assert r['case'].replace('_','').isalnum() and r['download']=='paths/'+r['case']+'.npz'
                for entry in ('v15/'+r['download'],'v16/integrity/differences/'+r['case']+'.npz'):
                    dest=base/entry;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(entry))
            run(base)
if __name__=='__main__':main()

