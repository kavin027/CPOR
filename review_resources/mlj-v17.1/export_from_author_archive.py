"""Export CPU inference graphs from the hash-identified AEMO checkpoints; no fitting."""
from pathlib import Path
import hashlib,json,sys
import torch
from torch import nn
import pandas as pd
import numpy as np
from neuralforecast.models import DLinear
k=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()/'output'
prior=Path(sys.argv[2]).resolve()
receipt=json.loads((prior/'predictions/DLinear/1/receipt.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ('ckpt','pir'):assert sha(receipt[name])==receipt[name+'_hash']
sys.path.insert(0,str(out/'c0_external_sota_20261014'))
import pir_adapter as pir
torch.set_num_threads(1)
dlinear=DLinear.load_from_checkpoint(receipt['ckpt'],map_location='cpu',weights_only=False,callbacks=[],logger=False)
class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.decomp=dlinear.decomp;self.trend=dlinear.linear_trend;self.season=dlinear.linear_season
    def forward(self,context):
        seasonal,trend=self.decomp(context.T.contiguous())
        return (self.trend(trend)+self.season(seasonal)).T
history=pd.read_csv(out/'cpor_aemo_underlying_october_20261107/observed.csv',parse_dates=['ds']).set_index('ds')
values=history.loc[:'2026-09-30 06:00'].to_numpy()
mu,sd=values.mean(0),values.std(0)
library=np.stack([values[i-24:i] for i in range(24,len(values))])
tensor=lambda a:torch.from_numpy(a.astype('float32'))
t=dict(lib_x=tensor((library-mu)/sd),lib_y=tensor((values[24:,None]-mu)/sd))
t['x']=t['lib_x'][:0]
adapter=pir.CachedPIR(pir.config(6)).float().cpu()
pir.prepare(adapter,t);adapter.load_state_dict(torch.load(receipt['pir'],map_location='cpu',weights_only=True));adapter.eval()
class Refine(nn.Module):
    def __init__(self):
        super().__init__();self.model=adapter
    def forward(self,x,base,mark):
        self.model.bind(x,base)
        return self.model(x,None,torch.zeros_like(base),mark,mode='refine')[0]
with torch.no_grad():
    torch.jit.trace(Base().eval(),torch.zeros(24,6)).save(str(k/'DLinear.pt'))
    torch.jit.trace(Refine().eval(),(t['lib_x'][:1],torch.zeros(1,1,6),torch.zeros(1,1,5))).save(str(k/'PIR.pt'))
(k/'model_receipt.json').write_text(json.dumps(dict(dataset='AEMO',run=1,backbone='DLinear',torch=torch.__version__,model_sha256=receipt['ckpt_hash'],pir_sha256=receipt['pir_hash'],DLinear_export=sha(k/'DLinear.pt'),PIR_export=sha(k/'PIR.pt'),original_forecast_npz_sha256=receipt['sha256'],retrieval_cutoff='2026-09-30 06:00 NEM time',source_data_sha256=sha(out/'cpor_aemo_underlying_october_20261107/observed.csv'),mean=mu.tolist(),sd=sd.tolist()),indent=2),'utf8')
