"""Actual CPU GradScaler skips, short tails, serialization and instrumentation parity."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,copy,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
import card086_common as C
sys.path.insert(0,str(C.R))
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler
from card086_exposure import observe,validate
from gpu_card060_canary import same

def run(instrument=True,split=None):
 torch.manual_seed(42);n=512;src=np.repeat(np.arange(n,dtype='i4'),15);dst=(src+1)%n;w=np.linspace(.01,1,len(src),dtype='f4')
 s=DeviceEdgeSampler(DeviceArrayDataset(np.arange(n,dtype='f4')[:,None],device='cpu'),src,dst,w,n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,device='cpu')
 p=SimpleNamespace(model=torch.nn.Linear(1,1),_train_stats={k:0 for k in ['attempted_batches','optimizer_steps_succeeded','positive_lr_optimizer_steps','amp_overflow_skips','nonfinite_loss_skips','nonfinite_gradient_skips']});opt=torch.optim.AdamW(p.model.parameters(),lr=.0001);scaler=torch.amp.GradScaler('cpu',init_scale=16);seen=[]
 from contextlib import nullcontext
 with observe(p) if instrument else nullcontext():
  for epoch in range(2):
   iter(s)
   for batch in range(len(s)):
    labels=next(s)[-1];p._train_stats['attempted_batches']+=1;opt.zero_grad();loss=p.model(torch.ones((1,1))).sum()
    if batch==4:loss=loss*float('inf')
    before=scaler.get_scale();scaler.scale(loss).backward();scaler.step(opt);scaler.update()
    if scaler.get_scale()<before:p._train_stats['amp_overflow_skips']+=1
    else:p._train_stats['optimizer_steps_succeeded']+=1;p._train_stats['positive_lr_optimizer_steps']+=1
    seen.append(len(labels)-s.num_neg)
    if instrument:validate(p._train_stats)
    if split==p._train_stats['attempted_batches']:
     with tempfile.TemporaryDirectory() as td:
      path=Path(td)/'state.pt';torch.save({'stats':p._train_stats,'model':p.model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'sampler_rng':s.gen.get_state(),'perm':s.perm,'pos':s.pos_idx},path);ck=torch.load(path,weights_only=False);p._train_stats=ck['stats'];p.model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);s.gen.set_state(ck['sampler_rng']);s.perm=ck['perm'];s.pos_idx=ck['pos']
 return p,opt,scaler,s,seen

def main():
 torch.set_num_threads(2);full=run();plain=run(False);checks={};e=validate(full[0]._train_stats)
 assert e['attempted_batches']==10 and e['successful_batches']==8 and e['attempted_positive_slots']==15360 and e['successful_positive_slots']==13104 and e['skipped_positive_slots']==2256 and e['skipped_short_tail_batches']==2;checks['actual_gradscaler_short_tail_overflows']=True
 for k in [2,5]:
  resumed=run(split=k);assert resumed[0]._train_stats==full[0]._train_stats and same(resumed[1].state_dict(),full[1].state_dict()) and torch.equal(resumed[3].gen.get_state(),full[3].gen.get_state());checks['serialized_'+('mid' if k==2 else 'epoch')+'_exposure']=True
 assert same(full[0].model.state_dict(),plain[0].model.state_dict()) and same(full[1].state_dict(),plain[1].state_dict()) and full[2].state_dict()==plain[2].state_dict() and torch.equal(full[3].gen.get_state(),plain[3].gen.get_state());checks['instrumentation_numerical_rng_parity']=True
 bad=copy.deepcopy(full[0]._train_stats);bad['card086_exposure']['attempted_positive_slots']=-1
 try:validate(bad)
 except AssertionError as error:assert str(error)=='invalid exposure counters';checks['corrupt_exposure_rejected']=True
 else:raise AssertionError('corrupt exposure accepted')
 C.write(C.O/'card086-reciprocal-readiness/exposure-contracts.json',{'PASS':True,'checks':checks,'exposure':e,'scope':'Actual CPU GradScaler overflow on both short tails; durable stats serialization at mid/epoch; unchanged model/Adam/scaler/sampler versus instrumentation off. Actual device full-state twins remain required.'});print('EXPOSURE PASS',len(checks),e)
if __name__=='__main__':main()
