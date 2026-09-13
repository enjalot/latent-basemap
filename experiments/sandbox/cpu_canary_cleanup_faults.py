"""CPU test of hooks intended for canary-only real production fit injection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,importlib,copy
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import torch,numpy as np
CARD=sys.argv[1];C=importlib.import_module('card'+CARD+'_common');sys.path.insert(0,str(C.R));E=importlib.import_module('card'+CARD+'_exposure')
from canary_cleanup_faults import force,validate,guard
from cpu_exposure_core_branches import codes
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler
from gpu_card060_canary import same
import logging
logging.disable(logging.CRITICAL)
def run(on):
 torch.manual_seed(42);n=512;k=60 if CARD=='088' else 15
 src=np.repeat(np.arange(n,dtype='i4'),k);dst=((np.arange(n)[:,None]+np.arange(1,k+1))%n).astype('i4').ravel();w=np.tile(np.r_[np.full(15,3.),np.ones(45)],n).astype('f4') if k==60 else np.ones(len(src),'f4')
 if CARD=='088':
  from card088_sampler import matched_sampler
  outer=matched_sampler()
 else:outer=nullcontext()
 with outer:
  p=SimpleNamespace(model=torch.nn.Linear(1,2),_train_stats={key:0 for key in ['attempted_batches','finite_loss_batches','positive_lr_optimizer_steps','optimizer_steps_succeeded','amp_overflow_skips','nonfinite_gradient_skips','nonfinite_loss_skips']});opt=torch.optim.AdamW(p.model.parameters(),lr=.0001);scaler=torch.amp.GradScaler('cpu',init_scale=16)
  s=DeviceEdgeSampler(DeviceArrayDataset(np.arange(n,dtype='f4')[:,None]/n,device='cpu'),src,dst,w,n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,device='cpu');r={}
  for key in ['_stash_ids','_last_all_src','_last_all_dst']:
   if hasattr(s,key):delattr(s,key)
  assert all(not hasattr(s,key) for key in ['_stash_ids','_last_all_src','_last_all_dst'])
  methods=(DeviceEdgeSampler.__next__,torch.optim.AdamW.zero_grad,torch.optim.AdamW.step)
  with (E.observe(p) if on else nullcontext()),force(p,r,CARD):
   for ep in range(4):
    iter(s)
    for _ in range(len(s)):
     x,y,labels=next(s);p._train_stats['attempted_batches']+=1;opt.zero_grad();loss=p.model(x[:8]).square().mean()
     env={'self':p,'optimizer':opt,'scaler':scaler,'torch':torch,'logging':logging,'global_step':p._train_stats['positive_lr_optimizer_steps'],'consecutive_nonfinite_losses':0,'consecutive_nonfinite_gradients':0,'_get_next':lambda:None,'pbar':SimpleNamespace(update=lambda n:None),'loss':loss}
     if not torch.isfinite(loss):exec(codes['loss'],env)
     else:
      p._train_stats['finite_loss_batches']+=1;scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.);env['total_norm']=norm
      if not torch.isfinite(norm):exec(codes['gradient'],env)
      else:scaler.step(opt);scaler.update();p._train_stats['optimizer_steps_succeeded']+=1;p._train_stats['positive_lr_optimizer_steps']+=1
  assert all(not hasattr(s,key) for key in ['_stash_ids','_last_all_src','_last_all_dst']),'temporary sampler ID instrumentation leaked'
  assert methods==(DeviceEdgeSampler.__next__,torch.optim.AdamW.zero_grad,torch.optim.AdamW.step)
  validate(r,p._train_stats,E.validate(p._train_stats) if on else None)
  return {'model':p.model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':s.gen.get_state(),'torch_rng':torch.get_rng_state(),'stats':p._train_stats,'record':r}
a=run(True);b=run(False);a['stats'].pop('card'+CARD+'_exposure');assert same(a,b)
checks={'actual_hook_core_cleanup_loss20attempt18success':True,'observer_off_exact_parity':True,'all_hook_methods_restored':True,'absent_stash_flag_and_ID_attributes_supported_and_restored':True}
for tag,n,d,dest,graph,off,resume in [('full',2000000,18,'/tmp/canary','/tmp/fixture',False,None),('dose',512,60000,'/tmp/canary','/tmp/fixture',False,None),('dest',512,18,str(C.TD/'a'),'/tmp/fixture',False,None),('graph',512,18,'/tmp/canary','/tmp/full',False,None),('offresume',512,18,'/tmp/canary','/tmp/fixture',True,'checkpoint')]:
 try:guard(n,d,dest,C.TD,graph,'/tmp/full',off,resume)
 except AssertionError as ex:assert str(ex) in ['fault injection restricted to canary fixture','observer-off control cannot resume'];checks[tag+'_rejected']=True
 else:raise AssertionError('unsafe fault route accepted')
C.write(C.O/f'card{CARD}-device-proof-readiness/hook-cpu-contracts.json',{'PASS':True,'n_checks':len(checks),'checks':checks,'record':a['record'],'scope':'CPU hooks tested through actual core AST branches; actual full production fit-path proof added to GPU canary but not executed.'});print('HOOK CPU PASS',CARD,len(checks))
