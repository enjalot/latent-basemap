"""CPU test of hooks intended for canary-only real production fit injection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,importlib,copy,ast
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import torch,numpy as np
CARD=sys.argv[1];C=importlib.import_module('card'+CARD+'_common');sys.path.insert(0,str(C.R));E=importlib.import_module('card'+CARD+'_exposure')
from canary_cleanup_faults import force,validate,guard
from cpu_exposure_core_branches import codes,tree,core
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler
from gpu_card060_canary import same
import logging
logging.disable(logging.CRITICAL)
# Execute the actual core per-element selection, tanh and fneg-weighted reduction.
selected=[]
for node in ast.walk(tree):
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_per_elem_loss' for t in node.targets):selected.append(node)
 if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='_per_elem_loss':selected.append(node)
assert len(selected)==2
prefix=ast.parse("""def fit(self,qs,targets_for_loss,src_embeddings,dst_embeddings):
 fneg_band_t=fneg_neg_t=fneg_extra_t=fneg_wsum_t=fneg_R_sum=fneg_R_sqsum=0
 fneg_R_min=torch.tensor(float('inf'));fneg_R_max=torch.tensor(float('-inf'));fneg_batches=0
""").body[0]
prefix.body.extend(copy.deepcopy(selected));prefix.body.append(ast.Return(value=ast.Name(id='umap_loss',ctx=ast.Load())))
space={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix],type_ignores=[])),str(core),'exec'),space);core_loss=space['fit']
# Bind real parent flags and run the actual configuration on CPU.
parent=torch.load(C.CHAMP,map_location='cpu',weights_only=False)
assert parent['fneg_weight']==1. and parent['neg_tanh_gamma']==4.
configured=SimpleNamespace(**{k:parent[k] for k in ['architecture','use_dropout','pos_ratio','fneg_weight','neg_tanh_gamma','positive_target_mode']})
C.configure(configured,{'lr':.0001,'dose':18,'rankneg_window':0},np.ones(512,dtype='f4'),[18])
assert configured.fneg_weight==1. and configured.neg_tanh_gamma==4. and configured.rankneg_window==0
def run(on):
 torch.manual_seed(42);n=512;k=60 if CARD=='088' else 15
 src=np.repeat(np.arange(n,dtype='i4'),k);dst=((np.arange(n)[:,None]+np.arange(1,k+1))%n).astype('i4').ravel();w=np.tile(np.r_[np.full(15,3.),np.ones(45)],n).astype('f4') if k==60 else np.ones(len(src),'f4')
 if CARD=='089':
  from card089_weights import weights
  gen=np.random.default_rng(89);neighbors=np.array([gen.choice(np.delete(np.arange(n),i),15,False) for i in range(n)],dtype='i4');dst=neighbors.ravel();mutual=(neighbors[neighbors]==np.arange(n)[:,None,None]).any(2);w=weights(mutual,'rank_count_control')[0].ravel()
 if CARD=='088':
  from card088_sampler import matched_sampler
  outer=matched_sampler()
 else:outer=nullcontext()
 with outer:
  p=SimpleNamespace(model=torch.nn.Linear(1,2),loss_fn=torch.nn.BCELoss(),fneg_weight=1.,neg_tanh_gamma=4.,rankneg_window=0,fneg_lo=.1,fneg_hi=.5,_train_stats={key:0 for key in ['attempted_batches','finite_loss_batches','positive_lr_optimizer_steps','optimizer_steps_succeeded','amp_overflow_skips','nonfinite_gradient_skips','nonfinite_loss_skips']});opt=torch.optim.AdamW(p.model.parameters(),lr=.0001);scaler=torch.amp.GradScaler('cpu',init_scale=16)
  s=DeviceEdgeSampler(DeviceArrayDataset(np.arange(n,dtype='f4')[:,None]/n,device='cpu'),src,dst,w,n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,device='cpu');r={}
  for key in ['_stash_ids','_last_all_src','_last_all_dst']:
   if hasattr(s,key):delattr(s,key)
  assert all(not hasattr(s,key) for key in ['_stash_ids','_last_all_src','_last_all_dst'])
  methods=(DeviceEdgeSampler.__next__,torch.optim.AdamW.zero_grad,torch.optim.AdamW.step,torch.nn.functional.binary_cross_entropy)
  with ((E.observe(p,'rank_count_control',mutual,neighbors) if CARD=='089' else E.observe(p)) if on else nullcontext()),force(p,r,CARD,('rank_count_control',neighbors,mutual) if CARD=='089' else None):
   for ep in range(4):
    iter(s)
    for _ in range(len(s)):
     x,y,labels=next(s);p._train_stats['attempted_batches']+=1;opt.zero_grad();q=p.model(x[:8]).sigmoid().mean(dim=1);q=torch.nan_to_num(q,nan=1e-7,posinf=1-1e-7,neginf=1e-7).clamp(1e-7,1-1e-7);target=torch.zeros_like(q);target[:4]=1.;emb=torch.cat([x[:8],x[:8]],dim=1)
     # An unrelated functional call at attempt7 must remain finite.
     assert torch.isfinite(torch.nn.functional.binary_cross_entropy(q,target,reduction='none')).all()
     other=copy.copy(p);other._train_stats=dict(p._train_stats)
     assert torch.isfinite(core_loss(other,q,target,emb,emb+.1))
     loss=core_loss(p,q,target,emb,emb+.1)
     env={'self':p,'optimizer':opt,'scaler':scaler,'torch':torch,'logging':logging,'global_step':p._train_stats['positive_lr_optimizer_steps'],'consecutive_nonfinite_losses':0,'consecutive_nonfinite_gradients':0,'_get_next':lambda:None,'pbar':SimpleNamespace(update=lambda n:None),'loss':loss}
     if not torch.isfinite(loss):exec(codes['loss'],env)
     else:
      p._train_stats['finite_loss_batches']+=1;scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.);env['total_norm']=norm
      if not torch.isfinite(norm):exec(codes['gradient'],env)
      else:scaler.step(opt);scaler.update();p._train_stats['optimizer_steps_succeeded']+=1;p._train_stats['positive_lr_optimizer_steps']+=1
  assert not hasattr(s,'_stash_ids'),'temporary stash flag leaked'
  if not (CARD=='089' and on):assert all(not hasattr(s,key) for key in ['_last_all_src','_last_all_dst']),'temporary sampler ID instrumentation leaked'
  assert not p.loss_fn._forward_hooks and not p.model._forward_hooks,'loss/model hook leaked'
  assert methods==(DeviceEdgeSampler.__next__,torch.optim.AdamW.zero_grad,torch.optim.AdamW.step,torch.nn.functional.binary_cross_entropy)
  validate(r,p._train_stats,E.validate(p._train_stats) if on else None)
  return {'model':p.model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':s.gen.get_state(),'torch_rng':torch.get_rng_state(),'stats':p._train_stats,'record':r}
# Reproduce why the old model-output NaN hook cannot prove nonfinite loss.
m=torch.nn.Linear(1,2);h=m.register_forward_hook(lambda module,args,out:out*float('nan'))
q=torch.nan_to_num(m(torch.ones(8,1)).sigmoid(),nan=1e-7,posinf=1-1e-7,neginf=1e-7).clamp(1e-7,1-1e-7)
assert torch.isfinite(torch.nn.BCELoss()(q,torch.zeros_like(q)));h.remove()
a=run(True);b=run(False);a['stats'].pop('card'+CARD+'_exposure');assert same(a,b)
checks={'old_forward_nan_sanitized_to_finite_BCE':True,'functional_BCE_restored':True,'actual_configured_per_element_flags_verified':True,'actual_core_tanh_weighted_loss_branch_executed':True,'unrelated_BCE_and_other_owner_untouched':True,'actual_hook_core_cleanup_loss20attempt18success':True,'observer_off_exact_parity':True,'all_hook_methods_restored':True,'absent_stash_flag_and_ID_attributes_supported_and_restored':True}
for tag,n,d,dest,graph,off,resume in [('full',2000000,18,'/tmp/canary','/tmp/fixture',False,None),('dose',512,60000,'/tmp/canary','/tmp/fixture',False,None),('dest',512,18,str(C.TD/'a'),'/tmp/fixture',False,None),('graph',512,18,'/tmp/canary','/tmp/full',False,None),('offresume',512,18,'/tmp/canary','/tmp/fixture',True,'checkpoint')]:
 try:guard(n,d,dest,C.TD,graph,'/tmp/full',off,resume)
 except AssertionError as ex:assert str(ex) in ['fault injection restricted to canary fixture','observer-off control cannot resume'];checks[tag+'_rejected']=True
 else:raise AssertionError('unsafe fault route accepted')
C.write(C.O/f'card{CARD}-observer-readiness/hook-cpu-contracts.json',{'PASS':True,'n_checks':len(checks),'checks':checks,'record':a['record'],'scope':'CPU hooks tested through actual core AST branches; actual full production fit-path proof added to GPU canary but not executed.'});print('HOOK CPU PASS',CARD,len(checks))
