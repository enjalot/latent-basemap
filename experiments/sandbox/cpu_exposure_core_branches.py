"""CPU injection through unchanged core AST branches, actual sampler/Adam/GradScaler."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import ast,copy,hashlib,importlib,importlib.util,json,logging,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np,torch
CARD=sys.argv[1];assert CARD in ('086','088')
C=importlib.import_module('card'+CARD+'_common');sys.path.insert(0,str(C.R));E=importlib.import_module('card'+CARD+'_exposure')
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler
from gpu_card060_canary import same
core=C.R/'basemap/pumap/parametric_umap/core.py';tree=ast.parse(core.read_text());branches={}
for node in ast.walk(tree):
 if isinstance(node,ast.If) and ast.unparse(node.test) in ['not torch.isfinite(loss)','not bool(torch.isfinite(total_norm))']:
  key='loss' if ast.unparse(node.test)=='not torch.isfinite(loss)' else 'gradient'
  if key in branches:raise AssertionError('ambiguous core branch')
  branches[key]=node
assert set(branches)=={'loss','gradient'}
codes={k:compile(ast.fix_missing_locations(ast.Module(body=[ast.For(target=ast.Name(id='_once',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1)],keywords=[]),body=[copy.deepcopy(v)],orelse=[])],type_ignores=[])),str(core),'exec') for k,v in branches.items()}
logging.disable(logging.CRITICAL)

def run(instrument=True,plan='gradient_tail',resume=None,stop=None,observe=None,counter_fault=None,extra_clears=True,arm="mixture"):
 torch.manual_seed(42);n=512;k=60 if CARD=='088' else 15
 src=np.repeat(np.arange(n,dtype='i4'),k);dst=((np.arange(n)[:,None]+np.arange(1,k+1))%n).astype('i4').ravel()
 w=np.tile(np.r_[np.full(15,6. if arm=='original15' else 3.),np.zeros(45) if arm=='original15' else np.ones(45)],n).astype('f4') if CARD=='088' else np.linspace(.01,1,len(src),dtype='f4')
 if CARD=='088':
  from card088_sampler import matched_sampler
  outer=matched_sampler()
 else:outer=nullcontext()
 with outer:
  p=SimpleNamespace(model=torch.nn.Linear(1,2),_train_stats={z:0 for z in ['attempted_batches','optimizer_steps_succeeded','positive_lr_optimizer_steps','amp_overflow_skips','nonfinite_loss_skips','nonfinite_gradient_skips']})
  opt=torch.optim.AdamW(p.model.parameters(),lr=.0001);scaler=torch.amp.GradScaler('cpu',init_scale=16)
  s=DeviceEdgeSampler(DeviceArrayDataset(np.arange(n,dtype='f4')[:,None]/n,device='cpu'),src,dst,w,n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,device='cpu');s._stash_ids=True
  rows=[];start_epoch=0;start_batch=0
  if resume is not None:
   p.model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler']);p._train_stats=copy.deepcopy(resume['stats']);rows=copy.deepcopy(resume['rows']);torch.set_rng_state(resume['torch_rng']);s.gen.set_state(resume['sampler_rng']);s.perm=resume['perm'].clone();s.pos_idx=resume['pos'];s.batch_no=resume['batch_no'];start_epoch=resume['next_epoch'];start_batch=resume['next_batch']
   if CARD=='088':
    s._card088_epoch_counter=resume['positive_epoch_counter'];s._card088_positive_gen=torch.Generator();s._card088_positive_gen.set_state(resume['positive_rng'])
  def state(epoch,batch):
   out={'model':copy.deepcopy(p.model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'scaler':copy.deepcopy(scaler.state_dict()),'stats':copy.deepcopy(p._train_stats),'rows':copy.deepcopy(rows),'torch_rng':torch.get_rng_state(),'sampler_rng':s.gen.get_state(),'perm':s.perm.clone(),'pos':s.pos_idx,'batch_no':s.batch_no,'next_epoch':epoch,'next_batch':batch}
   if CARD=='088':out.update(positive_epoch_counter=s._card088_epoch_counter,positive_rng=s._card088_positive_gen.get_state())
   return out
  with ((observe or E.observe)(p) if instrument else nullcontext()):
   for epoch in range(start_epoch,2):
    if not(epoch==start_epoch and start_batch):iter(s)
    for batch in range(start_batch if epoch==start_epoch else 0,len(s)):
     x,y,labels=next(s);npos=len(labels)-s.num_neg
     ids=(s._last_all_src.clone(),s._last_all_dst.clone());support=int((((ids[1][:npos]-ids[0][:npos])%n)-1<15).sum()) if CARD=='088' else npos
     p._train_stats['attempted_batches']+=1;opt.zero_grad(set_to_none=True)
     if counter_fault and batch==1:
      p._train_stats['attempted_batches']+=1 if counter_fault=='unselected_increment' else -1
     if extra_clears:opt.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True) # deliberate duplicate clears; no new selected batch
     loss=p.model(x[:8]).square().mean();assert torch.isfinite(loss)
     fault='gradient' if batch==1 or (batch==4 and plan=='gradient_tail') else ('loss' if batch==2 or (batch==4 and plan=='loss_tail') else 'none')
     env={'self':p,'optimizer':opt,'scaler':scaler,'torch':torch,'logging':logging,'global_step':p._train_stats['positive_lr_optimizer_steps'],'consecutive_nonfinite_losses':0,'consecutive_nonfinite_gradients':0,'_get_next':lambda:None,'pbar':SimpleNamespace(update=lambda n:None),'loss':loss}
     if fault=='loss':
      env['loss']=loss*float('nan');exec(codes['loss'],env)
     else:
      hook=next(p.model.parameters()).register_hook(lambda g:torch.full_like(g,float('inf'))) if fault=='gradient' else None
      scaler.scale(loss).backward()
      if hook:hook.remove()
      scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.);env['total_norm']=norm
      if fault=='gradient':assert torch.isfinite(loss) and not torch.isfinite(norm);exec(codes['gradient'],env)
      else:
       assert torch.isfinite(norm);scaler.step(opt);scaler.update();p._train_stats['optimizer_steps_succeeded']+=1;p._train_stats['positive_lr_optimizer_steps']+=1
     rows.append({'positive':npos,'negative':s.num_neg,'tail':int(npos<s.num_pos),'original15':support,'extra45':npos-support,'success':fault=='none','fault':fault,'ids':ids})
     if instrument:E.validate(p._train_stats)
     ne,nb=(epoch+1,0) if batch+1==len(s) else (epoch,batch+1)
     if stop==len(rows):return state(ne,nb)
  return state(2,0)

def checks_main():
 torch.set_num_threads(2);checks={};details={}
 import itertools
 for arm,plan in itertools.product(['original15','mixture'],['gradient_tail','loss_tail']):
  full=run(plan=plan,arm=arm);plain=run(False,plan=plan,arm=arm);e=E.validate(full['stats']);rows=full['rows']
  for pre in ['attempted','successful']:
   selected=rows if pre=='attempted' else [v for v in rows if v['success']]
   assert e[pre+'_batches']==len(selected)
   for field,source in [('positive_slots','positive'),('negative_slots','negative'),('short_tail_batches','tail')]:assert e[pre+'_'+field]==sum(v[source] for v in selected)
   if CARD=='088':
    for field in ['original15','extra45']:assert e[pre+'_'+field+'_slots']==sum(v[field] for v in selected)
  assert len(rows)==10 and e['successful_batches']==4 and e['skipped_short_tail_batches']==2
  assert full['stats']['amp_overflow_skips']==(4 if plan=='gradient_tail' else 2) and full['stats']['nonfinite_loss_skips']==(2 if plan=='gradient_tail' else 4)
  checks[arm+'_'+plan+'_exact_core_skip_tail_support']=True
  for key in full:
   if key!='stats':assert same(full[key],plain[key]),key
  checks[arm+'_'+plan+'_offpath_model_optimizer_scaler_rng_ids']=True
  for split in [2,3,5]:
   partial=run(plan=plan,stop=split,arm=arm)
   with tempfile.TemporaryDirectory() as td:
    f=Path(td)/'state.pt';torch.save(partial,f);restored=torch.load(f,weights_only=False)
   result=run(plan=plan,resume=restored,arm=arm);assert same(full,result)
   checks[arm+'_'+plan+'_fresh_observer_model_sampler_resume_'+str(split)]=True
  details[arm+'_'+plan]=e
  if arm=='original15':assert e['attempted_extra45_slots']==e['successful_extra45_slots']==0
 for fault in ['reset','unselected_increment']:
  try:run(counter_fault=fault)
  except AssertionError as ex:assert str(ex)=='exposure loop-entry mismatch';checks[fault+'_rejected']=True
  else:raise AssertionError('bad core counter accepted')
 oldroot=C.R.parent/('card086-reciprocal-code' if CARD=='086' else 'card088-code');spec=importlib.util.spec_from_file_location('old_exposure',oldroot/f'experiments/sandbox/card{CARD}_exposure.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 # Remove the synthetic extra clears to reproduce the original defect specifically in the unchanged core cleanup branch.
 # Separate direct invocation uses actual observer, sampler and a finite-loss nonfinite-gradient injection.
 try:run(observe=old.observe,extra_clears=False)
 except AssertionError as ex:
  import traceback
  frames=traceback.extract_tb(ex.__traceback__);assert str(ex)=='exposure loop-entry mismatch' and any(f.filename==str(core) and f.lineno==branches['gradient'].body[0].lineno for f in frames);checks['old_observer_actual_core_cleanup_reproduced']=True
 else:raise AssertionError('old observer bug not reproduced')
 r={'PASS':True,'card':CARD,'n_checks':len(checks),'checks':checks,'details':details,'core_sha':C.sha(core),'branches':{k:{'line':v.lineno,'ast_sha':hashlib.sha256(ast.dump(v,include_attributes=False).encode()).hexdigest()} for k,v in branches.items()},'scope':'Actual unchanged core nonfinite-loss and nonfinite-gradient AST branches executed in CPU single-attempt harness; actual sampler, AdamW, CPU GradScaler, finite forward/nonfinite backward injection, explicit repeated clears. CPU512-node2epoch fixtures; not whole production fit/device proof.'}
 C.write(C.O/f'card{CARD}-observer-readiness/core-branch-controls.json',r);print('CORE BRANCH CONTROLS PASS',CARD,len(checks))
if __name__=='__main__':checks_main()
