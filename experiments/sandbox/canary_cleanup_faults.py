"""Default-off hooks for actual fit-path faults; restricted by caller to512-node18-step canary."""
from contextlib import contextmanager
import torch
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceEdgeSampler
PLAN={5:'gradient',7:'loss'}
def guard(nodes,dose,dest,training_dir,graph,full_graph,off,resume):
 from pathlib import Path
 assert nodes==512 and dose==18 and Path(graph).resolve()!=Path(full_graph).resolve() and not Path(dest).resolve().is_relative_to(Path(training_dir).resolve()),'fault injection restricted to canary fixture'
 assert not off or resume is None,'observer-off control cannot resume'

@contextmanager
def force(p,record,card):
 old_next=DeviceEdgeSampler.__next__;old_zero=torch.optim.AdamW.zero_grad;old_step=torch.optim.AdamW.step
 pending=None;last=None;handles=[];row=None
 record.update(plan=PLAN,rows=[],gradient_cleanup_proved=False,loss_forward_injected=False)
 def ours(opt):return opt.param_groups[0]['params'][0] is next(p.model.parameters())
 def next_batch(s):
  nonlocal pending
  out=old_next(s);npos=len(out[-1])-s.num_neg
  # Actual emitted endpoint IDs independently determine support on ring fixture.
  a=s._last_all_src[:npos];b=s._last_all_dst[:npos];near=int((((b-a)%512)-1<15).sum()) if card=='088' else npos
  pending={'positive':npos,'negative':s.num_neg,'tail':int(npos<s.num_pos),'original15':near,'extra45':npos-near};return out
 def zero(opt,*args,**kwargs):
  nonlocal last,row
  if ours(opt):
   attempt=p._train_stats['attempted_batches']
   if attempt!=last:
    for h in handles:h.remove()
    handles.clear();assert pending is not None,'fault recorder missing selected batch';last=attempt;row=dict(pending,attempt=attempt,successful=False,zero_calls=0,finite_loss_before=p._train_stats.get('finite_loss_batches',0));record['rows'].append(row)
    if PLAN.get(attempt)=='gradient':
     handles.append(next(p.model.parameters()).register_hook(lambda g:torch.full_like(g,float('inf'))))
    if PLAN.get(attempt)=='loss':
     def corrupt(module,args,out):record['loss_forward_injected']=True;return out*float('nan')
     handles.append(p.model.register_forward_hook(corrupt))
   row['zero_calls']+=1
   if PLAN.get(attempt)=='gradient' and row['zero_calls']>1:
    assert p._train_stats['finite_loss_batches']==row['finite_loss_before']+1,'forced gradient did not follow finite loss'
    record['gradient_cleanup_proved']=True
  return old_zero(opt,*args,**kwargs)
 def step(opt,*args,**kwargs):
  result=old_step(opt,*args,**kwargs)
  if ours(opt):assert row is not None;row['successful']=True
  return result
 DeviceEdgeSampler.__next__=next_batch;torch.optim.AdamW.zero_grad=zero;torch.optim.AdamW.step=step
 try:yield
 finally:
  for h in handles:h.remove()
  DeviceEdgeSampler.__next__=old_next;torch.optim.AdamW.zero_grad=old_zero;torch.optim.AdamW.step=old_step

def validate(record,stats,exposure=None):
 assert stats['attempted_batches']==20 and stats['positive_lr_optimizer_steps']==stats['optimizer_steps_succeeded']==18,'forced fault exact dose/attempts'
 assert stats['amp_overflow_skips']==1 and stats['nonfinite_loss_skips']==1 and stats['nonfinite_gradient_skips']==0,'forced fault exact skip counts'
 rows=record['rows'];assert len(rows)==20 and record['gradient_cleanup_proved'] and record['loss_forward_injected'],'actual fit fault path missing'
 assert [r['attempt'] for r in rows if not r['successful']]==[5,7],'unexpected forced fault attempts'
 assert rows[4]['zero_calls']==2 and all(r['zero_calls']==1 for i,r in enumerate(rows) if i!=4),'unexpected zero_grad calls'
 if exposure is not None:
  for prefix in ['attempted','successful']:
   chosen=rows if prefix=='attempted' else [r for r in rows if r['successful']]
   assert exposure[prefix+'_batches']==len(chosen)
   for field,key in [('positive_slots','positive'),('negative_slots','negative'),('short_tail_batches','tail'),('original15_slots','original15'),('extra45_slots','extra45')]:
    if prefix+'_'+field in exposure:assert exposure[prefix+'_'+field]==sum(r[key] for r in chosen),'forced support exposure differs'
 return {'PASS':True,'attempts':20,'successful':18,'gradient_cleanup_attempt':5,'nonfinite_loss_attempt':7,'rows':rows}

def device_controls(fit,C,arm,kw,td,keys):
 import gc
 from pathlib import Path
 from gpu_card060_canary import same
 kw=dict(kw,canary_faults=True);dest=Path(td)/'forced-core-cleanup';p,full,on=fit(arm,18,dest,**kw);validate(on['canary_fault_record'],full['train_stats'],on['exposure']);del p;gc.collect();torch.cuda.empty_cache()
 p,off,off_report=fit(arm,18,Path(td)/'forced-observer-off',canary_observer_off=True,**kw);validate(off_report['canary_fault_record'],off['train_stats']);del p;gc.collect();torch.cuda.empty_cache()
 assert all(same(full[k],off[k]) for k in keys if k!='train_stats'),'forced observer-off endpoint mismatch'
 a=dict(full['train_stats']);b=dict(off['train_stats']);a.pop(C.__name__.replace('_common','')+'_exposure',None)
 assert same(a,b),'forced observer-off core counters mismatch'
 checks=['actual fit finite-loss/nonfinite-gradient cleanup and nonfinite-loss skips','actual fit exact attempted/success/skip/tail/support accounting','actual fit observer-off model/optimizer/scaler/RNG parity'];resumes=[]
 for tag in ['mid','epoch']:
  candidates=[]
  for path in (dest/'ckpts').glob('*.pt'):
   ck=torch.load(path,map_location='cpu',weights_only=False)
   if 0<ck['global_step']<18 and ((tag=='mid' and ck['step_checkpoint']) or (tag=='epoch' and not ck['step_checkpoint'])):candidates.append((ck['global_step'],path))
  assert candidates,'forced fit missing resume checkpoint';_,path=min(candidates)
  p,res,_=fit(arm,18,Path(td)/('forced-'+tag),resume=path,**kw);assert all(same(full[k],res[k]) for k in keys),'forced '+tag+' resume mismatch';resumes.append({'kind':tag,'checkpoint':path.name,'state_sha':C.state_sha(res['model'])});del p,res;gc.collect();torch.cuda.empty_cache();checks.append('actual forced fit '+tag+' resumed endpoint parity')
 return {'PASS':True,'checks':checks,'full_state_sha':C.state_sha(full['model']),'on':on['canary_fault_record'],'off':off_report['canary_fault_record'],'resumes':resumes,'scope':'Actual production fit entry/model/core on512-node canary only; forced fault plan is not available on full data.'}
