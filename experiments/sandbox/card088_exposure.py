"""Exact attempted/successful pair-slot accounting in persisted train_stats.

No inference from successful dose or mean batch size. The selected device batch
is counted at optimizer.zero_grad (actual loop entry), and successful exposure
at the real AdamW.step call, which GradScaler omits on overflow. No RNG/draws.
"""
from contextlib import contextmanager
import torch
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceEdgeSampler
SCHEMA='card088-pair-exposure-v1'
FIELDS=('attempted_batches','successful_batches','attempted_positive_slots','successful_positive_slots','attempted_negative_slots','successful_negative_slots','attempted_short_tail_batches','successful_short_tail_batches','attempted_original15_slots','successful_original15_slots','attempted_extra45_slots','successful_extra45_slots')
def validate(stats):
 e=stats.get('card088_exposure');assert isinstance(e,dict) and e.get('schema')==SCHEMA,'missing exposure state'
 assert all(type(e.get(k)) is int and e[k]>=0 for k in FIELDS),'invalid exposure counters'
 assert e['attempted_batches']==stats['attempted_batches'],'attempted exposure counter mismatch'
 assert e['successful_batches']==stats['optimizer_steps_succeeded']==stats['positive_lr_optimizer_steps'],'successful exposure counter mismatch'
 for prefix in ['attempted','successful']:
  assert e[prefix+'_positive_slots']==e[prefix+'_batches']*e['normal_positive_slots']-e[prefix+'_short_tail_batches']*(e['normal_positive_slots']-e['short_tail_positive_slots']),'positive exposure arithmetic mismatch'
  assert e[prefix+'_original15_slots']+e[prefix+'_extra45_slots']==e[prefix+'_positive_slots'],'support exposure partition mismatch'
  assert e[prefix+'_negative_slots']==e[prefix+'_batches']*e['negative_slots_per_batch'],'negative exposure arithmetic mismatch'
 for k in ['batches','positive_slots','negative_slots','short_tail_batches','original15_slots','extra45_slots']:
  assert e['successful_'+k]<=e['attempted_'+k],'exposure ordering mismatch'
 assert e['attempted_batches']-e['successful_batches']==sum(stats[k] for k in ['amp_overflow_skips','nonfinite_loss_skips','nonfinite_gradient_skips']),'skip exposure counter mismatch'
 return dict(e,skipped_batches=e['attempted_batches']-e['successful_batches'],skipped_positive_slots=e['attempted_positive_slots']-e['successful_positive_slots'],skipped_negative_slots=e['attempted_negative_slots']-e['successful_negative_slots'],skipped_short_tail_batches=e['attempted_short_tail_batches']-e['successful_short_tail_batches'])
def support_fractions(e):
 result={}
 for prefix in ['attempted','successful']:
  total=e[prefix+'_positive_slots'];assert total>0,'empty support exposure'
  result[prefix]={part:e[prefix+'_'+part+'_slots']/total for part in ['original15','extra45']}
 return result

@contextmanager
def observe(p):
 advance=DeviceEdgeSampler.__next__;zero=torch.optim.AdamW.zero_grad;step=torch.optim.AdamW.step;pending=None
 def ours(opt):
  return opt.param_groups[0]['params'][0] is next(p.model.parameters())
 def next_batch(s):
  nonlocal pending
  out=advance(s);assert s.weighted_edge_sampling and s.positive_target_mode=='binary'
  npos=len(out[-1])-s.num_neg;pending=(npos,s.num_neg,int(npos<s.num_pos),s.num_pos,s.n_pos%s.num_pos or s.num_pos,int((s._card088_last_edge_idx%60<15).sum()));return out
 def clear(opt,*args,**kwargs):
  if ours(opt):
   assert pending is not None,'exposure missing selected batch'
   e=p._train_stats.setdefault('card088_exposure',dict(schema=SCHEMA,normal_positive_slots=pending[3],negative_slots_per_batch=pending[1],short_tail_positive_slots=pending[4],**{k:0 for k in FIELDS}))
   npos,nneg,tail=pending[:3];e['attempted_batches']+=1;e['attempted_positive_slots']+=npos;e['attempted_negative_slots']+=nneg;e['attempted_short_tail_batches']+=tail;e['attempted_original15_slots']+=pending[5];e['attempted_extra45_slots']+=npos-pending[5]
   assert e['attempted_batches']==p._train_stats['attempted_batches'],'exposure loop-entry mismatch'
  return zero(opt,*args,**kwargs)
 def update(opt,*args,**kwargs):
  result=step(opt,*args,**kwargs)
  if ours(opt):
   assert opt.param_groups[0]['lr']>0,'exposure zero-LR step forbidden';e=p._train_stats['card088_exposure'];npos,nneg,tail=pending[:3];e['successful_batches']+=1;e['successful_positive_slots']+=npos;e['successful_negative_slots']+=nneg;e['successful_short_tail_batches']+=tail;e['successful_original15_slots']+=pending[5];e['successful_extra45_slots']+=npos-pending[5]
  return result
 DeviceEdgeSampler.__next__=next_batch;torch.optim.AdamW.zero_grad=clear;torch.optim.AdamW.step=update
 try:yield
 finally:DeviceEdgeSampler.__next__=advance;torch.optim.AdamW.zero_grad=zero;torch.optim.AdamW.step=step
