"""Exact attempted/successful pair-slot accounting in persisted train_stats.

No inference from successful dose or mean batch size. The selected device batch
is counted at optimizer.zero_grad (actual loop entry), and successful exposure
at the real AdamW.step call, which GradScaler omits on overflow. No RNG/draws.
"""
from contextlib import contextmanager
import torch
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceEdgeSampler
SCHEMA='card089-pair-exposure-v1'
FIELDS=('attempted_batches','successful_batches','attempted_positive_slots','successful_positive_slots','attempted_negative_slots','successful_negative_slots','attempted_short_tail_batches','successful_short_tail_batches')+tuple(p+'_'+k+'_slots' for p in ['attempted','successful'] for k in ['reciprocal','nonreciprocal','favored','unfavored','verified_positive'])
def validate(stats):
 e=stats.get('card089_exposure');assert isinstance(e,dict) and e.get('schema')==SCHEMA,'missing exposure state'
 assert all(type(e.get(k)) is int and e[k]>=0 for k in FIELDS),'invalid exposure counters'
 assert e['attempted_batches']==stats['attempted_batches'],'attempted exposure counter mismatch'
 assert e['successful_batches']==stats['optimizer_steps_succeeded']==stats['positive_lr_optimizer_steps'],'successful exposure counter mismatch'
 for prefix in ['attempted','successful']:
  assert e[prefix+'_positive_slots']==e[prefix+'_batches']*e['normal_positive_slots']-e[prefix+'_short_tail_batches']*(e['normal_positive_slots']-e['short_tail_positive_slots']),'positive exposure arithmetic mismatch'
  assert e[prefix+'_verified_positive_slots']==e[prefix+'_positive_slots'],'unverified positive consumption'
  for left,right in [('reciprocal','nonreciprocal'),('favored','unfavored')]:assert e[prefix+'_'+left+'_slots']+e[prefix+'_'+right+'_slots']==e[prefix+'_positive_slots'],'support partition mismatch'
  assert e[prefix+'_negative_slots']==e[prefix+'_batches']*e['negative_slots_per_batch'],'negative exposure arithmetic mismatch'
 for k in ['batches','positive_slots','negative_slots','short_tail_batches','reciprocal_slots','nonreciprocal_slots','favored_slots','unfavored_slots']:
  assert e['successful_'+k]<=e['attempted_'+k],'exposure ordering mismatch'
 assert e['attempted_batches']-e['successful_batches']==sum(stats[k] for k in ['amp_overflow_skips','nonfinite_loss_skips','nonfinite_gradient_skips']),'skip exposure counter mismatch'
 return dict(e,skipped_batches=e['attempted_batches']-e['successful_batches'],skipped_positive_slots=e['attempted_positive_slots']-e['successful_positive_slots'],skipped_negative_slots=e['attempted_negative_slots']-e['successful_negative_slots'],skipped_short_tail_batches=e['attempted_short_tail_batches']-e['successful_short_tail_batches'])
@contextmanager
def observe(p,arm,mutual_mask,ordered_targets):
 advance=DeviceEdgeSampler.__next__;zero=torch.optim.AdamW.zero_grad;step=torch.optim.AdamW.step;pending=None;mask=None;degree=None;targets=None;counted=False
 def ours(opt):
  return opt.param_groups[0]['params'][0] is next(p.model.parameters())
 def next_batch(s):
  nonlocal pending,mask,degree,targets,counted
  had_stash=hasattr(s,'_stash_ids');old_stash=getattr(s,'_stash_ids',False);s._stash_ids=True
  try:out=advance(s);counted=False
  finally:
   if had_stash:s._stash_ids=old_stash
   else:delattr(s,'_stash_ids')
  assert s.weighted_edge_sampling and s.positive_target_mode=='binary'
  if mask is None:
   mask=torch.as_tensor(mutual_mask.copy(),device=s.device,dtype=torch.bool);degree=mask.sum(1);mask=mask.reshape(-1);targets=torch.as_tensor(ordered_targets.copy(),device=s.device,dtype=torch.long)
  assert not s._per_batch and s.perm is not None,'unsupported exposure sampler'
  idx=s.perm[max(0,s.pos_idx-s.num_pos):min(s.pos_idx,s.n_pos)];actual_src=s._last_all_src[:len(idx)];actual_dst=s._last_all_dst[:len(idx)]
  assert torch.equal(actual_src,idx//15) and torch.equal(actual_dst,targets.reshape(-1)[idx]),'actual endpoint consumption mismatch'
  actual_mutual=(targets[actual_dst]==actual_src[:,None]).any(1);assert torch.equal(actual_mutual,mask[idx]),'actual reciprocal mask consumption mismatch'
  nr=int(actual_mutual.sum());nf=nr if arm=='reciprocal' else int((idx%15<degree[idx//15]).sum())
  npos=len(out[-1])-s.num_neg;assert len(idx)==npos;pending=(npos,s.num_neg,int(npos<s.num_pos),s.num_pos,s.n_pos%s.num_pos or s.num_pos,nr,nf);return out
 def clear(opt,*args,**kwargs):
  nonlocal counted
  if ours(opt):
   assert pending is not None,'exposure missing selected batch'
   if counted:
    assert p._train_stats['card089_exposure']['attempted_batches']==p._train_stats['attempted_batches'],'exposure loop-entry mismatch'
    return zero(opt,*args,**kwargs)
   counted=True
   e=p._train_stats.setdefault('card089_exposure',dict(schema=SCHEMA,normal_positive_slots=pending[3],negative_slots_per_batch=pending[1],short_tail_positive_slots=pending[4],**{k:0 for k in FIELDS}))
   npos,nneg,tail=pending[:3];e['attempted_batches']+=1;e['attempted_positive_slots']+=npos;e['attempted_negative_slots']+=nneg;e['attempted_short_tail_batches']+=tail
   for k,v in [('reciprocal',pending[5]),('nonreciprocal',npos-pending[5]),('favored',pending[6]),('unfavored',npos-pending[6]),('verified_positive',npos)]:e['attempted_'+k+'_slots']+=v
   assert e['attempted_batches']==p._train_stats['attempted_batches'],'exposure loop-entry mismatch'
  return zero(opt,*args,**kwargs)
 def update(opt,*args,**kwargs):
  result=step(opt,*args,**kwargs)
  if ours(opt):
   assert opt.param_groups[0]['lr']>0,'exposure zero-LR step forbidden';e=p._train_stats['card089_exposure'];npos,nneg,tail=pending[:3];e['successful_batches']+=1;e['successful_positive_slots']+=npos;e['successful_negative_slots']+=nneg;e['successful_short_tail_batches']+=tail
   for k,v in [('reciprocal',pending[5]),('nonreciprocal',npos-pending[5]),('favored',pending[6]),('unfavored',npos-pending[6]),('verified_positive',npos)]:e['successful_'+k+'_slots']+=v
  return result
 DeviceEdgeSampler.__next__=next_batch;torch.optim.AdamW.zero_grad=clear;torch.optim.AdamW.step=update
 try:yield
 finally:DeviceEdgeSampler.__next__=advance;torch.optim.AdamW.zero_grad=zero;torch.optim.AdamW.step=step

def fractions(e):
 return {p:{k:e[p+'_'+k+'_slots']/e[p+'_positive_slots'] for k in ['reciprocal','nonreciprocal','favored','unfavored']} for p in ['attempted','successful']}
