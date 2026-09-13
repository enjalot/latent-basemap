"""Actual base sampler restoration control, callable on CPU or released CUDA."""
import torch
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler
from card088_sampler import matched_sampler

def restoration_control(X,src,dst,weights,n,device):
 names=['__init__','__iter__','__next__','_draw_idx'];methods={k:getattr(DeviceEdgeSampler,k) for k in names}
 def make(weighted):
  s=DeviceEdgeSampler(DeviceArrayDataset(X,device=device),src,dst,weights,n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=weighted,device=device);s._stash_ids=True;return s
 def capture(weighted):
  s=make(weighted);iter(s);out=[]
  for _ in range(2):
   batch=next(s);out.append([v.clone() for v in batch]+[s._last_all_src.clone(),s._last_all_dst.clone(),s.gen.get_state().clone()])
  return out
 before={weighted:capture(weighted) for weighted in [False,True]}
 with matched_sampler('mixture'):
  s=make(True);iter(s);next(s)
 assert all(getattr(DeviceEdgeSampler,k) is v for k,v in methods.items()),'offpath method restoration failed'
 for weighted in [False,True]:
  after=capture(weighted)
  assert all(torch.equal(a,b) for left,right in zip(before[weighted],after) for a,b in zip(left,right)),'offpath IDs values labels RNG changed'
 return {'PASS':True,'device':device,'routes':['base_unweighted_permutation','base_weighted_replacement'],'batches_per_route':2,'checks':'actual IDs/values/labels/negative RNG and method objects before/after adapter'}
