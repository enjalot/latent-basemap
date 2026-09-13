"""Isolated common CDF law:120M endpoints,30M logical draws,strict zero exclusion.

Base sampler/core bytes unchanged. Both arms reconstruct this adapter on resume.
One uniformF64 draw per positive, shared sampler RNG, exactly15N epoch slots.
"""
from contextlib import contextmanager
import numpy as np,torch
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceEdgeSampler

def inverse(cdf,u):
 assert cdf.ndim==1 and u.dtype==torch.float64
 assert bool(((u>=0)&(u<1)).all()),'CDF uniform outside[0,1)'
 idx=torch.searchsorted(cdf,u,right=True)
 assert bool((idx<len(cdf)).all()),'CDF draw beyond terminal'
 return idx

@contextmanager
def matched_sampler():
 init=DeviceEdgeSampler.__init__;iterate=DeviceEdgeSampler.__iter__;draw=DeviceEdgeSampler._draw_idx;advance=DeviceEdgeSampler.__next__
 def initialize(s,*a,**kw):
  init(s,*a,**kw)
  assert s.weighted_edge_sampling and not s.uniform_with_replacement and not s._per_batch and s.positive_target_mode=='binary','card088 requires weighted epoch binary route'
  assert s.source_n_pos==s.n_nodes*60,'card088 common60 endpoint shape'
  w=np.asarray(a[3] if len(a)>3 else kw['weights']).reshape(s.n_nodes,60)
  s._card088_control=bool(w[0,15]==0)
  for lo in range(0,s.n_nodes,32768):
   part=w[lo:lo+32768];assert np.all(part[:,:15]==(6 if s._card088_control else 3)) and np.all(part[:,15:]==(0 if s._card088_control else 1)), 'card088 exact weight law mismatch'
  s.n_pos=s.n_nodes*15
  s._card088_support_edges=s.source_n_pos
  # Exact total mass with integer weights; force no terminal fallthrough.
  assert float(s.sample_cdf[-1])==1.,'card088 CDF terminal must be exact1'
 def sample(s,m):
  return inverse(s.sample_cdf,torch.rand(m,generator=s.gen,device=s.device,dtype=torch.float64))
 def iteration(s):
  s.pos_idx=0;s.batch_no=0;s.perm=sample(s,s.n_pos);return s
 def next_batch(s):
  if s.perm is None:iter(s)
  if s.pos_idx>=s.n_pos:raise StopIteration
  indices=s.perm[s.pos_idx:min(s.pos_idx+s.num_pos,s.n_pos)]
  s._card088_last_edge_idx=indices
  if s._card088_control:assert bool((indices%60<15).all()),'zero control weight sampled'
  return advance(s)
 DeviceEdgeSampler.__init__=initialize;DeviceEdgeSampler.__iter__=iteration;DeviceEdgeSampler._draw_idx=sample;DeviceEdgeSampler.__next__=next_batch
 try:yield
 finally:DeviceEdgeSampler.__init__=init;DeviceEdgeSampler.__iter__=iterate;DeviceEdgeSampler._draw_idx=draw;DeviceEdgeSampler.__next__=advance
