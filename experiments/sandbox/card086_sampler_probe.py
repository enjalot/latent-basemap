"""Default-off wrapper instrumentation; core and production sampler bytes unchanged."""
from contextlib import contextmanager
import time,hashlib
import numpy as np,torch
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceEdgeSampler

@contextmanager
def probe(record,expected_weight_sha):
 init=DeviceEdgeSampler.__init__;iterate=DeviceEdgeSampler.__iter__;advance=DeviceEdgeSampler.__next__
 def initialize(self,*args,**kwargs):
  start=time.monotonic();init(self,*args,**kwargs);torch.cuda.synchronize()
  assert self.weighted_edge_sampling and not self.uniform_with_replacement and self.positive_target_mode=='binary' and not self._per_batch,'actual weighted route differs'
  w=np.asarray(args[3] if len(args)>3 else kwargs['weights'],dtype='f4');assert hashlib.sha256(w.tobytes()).hexdigest()==expected_weight_sha,'actual sampler weight mismatch'
  cdf=self.sample_cdf;assert cdf.dtype==torch.float64 and bool(torch.isfinite(cdf).all())
  wf=torch.as_tensor(w,dtype=torch.float64,device=cdf.device);mass=wf.sum();interval=torch.diff(cdf,prepend=torch.zeros(1,dtype=cdf.dtype,device=cdf.device));assert bool((interval>=0).all()) and abs(float(cdf[-1])-1)<1e-12
  lost=(wf>0)&(interval==0)
  record['cdf']={'positive_zero_width_count':int(lost.sum()),'positive_zero_width_probability':float(wf[lost].sum()/mass),'zero_weights':int((wf==0).sum()),'tiny_below_2pow24_probability':float(wf[(wf>0)&(wf<2**-24)].sum()/mass),'cdf_terminal':float(cdf[-1]),'weight_values_sha':expected_weight_sha,'weighted_requested':True,'weighted_effective':True,'positive_target_mode':'binary','negative_policy':'global_uniform_nonself','edges':len(w)}
  torch.cuda.synchronize();record['sampler_init_s']=time.monotonic()-start
 def iteration(self):
  torch.cuda.synchronize();start=time.monotonic();out=iterate(self);torch.cuda.synchronize();record.setdefault('epoch_construction_s',[]).append(time.monotonic()-start);return out
 def next_batch(self):
  out=advance(self)
  if 'actual_first_batch' not in record:
   labels=out[2];npos=len(labels)-self.num_neg;assert bool((labels[:npos]==1).all() and (labels[npos:]==0).all());assert npos==self.num_pos
   src=self._last_all_src;dst=self._last_all_dst;assert bool((src[npos:]!=dst[npos:]).all())
   record['actual_first_batch']={'positive_slots':npos,'negative_slots':self.num_neg,'binary_labels':True,'nonself_negatives':True,'positive_source_sha':hashlib.sha256(src[:npos].cpu().numpy().tobytes()).hexdigest(),'positive_target_sha':hashlib.sha256(dst[:npos].cpu().numpy().tobytes()).hexdigest(),'negative_source_sha':hashlib.sha256(src[npos:].cpu().numpy().tobytes()).hexdigest(),'negative_target_sha':hashlib.sha256(dst[npos:].cpu().numpy().tobytes()).hexdigest(),'sampler_rng_sha':hashlib.sha256(self.gen.get_state().cpu().numpy().tobytes()).hexdigest()}
  return out
 DeviceEdgeSampler.__next__=next_batch
 DeviceEdgeSampler.__init__=initialize;DeviceEdgeSampler.__iter__=iteration
 try:yield
 finally:DeviceEdgeSampler.__next__=advance;DeviceEdgeSampler.__init__=init;DeviceEdgeSampler.__iter__=iterate
