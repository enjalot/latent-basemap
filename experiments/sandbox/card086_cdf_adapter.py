"""Default-off constructor adapter: upload frozen CPU CDF without sampler/core edits."""
from contextlib import contextmanager
import hashlib
import numpy as np,torch
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceEdgeSampler

def values_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
@contextmanager
def fixed_cdf(cdf,expected_cdf_sha,expected_weight_sha,*,require_old_exact=False,record=None,proof_path=None):
 cdf=np.array(cdf,dtype='f8',copy=True);assert values_sha(cdf)==expected_cdf_sha,'CDF values identity mismatch'
 assert cdf.ndim==1 and len(cdf) and np.isfinite(cdf).all() and (np.diff(cdf)>=0).all() and cdf[-1]==1.,'invalid frozen CDF'
 cdf.flags.writeable=False;original=DeviceEdgeSampler.__init__
 def initialize(s,*args,**kwargs):
  weights=np.asarray(args[3] if len(args)>3 else kwargs['weights'],dtype='f4');assert values_sha(weights)==expected_weight_sha,'CDF weight identity mismatch';assert len(weights)==len(cdf),'CDF length mismatch'
  original(s,*args,**kwargs)
  assert s.weighted_edge_sampling and not s.uniform_with_replacement and not s._per_batch and s.positive_target_mode=='binary','CDF adapter unsupported route'
  old=s.sample_cdf;uploaded=torch.from_numpy(cdf.copy()).to(device=s.device);assert values_sha(uploaded.cpu().numpy())==expected_cdf_sha,'CDF upload differs'
  exact=torch.equal(old,uploaded)
  interval=torch.diff(uploaded,prepend=torch.zeros(1,dtype=uploaded.dtype,device=uploaded.device));w=torch.as_tensor(weights,dtype=torch.float64,device=uploaded.device);lost=(w>0)&(interval==0);prob=float(w[lost].sum()/w.sum());zero_mass=float(interval[w==0].sum());old_interval=torch.diff(old,prepend=torch.zeros(1,dtype=old.dtype,device=old.device))
  proof={'cdf_values_sha':expected_cdf_sha,'weight_values_sha':expected_weight_sha,'uploaded_bit_identical':True,'old_CDF_bit_identical':bool(exact),'max_CDF_distortion_vs_actual_old_device':float((uploaded-old).abs().max()),'old_min_interval':float(old_interval.min()),'old_decreasing_count':int((old_interval<0).sum()),'old_terminal_residual':float(old[-1])-1.,'cdf_terminal':float(uploaded[-1]),'min_interval':float(interval.min()),'decreasing_count':int((interval<0).sum()),'positive_zero_width_count':int(lost.sum()),'positive_zero_width_probability':prob,'zero_weight_interval_mass':zero_mass,'zero_weight_nonzero_interval_count':int(((w==0)&(interval!=0)).sum())}
  if record is not None:record.update(proof)
  if proof_path is not None:
   from card086_common import write
   write(proof_path,proof)
  if require_old_exact:assert exact,'all_one old CDF mismatch STOP'
  assert proof['decreasing_count']==0 and proof['cdf_terminal']==1.,'uploaded CDF invariant'
  assert prob<=1e-10 and zero_mass==0 and proof['zero_weight_nonzero_interval_count']==0,'uploaded CDF mass failure'
  if require_old_exact:
   saved=s.gen.get_state().clone()
   try:
    old_ids=s._draw_idx(100000);old_neg=s._sample_negatives(s.num_neg);old_rng=s.gen.get_state().clone();s.gen.set_state(saved);s.sample_cdf=uploaded
    new_ids=s._draw_idx(100000);new_neg=s._sample_negatives(s.num_neg);new_rng=s.gen.get_state().clone()
    assert torch.equal(old_ids,new_ids) and all(torch.equal(a,b) for a,b in zip(old_neg,new_neg)) and torch.equal(old_rng,new_rng),'all_one actual sampler outputs mismatch STOP'
    proof['allone_actual100000_positive_and_negative_ID_RNG_exact']=True
   finally:s.gen.set_state(saved)
  s.sample_cdf=uploaded;s._card086_cdf_proof=proof
  if record is not None:record.update(proof)
  if proof_path is not None:
   from card086_common import write
   write(proof_path,proof)
 DeviceEdgeSampler.__init__=initialize
 try:yield
 finally:DeviceEdgeSampler.__init__=original
