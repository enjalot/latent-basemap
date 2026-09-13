"""Production DeviceEdgeSampler on CPU: actual IDs, labels, law and boundary RNG parity."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,tempfile,json
import numpy as np,torch
import card086_data as D
sys.path.insert(0,str(D.R))
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler

def sampler(w,src,dst,n):
 s=DeviceEdgeSampler(DeviceArrayDataset(np.arange(n,dtype='f4')[:,None],device='cpu'),src,dst,w,n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,uniform_with_replacement=False,device='cpu');s._stash_ids=True;return s

def main():
 torch.set_num_threads(2);m=D.validate_bundle();n=512;src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.array([(i+j)%n for i in range(n) for j in range(1,16)],dtype='i4');w=np.array(np.load(D.D/'membership-weights.npy',mmap_mode='r')[:n].ravel());samplers=[sampler(np.ones(len(w),'f4'),src,dst,n),sampler(w,src,dst,n)];checks={};batch_proof=[]
 for epoch in range(2):
  for s in samplers:iter(s)
  assert torch.equal(samplers[0].gen.get_state(),samplers[1].gen.get_state())
  for batch in range(len(samplers[0])):
   labels=[]
   for s in samplers:labels.append(next(s)[2])
   npos=len(labels[0])-samplers[0].num_neg
   assert npos==min(1638,len(w)-batch*1638)
   assert all(bool((v[:npos]==1).all() and (v[npos:]==0).all()) for v in labels)
   a,b=samplers;assert torch.equal(a._last_all_src[npos:],b._last_all_src[npos:]) and torch.equal(a._last_all_dst[npos:],b._last_all_dst[npos:]);assert bool((a._last_all_src[npos:]!=a._last_all_dst[npos:]).all());assert torch.equal(a.gen.get_state(),b.gen.get_state())
   batch_proof.append({'epoch':epoch,'attempt':batch,'positive_slots':npos,'negative_slots':a.num_neg,'negative_ids_equal':True,'rng_equal':True})
 checks['actual_binary_labels_and_slots']=True;checks['actual_negative_ID_RNG_parity_two_epochs']=True;checks['same_weighted_CDF_route']=all(s.weighted_edge_sampling and not s.uniform_with_replacement and s.sample_cdf.dtype==torch.float64 for s in samplers)
 s=sampler(w,src,dst,n);draw=s._draw_idx(500000).numpy();bins=np.arange(len(w))%32;observed=np.bincount(bins[draw],minlength=32)/len(draw);expected=np.bincount(bins,weights=w.astype('f8'),minlength=32)/w.astype('f8').sum();assert abs(observed-expected).max()<.002;checks['declared_empirical_positive_law']=True
 # Reversed association is a different graph treatment even though weight histogram agrees.
 assert not np.array_equal(w,w[::-1]);checks['weight_pairing_changes_cdf']=not torch.equal(s.sample_cdf,sampler(w[::-1].copy(),src,dst,n).sample_cdf)
 with tempfile.TemporaryDirectory() as td:
  p=Path(td);(p/'graph').write_bytes(b'original');bad=dict(m);bad['files']={'graph':D.sha(p/'graph')};D.write(p/'manifest.json',bad);(p/'graph').write_bytes(b'corrupted')
  try:D.validate_bundle(p)
  except AssertionError as e:assert str(e)=='card086 data artifact hash mismatch';checks['corrupt_graph_rejected']=True
  else:raise AssertionError('corrupt graph accepted')
 assert all(checks.values());out={'CPU_PASS':True,'GPU_READY':False,'checks':checks,'batch_proof':batch_proof,'empirical_bin_max_error':float(abs(observed-expected).max()),'data_manifest_sha':D.sha(D.D/'manifest.json'),'sampler_source_sha':D.sha(D.R/'basemap/pumap/parametric_umap/datasets/edge_list_dataset.py'),'scope':'Real weighted sampler on CPU512nodes at production16384batch, two genuine sampler epochs incl short tail; GPU full2M CDF and optimizer/resume still mandatory.'};D.write(D.O/'card086-consumer-cpu.json',out);print('CPU CONSUMER PASS',len(checks))
if __name__=='__main__':main()
