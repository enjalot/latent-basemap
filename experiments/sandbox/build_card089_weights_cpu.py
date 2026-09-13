"""Materialize weights-only original15 tables after independent diagnostic audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time,resource,hashlib
import numpy as np
from card089_weights import weights,ARMS
O=Path('/data/latent-basemap/sandbox/overseer-codex');ROOT=Path(__file__).resolve().parents[2];D=Path('/data/latent-basemap/substrates/card089-soft-reciprocity');DIAG=O/'reciprocity-readiness-20260913'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();audit=json.loads((O/'card089-runner-diagnostic-audit.json').read_text());assert audit['PASS'] and audit['root_result_sha']==sha(DIAG/'result.json')
 assert not D.exists(),'immutable weights directory already exists';D.mkdir()
 a=np.load('/data/latent-basemap/sandbox/dino-arrival-t0/knn_indices.npy',mmap_mode='r');mask=np.load(DIAG/'mutual-mask.npy',mmap_mode='r');n=len(a);maxerr=0.;fp32_different=0;different_rows=0;mass={a:0. for a in ARMS};reciprocal_exposure={a:0. for a in ARMS};favored_exposure={a:0. for a in ARMS}
 out={arm:np.lib.format.open_memmap(D/f'{arm}-weights.npy',mode='w+',dtype='f4',shape=a.shape) for arm in ARMS}
 for lo in range(0,n,8192):
  hi=min(lo+8192,n);m=np.array(mask[lo:hi]);wr,fr=weights(m,'reciprocal');wc,fc=weights(m,'rank_count_control')
  assert np.array_equal(np.sort(wr,axis=1),np.sort(wc,axis=1)) and np.array_equal(wr.sum(1,dtype='f8'),wc.sum(1,dtype='f8')),'row mass/multiset mismatch'
  maxerr=max(maxerr,float(np.max(np.abs(wr.sum(1,dtype='f8')-15))));fp32_different+=int((wr.sum(1)!=wc.sum(1)).sum());different_rows+=int((wr!=wc).any(1).sum())
  for arm,w,f in [('reciprocal',wr,fr),('rank_count_control',wc,fc)]:out[arm][lo:hi]=w;mass[arm]+=w.sum(dtype='f8');reciprocal_exposure[arm]+=w[m].sum(dtype='f8');favored_exposure[arm]+=w[f].sum(dtype='f8')
 for v in out.values():v.flush()
 src=np.repeat(np.arange(n,dtype='i4'),15)
 for arm in ARMS:np.savez(D/f'{arm}-edges.npz',sources=src,targets=a.reshape(-1),weights=out[arm].reshape(-1),n_nodes=np.int64(n))
 inputs={p:h for p,h in json.loads((DIAG/'result.json').read_text())['inputs'].items()};inputs.update({str(DIAG/n):sha(DIAG/n) for n in ['result.json','mutual-mask.npy','mutual-degree.npy']});inputs[str(O/'card089-soft-reciprocity-cpu-scope.md')]=sha(O/'card089-soft-reciprocity-cpu-scope.md')
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert rss<8192
 r={'PASS':True,'status':'CPU_WEIGHTS_ONLY_NOT_GPU_READY','n':n,'k':15,'arms':ARMS,'inputs':inputs,'builder_sha':sha(__file__),'weight_law_sha':sha(ROOT/'experiments/sandbox/card089_weights.py'),'files':{p.name:sha(p) for p in D.iterdir()},'max_row_mass_error_FP64_sum_stored_FP32':maxerr,'row_sums_equal_FP64':True,'sorted_row_multisets_exact':True,'rows_with_order_sensitive_FP32_reduction_difference':fp32_different,'different_weight_rows':different_rows,'total_mass':mass,'expected_reciprocal_positive_fraction':{a:reciprocal_exposure[a]/mass[a] for a in ARMS},'expected_favored_positive_fraction':{a:favored_exposure[a]/mass[a] for a in ARMS},'full_support_no_zero':True,'cpu_wall_s':time.monotonic()-start,'max_rss_MiB':rss,'limits':'Actual stored FP32 values have equal mathematical row mass measured byFP64; FP32 reduction can depend on column order. Actual CUDA CDF and row exposure still require device audit. No labels/targets/endpoints changed; no features/models/queries/GPU.'};(D/'manifest.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
