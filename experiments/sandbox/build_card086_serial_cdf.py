"""CPU-only immutable CDF arrays; existing graph and weights remain unchanged."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,time,resource
import numpy as np,torch,numba
from card086_serial_cdf import build
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex';G=Path('/data/latent-basemap/substrates/card086-directed-membership');D=Path('/data/latent-basemap/substrates/card086-serial-cdf-v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();torch.set_num_threads(2);assert not D.exists(),'immutable CDF directory exists';D.mkdir();m=json.loads((G/'manifest.json').read_text());arms={}
 for arm in ['all_one','membership']:
  p=G/f'{arm}-edges.npz';assert sha(p)==m['files'][p.name]
  with np.load(p) as z:w=z['weights']
  cdf,terminal=build(w);interval=np.diff(cdf,prepend=0.);lost=(w>0)&(interval==0);zero=w==0;mass=np.sum(w,dtype=np.longdouble)
  t=torch.from_numpy(w).double();old=(t.cumsum(0)/float(t.sum())).numpy();max_old=float(np.max(np.abs(cdf-old)));equal_old=bool(np.array_equal(cdf,old));del t,old
  reference=np.cumsum(w,dtype=np.longdouble);reference/=reference[-1];max_ref=0.
  for lo in range(0,len(w),65536):max_ref=max(max_ref,float(np.max(np.abs(cdf[lo:lo+65536].astype(np.longdouble)-reference[lo:lo+65536]))))
  del reference
  probability=float(np.sum(w[lost],dtype=np.longdouble)/mass);zero_interval_mass=float(interval[zero].sum());assert probability<=1e-10 and zero_interval_mass==0 and (interval>=0).all()
  if arm=='all_one':assert equal_old,'all_one CDF mismatch STOP'
  out=D/f'{arm}-cdf.npy';np.save(out,cdf)
  arms[arm]={'cdf_file':out.name,'cdf_file_sha':sha(out),'cdf_values_sha':hashlib.sha256(cdf.tobytes()).hexdigest(),'weight_values_sha':hashlib.sha256(w.tobytes()).hexdigest(),'graph_sha':sha(p),'edges':len(w),'serial_terminal_before_normalization':terminal,'normalization':'divide_by_own_serial_terminal_FP64','cdf_terminal':float(cdf[-1]),'min_interval':float(interval.min()),'decreasing_count':int((interval<0).sum()),'positive_zero_width_count':int(lost.sum()),'positive_zero_width_weight_mass':float(w[lost].sum(dtype='f8')),'positive_zero_width_probability':probability,'zero_weight_count':int(zero.sum()),'zero_weight_interval_mass':zero_interval_mass,'zero_weight_nonzero_interval_count':int((zero&(interval!=0)).sum()),'max_CDF_distortion_vs_old_CPU':max_old,'max_CDF_distortion_vs_extended_precision_reference':max_ref,'old_CPU_CDF_bit_identical':equal_old}
  del w,cdf,interval,lost,zero
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert rss<8192
 result={'PASS':True,'status':'CPU_CDF_READY_DEVICE_PARITY_REQUIRED','algorithm':'serial_FP64_addition_numba_fastmathFalse_parallelFalse_then_own_terminal_division','numpy_version':np.__version__,'numba_version':numba.__version__,'algorithm_source_sha':sha(R/'experiments/sandbox/card086_serial_cdf.py'),'builder_sha':sha(__file__),'old_graph_manifest_sha':sha(G/'manifest.json'),'arms':arms,'CPU_wall_s':time.monotonic()-start,'max_rss_MiB':rss,'limitations':'CPU proof only. Actual device uploaded-array equality,monotonicity/lostmass/distortion,all_one oldCUDA exact CDF and model/output/RNG parity mandatory. No GPU/epoch/model.'};(D/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
