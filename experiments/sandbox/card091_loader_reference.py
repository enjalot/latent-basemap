"""Released device reference using actual original090 loader in isolated imports."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,hashlib
from pathlib import Path
import numpy as np,torch
import card091_common as C

def main():
 r=C.require_release();s=C.validate_selection(C.read(C.D/'selection.json'));C.verify_sources(s)
 sys.path.insert(0,str(C.R090))
 from basemap.pumap.parametric_umap.core import ParametricUMAP
 assert Path(sys.modules[ParametricUMAP.__module__].__file__).resolve().is_relative_to(C.R090),'original090 import escaped'
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 ids=np.load(C.B/'reference-250000-ids.npy',mmap_mode='r')[:1024].copy()
 # Importing producer would select091 core; raw access is explicit and bounded here instead.
 import mmap
 buf=np.empty((len(ids),1536),'f2')
 for lo,hi,p in [(0,C.LOW,'/data2/monet/pool-20m/dino1536.f16.npy'),(C.LOW,C.HIGH,'/data2/monet/pool-complement-88m/dino1536.f16.npy')]:
  ix=np.flatnonzero((ids>=lo)&(ids<hi))
  if len(ix):
   mm=np.load(p,mmap_mode='r');mm._mmap.madvise(mmap.MADV_RANDOM);buf[ix]=mm[ids[ix]-lo];mm._mmap.madvise(mmap.MADV_DONTNEED)
 x=torch.nn.functional.normalize(torch.tensor(buf.astype('f4'),device='cuda'),dim=1);out={'ids':ids};states={}
 with torch.inference_mode():
  for a in C.ARMS:
   model=ParametricUMAP.load(s['models'][a]['path'],device='cpu').model.cuda().eval().requires_grad_(False)
   h=hashlib.sha256()
   for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.cpu().numpy().tobytes())
   states[a]=h.hexdigest();out[a]=np.concatenate([model(x[lo:lo+256]).cpu().numpy() for lo in range(0,len(x),256)])
   del model
 np.savez(C.D/'loader-reference.npz',**out)
 C.write(C.D/'loader-reference.json',{'PASS':True,'runtime_sha':C.source_check(),'selection_sha':C.sha(C.D/'selection.json'),'release_sha':C.sha(C.O/'card091-release.json'),'baseline_runtime_sha':C.RUNTIME090,'states':states,'output_sha':C.sha(C.D/'loader-reference.npz'),'n':1024})
if __name__=='__main__':main()
