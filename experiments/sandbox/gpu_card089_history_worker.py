"""Root-gated full2M baseline replay; separate original/proposed import processes."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,argparse,json
import numpy as np,torch
import card089_common as C
from card089_baseline import bound_baseline

def main():
 release=C.require_release();bound_base,bound_epoch,bound_files=bound_baseline(release);ap=argparse.ArgumentParser();ap.add_argument('runtime',choices=['original','proposed']);ap.add_argument('mode',choices=['fresh','epoch']);ap.add_argument('dest');a=ap.parse_args();dest=Path(a.dest);dest.mkdir(parents=True)
 root=C.R.parent/'card086-code' if a.runtime=='original' else C.R;sys.path[:0]=[str(root/'experiments/sandbox'),str(root)]
 from basemap.pumap.parametric_umap.core import ParametricUMAP
 import basemap.pumap.parametric_umap.core as core
 import card086_common as H
 from card086_exposure import observe as old_observe
 assert Path(core.__file__).is_relative_to(root)
 hist=C.R.parent/'card086-train/all_one';ident=C.read(hist/'admission.json');radii=np.load(C.D/'radii.npy').astype('f4');resume=None;step=0
 if a.mode=='epoch':
  resume=bound_epoch;ck=torch.load(resume,map_location='cpu',weights_only=False);assert not ck['step_checkpoint'];step=ck['global_step'];del ck
 dose=step+8;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
 p=ParametricUMAP.load(str(C.CHAMP),device='cuda');config=dict(ident,dose=dose)
 (H.configure if a.runtime=='original' else C.configure)(p,config,radii,[dose]);p._card012_identity=ident
 warm=torch.load(hist/'prepared.pt',map_location='cpu',weights_only=False)['model_state'];X=np.asarray(np.load(C.D/'train.f16.npy',mmap_mode='r'),dtype='f4')
 from contextlib import nullcontext
 # Actual new telemetry, with reciprocal counters as metadata only for all_one. It may add metadata only.
 if a.runtime=='proposed':
  from card089_exposure import observe
  mask=np.load(C.O/'reciprocity-readiness-20260913/mutual-mask.npy',mmap_mode='r')
  if resume:
   # Historical lacks089 counters: telemetry is tested separately from this retained state.
   extra=nullcontext()
  else:
   with np.load('/data/latent-basemap/substrates/card086-directed-membership/all_one-edges.npz') as z:ordered_targets=z['targets'].reshape(-1,15)
   extra=observe(p,'reciprocal',mask,ordered_targets)
 else:extra=nullcontext()
 with old_observe(p),extra:
  p.fit(X,precomputed_edges_path='/data/latent-basemap/substrates/card086-directed-membership/all_one-edges.npz',random_state=42,verbose=False,warm_start_state=warm if resume is None else None,checkpoint_every_epochs=1,checkpoint_dir=str(dest/'ckpts'),resume_from=str(resume) if resume else None)
 end=dest/'ckpts'/f'ckpt-step{dose}.pt';C.write(dest/'receipt.json',{'PASS':True,'runtime':a.runtime,'mode':a.mode,'core_path':str(core.__file__),'core_sha':C.sha(core.__file__),'endpoint':str(end),'epoch_source':str(resume) if resume else None,'epoch_source_sha':C.sha(resume) if resume else None,'end_step':dose,'baseline_files':bound_files,'telemetry_scope':'New telemetry enabled for fresh8; historical retained-epoch replay checks unchanged numerical trainer/configuration.089 telemetry resume proven separately on089 states.'})
if __name__=='__main__':main()
