"""Separate-process unchanged-path replay using actual historical modules and checkpoints."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,argparse
from pathlib import Path
import numpy as np,torch
import card085_common as C

def main():
 C.require_release()
 ap=argparse.ArgumentParser();ap.add_argument('card',choices=['023','075']);ap.add_argument('runtime',choices=['original','proposed']);ap.add_argument('mode',choices=['fresh','epoch']);ap.add_argument('dest');a=ap.parse_args();dest=Path(a.dest);dest.mkdir(parents=True)
 root=C.R.parent/f'card{a.card}-code' if a.runtime=='original' else C.R
 sys.path.insert(0,str(root));from basemap.pumap.parametric_umap.core import ParametricUMAP
 import basemap.pumap.parametric_umap.core as core
 assert Path(core.__file__).is_relative_to(root)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(42);np.random.seed(42);torch.cuda.manual_seed_all(42)
 hist=C.R.parent/f'card{a.card}-train';sub='actual3d' if a.card=='023' else 'uniform'
 admission=C.read(hist/('admission-actual3d.json' if a.card=='023' else 'uniform/admission.json'));ident=admission['card012_identity'] if a.card=='023' else admission
 parent=C.R.parent/'dino-arrival-t0/champion-bs16k/model.pt' if a.card=='023' else C.CHAMP
 p=ParametricUMAP.load(str(parent),device='cuda');p.model=None;p.n_components=3;p.learning_rate=.001 if a.card=='023' else .0001;p.lr_schedule='constant';p.warmup_steps=0;p.batch_size=16384;p.n_epochs=100000;p.rankneg_window=500000 if a.card=='023' else 0;p.x_residency='auto';p.required_input_pipeline='device'
 if a.card=='075':p.gpu_resident_vram_budget_gb=14.;p._rankneg_scale=None;p.clip_grad_norm=1.
 for n,v in [('anchor_ids_path',''),('anchor_hold_weight',0.),('replay_bank_path',''),('replay_weight',0.),('deriv_bank_path',''),('deriv_weight',0.)]:
  if hasattr(p,n):setattr(p,n,v)
 p._card013_radii=np.load(C.D/'radii.npy').astype('f4');p._card012_identity=ident
 warm_path=C.R.parent/'card015-init/init-card015-3d.pt' if a.card=='023' else hist/'uniform/prepared.pt';warm=torch.load(warm_path,map_location='cpu',weights_only=False)['model_state']
 resume=None;step=0
 if a.mode=='epoch':
  resume=hist/sub/'ckpts/ckpt-epoch1.pt';ck=torch.load(resume,map_location='cpu',weights_only=False);assert not ck['step_checkpoint'];step=ck['global_step'];del ck
 dose=step+8
 if a.runtime=='proposed':
  C.configure(p,{'arm':'fresh' if a.card=='023' else 'finish','dose':dose,'lr':p.learning_rate,'rankneg_window':p.rankneg_window},p._card013_radii,[dose]);p._card012_identity=ident
 p._max_train_steps=dose;p._checkpoint_step_targets={dose}
 X=np.asarray(np.load(C.D/'train.f16.npy',mmap_mode='r'),dtype='f4')
 p.fit(X,precomputed_edges_path=str(C.D/'edges-fixed15.npz'),random_state=42,verbose=False,warm_start_state=warm if resume is None else None,checkpoint_every_epochs=1,checkpoint_dir=str(dest/'ckpts'),resume_from=str(resume) if resume else None)
 ck=torch.load(dest/'ckpts'/f'ckpt-step{dose}.pt',map_location='cpu',weights_only=False)
 assert ck['global_step']==dose and ck['train_stats']['positive_lr_optimizer_steps']==dose
 C.write(dest/'receipt.json',{'PASS':True,'core_path':core.__file__,'core_sha':C.sha(core.__file__),'endpoint':str(dest/'ckpts'/f'ckpt-step{dose}.pt'),'resume_epoch_boundary':a.mode=='epoch','end_epoch':ck['epoch'],'end_step':dose})
if __name__=='__main__':main()
