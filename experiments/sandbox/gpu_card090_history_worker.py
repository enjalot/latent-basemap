import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,argparse
import numpy as np,torch
import card090_common as C

def main():
 rel=C.require_release();ap=argparse.ArgumentParser();ap.add_argument('policy',choices=['ranked','uniform']);ap.add_argument('runtime',choices=['original','proposed']);ap.add_argument('mode',choices=['fresh','epoch']);ap.add_argument('dest');a=ap.parse_args();dest=Path(a.dest);dest.mkdir(parents=True)
 card='073' if a.policy=='ranked' else '075';old=C.R.parent/('card'+card+'-code');hist=C.R.parent/('card'+card+'-train')/('gentle' if a.policy=='ranked' else 'uniform');epoch=hist/'ckpts/ckpt-epoch2.pt'
 needed=[hist/n for n in ['model.pt','prepared.pt','preparation.json','manifest.json','admission.json']]+[epoch]
 assert all(str(p) in rel['files'] and C.sha(p)==rel['files'][str(p)] for p in needed),'historical files not release-bound'
 root=old if a.runtime=='original' else C.R;sys.path[:0]=[str(root),str(old/'experiments/sandbox')]
 from basemap.pumap.parametric_umap.core import ParametricUMAP
 import basemap.pumap.parametric_umap.core as core
 import importlib
 H=importlib.import_module('card'+card+'_common')
 assert Path(core.__file__).is_relative_to(root),'historical core import escaped'
 hm=old/('card'+card+'-runtime-sha.json');assert str(hm) in rel['files'],'historical runtime manifest unbound'
 assert all(rel['files'].get(str(old/n))==h for n,h in C.read(hm).items()),'historical source contract incomplete'
 assert H.source_check()==C.sha(hm),'historical source changed'
 ident=C.read(hist/'admission.json');assert ident['seed']==42 and ident['parent_sha']==C.sha(C.CHAMP) and ident['dose']==60000
 resume=epoch if a.mode=='epoch' else None;step=0
 if resume:
  ck=torch.load(resume,map_location='cpu',weights_only=False);assert not ck['step_checkpoint'];H.validate_ckpt(ck,ident);step=ck['global_step'];del ck
 dose=step+8;radii=np.load(C.D/'radii.npy').astype('f4');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
 p=ParametricUMAP.load(str(C.CHAMP),device='cuda');config=dict(ident,dose=dose)
 (H.configure if a.runtime=='original' else C.configure)(p,config,radii,[dose]);p._card012_identity=ident
 warm=torch.load(hist/'prepared.pt',map_location='cpu',weights_only=False)['model_state'];X=np.asarray(np.load(C.D/'train.f16.npy',mmap_mode='r'),dtype='f4')
 p.fit(X,precomputed_edges_path=str(C.D/'edges-fixed15.npz'),random_state=42,verbose=False,warm_start_state=warm if resume is None else None,checkpoint_every_epochs=1,checkpoint_dir=str(dest/'ckpts'),resume_from=str(resume) if resume else None)
 end=dest/'ckpts'/f'ckpt-step{dose}.pt';C.write(dest/'receipt.json',{'PASS':True,'runtime':a.runtime,'policy':a.policy,'mode':a.mode,'core_path':str(core.__file__),'core_sha':C.sha(core.__file__),'endpoint':str(end),'epoch_source':str(resume) if resume else None,'end_step':dose,'baseline_files':{str(p):rel['files'][str(p)] for p in needed}})
if __name__=='__main__':main()
