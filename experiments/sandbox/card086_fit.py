"""Same core fit entry point for052 production, device canary and cost preflight."""
from pathlib import Path
import sys,time,gc,hashlib
import numpy as np,torch
import card086_common as C
sys.path.insert(0,str(C.R))
from basemap.pumap.parametric_umap.core import ParametricUMAP

def fit(arm,dose,dest,*,X=None,graph=None,radii=None,radius_path=None,checkpoints=None,resume=None,ident_override=None,warm_path=None):
 C.require_release();dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);start=time.monotonic();assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 graph=Path(graph) if graph else C.GD/f'{arm}-edges.npz'
 if X is None:X=np.asarray(np.load(C.D/'train.f16.npy',mmap_mode='r'),dtype='f4')
 if radius_path is None:radius_path=C.D/'radii.npy'
 if radii is None:radii=np.load(radius_path).astype('f4')
 assert len(radii)==len(X) and np.isfinite(radii).all() and (radii>0).all()
 ident=C.identity(arm,dose,len(X),graph,radius_path,warm_path)
 assert hashlib.sha256(np.ascontiguousarray(radii,dtype='f4').tobytes()).hexdigest()==ident['radii_values_sha']
 ident.update(ident_override or {})
 torch.set_num_threads(2);torch.manual_seed(C.SEED);torch.cuda.manual_seed_all(C.SEED);np.random.seed(C.SEED)
 p=ParametricUMAP.load(str(C.CHAMP),device='cuda');C.configure(p,ident,radii,checkpoints or [dose]);prepared=torch.load(warm_path or C.warm(arm),map_location='cpu',weights_only=False);assert prepared['READY'];warm=prepared['model_state']
 assert C.state_sha(warm)==prepared['prepared_state_sha'];expected_warm=hashlib.sha256(b''.join(np.ascontiguousarray(t.numpy()).tobytes() for t in warm.values())).hexdigest()[:16]
 if resume is not None:C.validate_ckpt(torch.load(resume,map_location='cpu',weights_only=False),ident)
 adm=dest/'admission.json'
 if adm.exists():assert C.read(adm)==ident,'saved admission mismatch'
 else:C.write(adm,ident)
 p.fit(X,precomputed_edges_path=str(graph),random_state=C.SEED,verbose=False,warm_start_state=None if resume else warm,snapshot_steps=tuple(checkpoints or [dose]),snapshot_dir=str(dest),checkpoint_every_epochs=1,checkpoint_dir=str(dest/'ckpts'),resume_from=str(resume) if resume else None)
 if resume is None:assert p.warm_start_sha256==expected_warm,'actual warm-start differs'
 ts=dict(p._train_stats);pi=dict(p._pipeline_info);assert ts['executed_iters']==ts['positive_lr_optimizer_steps']==dose;assert ts['lr_used_min']==ts['lr_used_max']==ident['lr'] and pi['x_residency']=='device_fp16'
 end=dest/'ckpts'/f'ckpt-step{dose}.pt';ck=torch.load(end,map_location='cpu',weights_only=False);C.validate_ckpt(ck,ident);assert C.state_sha(p.model.state_dict())==C.state_sha(ck['model'])
 assert pi['weighted_requested'] is True and pi['weighted_effective'] is True and pi['positive_sampling']=='weighted_with_replacement' and p.weighted_edge_sampling;assert p.rankneg_window==0 and p._rankneg_scale is None, 'actual negative policy differs'
 free,total=torch.cuda.mem_get_info();assert (total-free)/2**30<30
 report={'arm':arm,'dose':dose,'wall_s':time.monotonic()-start,'train_stats':ts,'pipeline_info':pi,'identity':ident,'state_sha':C.state_sha(ck['model']),'global_vram_GiB':(total-free)/2**30,'endpoint_checkpoint':str(end),'loaded_modules':C.loaded_modules(),'actual_warm_sha':getattr(p,'warm_start_sha256',None),'resumed_from':str(resume) if resume else None,'expected_fresh_warm_sha':expected_warm,'lmc_stats':None,'resume_start_step':int(torch.load(resume,map_location='cpu',weights_only=False)['global_step']) if resume else 0}
 return p,ck,report

def validate_arm(arm):
 dest=C.TD/arm;m=C.read(dest/'manifest.json');ident=C.identity(arm);assert m['identity']==ident and m['dose']==C.DOSE and m['model_sha']==C.sha(dest/'model.pt')
 model=torch.load(dest/'model.pt',map_location='cpu',weights_only=False);assert model['rankneg_window']==0 and model['fneg_weight']==1. and model['neg_tanh_gamma']==4.
 assert model['input_dim']==1536 and model['n_components']==3 and model['hidden_dim']==2048
 assert C.state_sha(model['model_state_dict'])==m['state_sha'];assert m['train_stats']['positive_lr_optimizer_steps']==C.DOSE and m['pipeline_info']['x_residency']=='device_fp16'
 C.source_check();assert m['loaded_modules'] and all(C.sha(v['path'])==v['sha'] and Path(v['path']).is_relative_to(C.R) for v in m['loaded_modules'].values())
 ts=m['train_stats'];assert ts['executed_iters']==ts['optimizer_steps_succeeded']==ts['positive_lr_optimizer_steps']==C.DOSE and ts['lr_used_min']==ts['lr_used_max']==ident['lr']
 assert m['lmc_stats'] is None;assert m['pipeline_info']['weighted_requested'] is True and m['pipeline_info']['weighted_effective'] is True and m['pipeline_info']['positive_sampling']=='weighted_with_replacement'
 checks=0;epochs=0;states=[]
 for path in sorted((dest/'ckpts').glob('*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=False);C.validate_ckpt(ck,ident);assert path.stat().st_mtime>=(dest/'admission.json').stat().st_mtime;checks+=1;epochs+=int(path.name.startswith('ckpt-epoch') and not ck['step_checkpoint'])
 assert epochs>=1,'no genuine epoch checkpoint'
 assert checks>=len(C.SNAPS)+1,'missing real epoch checkpoint'
 for step in C.SNAPS:
  snap=torch.load(dest/f'model-step{step}.pt',map_location='cpu',weights_only=False);ck=torch.load(dest/'ckpts'/f'ckpt-step{step}.pt',map_location='cpu',weights_only=False);assert ck['global_step']==step and ck['step_checkpoint'] is True;assert C.state_sha(snap['model_state_dict'])==C.state_sha(ck['model']);assert snap['n_components']==3 and snap['input_dim']==1536
  states.append(C.state_sha(ck['model']))
  if step==C.DOSE:assert C.state_sha(snap['model_state_dict'])==m['state_sha']
 assert len(set(states))==len(C.SNAPS),'identical checkpoint states'
 return {'PASS':True,'arm':arm,'model_sha':m['model_sha'],'state_sha':m['state_sha'],'identity':ident,'checked_checkpoints':checks,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.GD/'manifest.json')}
