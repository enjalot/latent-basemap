"""Card049 constants, source identity and the existing core fit contract."""
from pathlib import Path
import json,hashlib,os
import numpy as np
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card049-clip-scale');TD=R.parent/'card049-train';CHAMP=R.parent/'dino-arrival-t0/champion-bs16k/model.pt'
CHAMP_SHA='d62b18cb0df94981f39fca11e4e55b42debef73ec5c3f7fff169717cf0690ad7'
ARMS=['ordinary3d','half3d','shuffled_half3d'];RAD={'ordinary3d':None,'half3d':'r-half.npy','shuffled_half3d':'r-shuffled-half.npy'};N=300000;DIM=512;NC=3;DOSE=60000;BATCH=16384;SEED=42;LR=.001;SNAPS=[20000,40000,60000]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def state_sha(sd):
 h=hashlib.sha256()
 for k in sorted(sd):h.update(k.encode());h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
 return h.hexdigest()
def source_check():
 m=read(R/'card049-runtime-sha.json');assert all(sha(R/n)==h for n,h in m.items()),'runtime source changed'
 return sha(R/'card049-runtime-sha.json')
def input_check():
 m=read(D/'inputs-manifest.json');assert m['PASS'] and m['n_train']==N and m['dim']==DIM
 for n,h in m['files'].items():assert sha(D/n)==h,n
 return m
def graph_check():
 m=read(D/'graph-manifest.json');assert m['PASS'] and m['input_manifest_sha']==sha(D/'inputs-manifest.json') and m['builder_sha']==sha(R/'experiments/sandbox/gpu_card049_graph.py') and m['runtime_sha']==source_check()
 for n,h in m['files'].items():assert sha(D/n)==h,n
 return m
def identity(arm,dose=DOSE,nodes=N,graph=None,radii_path=None):
 assert arm in ARMS;assert sha(CHAMP)==CHAMP_SHA,'parent model changed';graph=Path(graph) if graph else D/'edges-fixed15.npz';radii_path=Path(radii_path) if radii_path else (D/RAD[arm] if RAD[arm] else None)
 return {'card':'card049','arm':arm,'n_nodes':nodes,'dose':dose,'input_dim':DIM,'n_components':NC,'hidden_dim':2048,'architecture':'residual_bottleneck','seed':SEED,'lr':LR,'lr_schedule':'constant','batch_size':BATCH,'positive_fraction':.1,'rankneg_window':75000 if nodes==N else min(75000,nodes-1),'gpu_resident_vram_budget_gb':14.,'fneg_weight':1.,'neg_tanh_gamma':4.,'positive_target_mode':'binary','input_manifest_sha':sha(D/'inputs-manifest.json'),'init_sha':sha(D/'init.pt'),'champion_sha':CHAMP_SHA,'graph_sha':sha(graph),'graph_manifest_sha':sha(D/'graph-manifest.json'),'radii_sha':sha(radii_path) if radii_path else None,'radii_values_sha':hashlib.sha256(np.ascontiguousarray(np.load(radii_path),dtype='f4').tobytes()).hexdigest() if radii_path else None,'runtime_manifest_sha':source_check(),'weight_decay':.01,'grad_clip':1.,'precision':'device_fp16'}
def configure(p,ident,radii,checkpoints=()):
 p.model=None;p.n_components=3;p.hidden_dim=2048;p.learning_rate=.001;p.lr_schedule='constant';p.warmup_steps=0;p.n_epochs=100000;p._max_train_steps=ident['dose'];p.rankneg_window=ident['rankneg_window'];p.batch_size=BATCH;p.gpu_resident_vram_budget_gb=14.;p.x_residency='auto';p.required_input_pipeline='device'
 assert p.architecture=='residual_bottleneck' and not p.use_dropout and p.pos_ratio==.1 and p.fneg_weight==1. and p.neg_tanh_gamma==4. and p.positive_target_mode=='binary'
 for n,v in [('anchor_ids_path',''),('anchor_hold_weight',0.),('replay_bank_path',''),('replay_weight',0.),('deriv_bank_path',''),('deriv_weight',0.),('midnear_enabled',False),('density_weight',0.),('correlation_weight',0.)]:
  if hasattr(p,n):setattr(p,n,v)
 assert not getattr(p,'_card011_schedule',None),'unexpected collision schedule'
 p.clip_grad_norm=1.;p._card013_radii=radii;p._card012_identity=ident;p._checkpoint_step_targets=set(checkpoints)
def validate_ckpt(ck,ident):
 import torch
 import card033_validate as V
 assert ck['card012_identity']==ident,'card049 admission-identity mismatch'
 step=ck['global_step'];assert isinstance(step,int) and 0<step<=ident['dose'];V.validate_resume_payload(ck,step,n_nodes=ident['n_nodes'])
 assert ck['config']['learning_rate']==.001 and ck['config']['lr_schedule']=='constant'
 init=torch.load(D/'init.pt',map_location='cpu',weights_only=False)['model_state'];assert set(init)==set(ck['model']) and all(init[k].shape==ck['model'][k].shape and torch.isfinite(ck['model'][k]).all() for k in init)
 return step

def loaded_modules():
 import sys
 out={}
 for name,m in list(sys.modules.items()):
  if name.startswith('basemap') and getattr(m,'__file__',None):
   p=Path(m.__file__).resolve();assert p.is_relative_to(R),'basemap import outside049: '+str(p);out[name]={'path':str(p),'sha':sha(p)}
 return out
