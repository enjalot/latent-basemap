"""Informative-start pilot: immutable inputs, identity and shared graph configuration."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card073-finishing');TD=R.parent/'card079-train';CHAMP=R.parent/'card023-train/model-actual3d.pt'
ARMS=['uniform'];N=2000000;DIM=1536;DOSE=60000;BATCH=16384;SEED=42;LR={'uniform':.0001};SNAPS=[20000,40000,60000];PREP_STEPS=2000;PREP_BATCH=2048;PREP_SEED=65068
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def state_sha(sd):
 h=hashlib.sha256()
 for k in sorted(sd):h.update(k.encode());h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
 return h.hexdigest()
def source_check():
 m=read(R/'card079-runtime-sha.json');assert all(sha(R/n)==h for n,h in m.items()),'card079 runtime changed';return sha(R/'card079-runtime-sha.json')
def input_check():
 m=read(D/'inputs-manifest.json');assert m['status']=='MATCHED_FINISH_READY_NO_DATA_CHANGED' and m['n']==N and m['dim']==DIM
 assert all(sha(D/n)==h for n,h in m['files'].items());return m
def loaded_modules():
 out={}
 for k,m in list(sys.modules.items()):
  if k.startswith('basemap') and getattr(m,'__file__',None):
   p=Path(m.__file__).resolve();assert p.is_relative_to(R),'basemap import escaped';out[k]={'path':str(p),'sha':sha(p)}
 return out
def warm(arm):return TD/arm/'prepared.pt'
def identity(arm,dose=DOSE,nodes=N,graph=None,radii_path=None,warm_path=None):
 assert arm in ARMS;graph=Path(graph or D/'edges-fixed15.npz');rp=Path(radii_path or D/'radii.npy');wp=Path(warm_path or warm(arm))
 if warm_path is None:
  prepared=read(TD/arm/'preparation.json');assert prepared['READY'] and prepared['prepared_sha']==sha(wp) and prepared['runtime_sha']==source_check(),'prepared-start identity mismatch'
 return {'card':'079','arm':arm,'phase':'graph','output_dim':3,'initialization_sha':sha(TD/arm/'preparation.json'),'parent_sha':read(TD/arm/'preparation.json')['parent_sha'],'n_nodes':nodes,'dose':dose,'lr':LR[arm],'lr_schedule':'constant','batch_size':BATCH,'pos_ratio':.1,'seed':SEED,'warm_sha':sha(wp),'original_init_sha':sha(CHAMP),'scale_manifest_sha':sha(O/'card023-data-manifest.json'),'input_manifest_sha':sha(D/'inputs-manifest.json'),'graph_sha':sha(graph),'radii_sha':sha(rp),'radii_values_sha':hashlib.sha256(np.ascontiguousarray(np.load(rp),dtype='f4').tobytes()).hexdigest(),'runtime_manifest_sha':source_check(),'rankneg_window':0,'negative_policy':'uniform_nonself_explicit_base_weight','base_negative_weight':0.5**0.75,'precision':'device_fp16','gpu_resident_vram_budget_gb':14.,'fneg_weight':1.,'neg_tanh_gamma':4.,'positive_target_mode':'binary','replay_weight':0.,'lmc_weight':0.,'weight_decay':.01,'grad_clip':1.,'all_auxiliary_losses_off':True}
def configure(p,ident,radii,checkpoints):
 p.model=None;p.n_components=3;p.hidden_dim=2048;p.learning_rate=ident['lr'];p.lr_schedule='constant';p.warmup_steps=0;p.n_epochs=100000;p._max_train_steps=ident['dose'];p.rankneg_window=ident['rankneg_window'];p._rankneg_scale=None;p._card079_base_negative_weight=ident['base_negative_weight'];p.batch_size=BATCH;p.gpu_resident_vram_budget_gb=14.;p.x_residency='auto';p.required_input_pipeline='device'
 assert p.architecture=='residual_bottleneck' and not p.use_dropout and p.pos_ratio==.1 and p.fneg_weight==1. and p.neg_tanh_gamma==4. and p.positive_target_mode=='binary'
 for n,v in [('anchor_ids_path',''),('anchor_hold_weight',0.),('replay_bank_path',''),('replay_weight',0.),('deriv_bank_path',''),('deriv_weight',0.),('midnear_enabled',False),('density_weight',0.),('correlation_weight',0.)]:
  if hasattr(p,n):setattr(p,n,v)
 assert not getattr(p,'_card011_schedule',None);p.clip_grad_norm=1.;p._card013_radii=radii;p._card012_identity=ident;p._checkpoint_step_targets=set(checkpoints);p._card054_lmc=None
def validate_ckpt(ck,ident):
 import torch
 import card079_resume as V
 assert ck['card012_identity']==ident,'card079 admission-identity mismatch';step=ck['global_step'];assert isinstance(step,int) and 0<step<=ident['dose'];V.validate_resume_payload(ck,step,n_nodes=ident['n_nodes']);assert ck['config']['learning_rate']==ident['lr'] and ck['config']['lr_schedule']=='constant' and ck['config']['replay_weight']==0. and not ck['config']['replay_enabled']
 assert ck['model']['proj_out.weight'].shape==(3,2048) and ck['model']['proj_out.bias'].shape==(3,);original=torch.load(warm(ident['arm']),map_location='cpu',weights_only=False)['model_state'];assert set(original)==set(ck['model']) and all(original[k].shape==ck['model'][k].shape and torch.isfinite(ck['model'][k]).all() for k in original);return step
