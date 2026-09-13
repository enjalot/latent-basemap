"""Informative-start pilot: immutable inputs, identity and shared graph configuration."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card073-finishing');TD=R.parent/'card085-train';CHAMP=R.parent/'card023-train/model-actual3d.pt'
ARMS=['fresh','finish'];N=2000000;DIM=1536;DOSES={'fresh':400000,'finish':60000};DOSE=60000;BATCH=16384;SEED=42;LR={'fresh':.001,'finish':.0001};SNAPS={'fresh':[50000,100000,150000,200000,250000,300000,350000,400000],'finish':[20000,40000,60000]};PREP_STEPS=2000;PREP_BATCH=2048;PREP_SEED=65068
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
 m=read(R/'card085-runtime-sha.json');assert all(sha(R/n)==h for n,h in m.items()),'card085 runtime changed';return sha(R/'card085-runtime-sha.json')
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
def identity(arm,dose=None,nodes=N,graph=None,radii_path=None,warm_path=None):
 assert arm in ARMS;dose=DOSES[arm] if dose is None else dose;graph=Path(graph or D/'edges-fixed15.npz');rp=Path(radii_path or D/'radii.npy');wp=Path(warm_path or warm(arm))
 if warm_path is None:
  prepared=read(TD/arm/'preparation.json');assert prepared['READY'] and prepared['prepared_sha']==sha(wp) and prepared['runtime_sha']==source_check(),'prepared-start identity mismatch'
 else:
  prepared={'parent_sha':sha(wp)}
 return {'card':'085','arm':arm,'phase':arm,'stage_full_dose':DOSES[arm],'optimizer_reset':True,'validation_rows':10660,'original_untrained_sha':sha(R.parent/'card015-init/init-card015-3d.pt'),'history_cpu_sha':sha(O/'card085-history-compatibility.json'),'protocol_sha':sha(O/'card085-fresh-uniform-history.md'),'output_dim':3,'initialization_sha':sha(TD/arm/'preparation.json') if warm_path is None else sha(wp),'parent_sha':prepared['parent_sha'],'warm_path':str(wp.resolve()),'test_warm_override':warm_path is not None,'n_nodes':nodes,'dose':dose,'lr':LR[arm],'lr_schedule':'constant','batch_size':BATCH,'pos_ratio':.1,'seed':SEED,'warm_sha':sha(wp),'historical_ranked_parent_sha':sha(CHAMP),'scale_manifest_sha':sha(O/'card023-data-manifest.json'),'input_manifest_sha':sha(D/'inputs-manifest.json'),'graph_sha':sha(graph),'radii_sha':sha(rp),'radii_values_sha':hashlib.sha256(np.ascontiguousarray(np.load(rp),dtype='f4').tobytes()).hexdigest(),'runtime_manifest_sha':source_check(),'source_files':read(R/'card085-runtime-sha.json'),'rankneg_window':0,'negative_policy':'uniform_nonself_rank_scaling_off','precision':'device_fp16','gpu_resident_vram_budget_gb':10. if arm=='fresh' else 14.,'fneg_weight':1.,'neg_tanh_gamma':4.,'positive_target_mode':'binary','replay_weight':0.,'lmc_weight':0.,'weight_decay':.01,'grad_clip':1.,'all_auxiliary_losses_off':True}
def configure(p,ident,radii,checkpoints):
 p.model=None;p.n_components=3;p.hidden_dim=2048;p.learning_rate=ident['lr'];p.lr_schedule='constant';p.warmup_steps=0;p.n_epochs=100000;p._max_train_steps=ident['dose'];p.rankneg_window=ident['rankneg_window'];p._rankneg_scale=None;p.batch_size=BATCH;p.gpu_resident_vram_budget_gb=10. if ident['arm']=='fresh' else 14.;p.x_residency='auto';p.required_input_pipeline='device'
 assert p.architecture=='residual_bottleneck' and not p.use_dropout and p.pos_ratio==.1 and p.fneg_weight==1. and p.neg_tanh_gamma==4. and p.positive_target_mode=='binary'
 for n,v in [('anchor_ids_path',''),('anchor_hold_weight',0.),('replay_bank_path',''),('replay_weight',0.),('deriv_bank_path',''),('deriv_weight',0.),('midnear_enabled',False),('density_weight',0.),('correlation_weight',0.)]:
  if hasattr(p,n):setattr(p,n,v)
 assert not getattr(p,'_card011_schedule',None);p.clip_grad_norm=1.;p._card013_radii=radii;p._card012_identity=ident;p._checkpoint_step_targets=set(checkpoints);p._card054_lmc=None
def validate_ckpt(ck,ident):
 import torch
 import card085_resume as V
 assert_identity(ck['card012_identity'],ident);step=ck['global_step'];assert isinstance(step,int) and 0<step<=ident['dose'];V.validate_resume_payload(ck,step,n_nodes=ident['n_nodes']);assert ck['config']['learning_rate']==ident['lr'] and ck['config']['lr_schedule']=='constant' and ck['config']['replay_weight']==0. and not ck['config']['replay_enabled']
 assert ck['model']['proj_out.weight'].shape==(3,2048) and ck['model']['proj_out.bias'].shape==(3,);original=torch.load(ident['warm_path'],map_location='cpu',weights_only=False)['model_state'];assert set(original)==set(ck['model']) and all(original[k].shape==ck['model'][k].shape and torch.isfinite(ck['model'][k]).all() for k in original);return step


def assert_identity(actual, expected):
 assert set(actual)==set(expected), 'card085 identity mismatch: keys'
 for key in sorted(expected):
  assert actual[key]==expected[key], 'card085 identity mismatch: '+key

def require_release():
 import os
 p=O/'card085-release.json'
 assert p.is_file(), 'NO ROOT GPU RELEASE'
 r=read(p)
 assert r.get('PASS') is True and r.get('card')=='085', 'invalid root release'
 assert r['runtime_sha']==source_check(), 'release runtime mismatch'
 assert os.environ.get('CARD085_RELEASE_SHA')==sha(p), 'missing root controller release binding'
 assert all(sha(p)==h for p,h in r['files'].items()), 'release input mismatch'
 return r
