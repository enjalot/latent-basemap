"""Informative-start pilot: immutable inputs, identity and shared graph configuration."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card073-finishing');TD=R.parent/'card088-train';CHAMP=R.parent/'card023-train/model-actual3d.pt'
GD=Path('/data/latent-basemap/substrates/card088-positive-support');ARMS=['original15','mixture'];N=2000000;DIM=1536;DOSE=60000;BATCH=16384;SEED=42;LR={a:.0001 for a in ARMS};SNAPS=[20000,40000,60000];PREP_STEPS=2000;PREP_BATCH=2048;PREP_SEED=65068
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
 m=read(R/'card088-runtime-sha.json');assert all(sha(R/n)==h for n,h in m.items()),'card088 runtime changed';return sha(R/'card088-runtime-sha.json')
def input_check(require_graph=True):
 m=read(D/'inputs-manifest.json');assert m['status']=='MATCHED_FINISH_READY_NO_DATA_CHANGED' and m['n']==N and m['dim']==DIM
 assert all(sha(D/n)==h for n,h in m['files'].items())
 if require_graph:
  from card088_graph import validate_bundle
  validate_bundle(GD)
 return m
def loaded_modules():
 out={}
 for k,m in list(sys.modules.items()):
  if k.startswith('basemap') and getattr(m,'__file__',None):
   p=Path(m.__file__).resolve();assert p.is_relative_to(R),'basemap import escaped';out[k]={'path':str(p),'sha':sha(p)}
 return out
def warm(arm):return TD/arm/'prepared.pt'
def identity(arm,dose=DOSE,nodes=N,graph=None,radii_path=None,warm_path=None):
 assert arm in ARMS;graph=Path(graph or GD/f'{arm}-edges.npz');rp=Path(radii_path or D/'radii.npy');wp=Path(warm_path or warm(arm))
 if warm_path is None:
  prepared=read(TD/arm/'preparation.json');assert prepared['READY'] and prepared['prepared_sha']==sha(wp) and prepared['runtime_sha']==source_check(),'prepared-start identity mismatch'
 return {'card':'088','arm':arm,'phase':'graph','exposure_schema':'card088-pair-exposure-v1','weighted_edge_sampling':True,'uniform_with_replacement':False,'data_manifest_sha':sha(GD/'manifest.json'),'weight_values_sha':weight_sha(graph),'endpoints_sha':endpoints_sha(graph),'protocol_sha':sha(O/'card088-positive-support.md'),'quality_prereg_sha':sha(O/'card088-quality-prereg.md'),'reference_sha':read(GD/'manifest.json')['graph_builder_sha'],'positive_support_law':{'original15':[6,0],'mixture':[3,1]}[arm],'endpoint_columns':60,'logical_epoch_draws':nodes*15,'cdf_side':'right','positive_rng_policy':'stateless_epoch_seed88130_plus_completed_attempt_epochs','negative_rng_policy':'original_sampler_seed42_no_positive_draws','original15_preserved':True,'output_dim':3,'initialization_sha':sha(TD/arm/'preparation.json'),'parent_sha':read(TD/arm/'preparation.json')['parent_sha'],'n_nodes':nodes,'dose':dose,'lr':LR[arm],'lr_schedule':'constant','batch_size':BATCH,'pos_ratio':.1,'seed':SEED,'warm_sha':sha(wp),'original_init_sha':sha(CHAMP),'scale_manifest_sha':sha(O/'card023-data-manifest.json'),'input_manifest_sha':sha(D/'inputs-manifest.json'),'graph_sha':sha(graph),'radii_sha':sha(rp),'radii_values_sha':hashlib.sha256(np.ascontiguousarray(np.load(rp),dtype='f4').tobytes()).hexdigest(),'runtime_manifest_sha':source_check(),'rankneg_window':0,'negative_policy':'uniform_nonself_rank_scaling_off','precision':'device_fp16','gpu_resident_vram_budget_gb':14.,'fneg_weight':1.,'neg_tanh_gamma':4.,'positive_target_mode':'binary','replay_weight':0.,'lmc_weight':0.,'weight_decay':.01,'grad_clip':1.,'all_auxiliary_losses_off':True}
def configure(p,ident,radii,checkpoints):
 p.weighted_edge_sampling=True;p.model=None;p.n_components=3;p.hidden_dim=2048;p.learning_rate=ident['lr'];p.lr_schedule='constant';p.warmup_steps=0;p.n_epochs=100000;p._max_train_steps=ident['dose'];p.rankneg_window=ident['rankneg_window'];p._rankneg_scale=None;p.batch_size=BATCH;p.gpu_resident_vram_budget_gb=14.;p.x_residency='auto';p.required_input_pipeline='device'
 assert p.architecture=='residual_bottleneck' and not p.use_dropout and p.pos_ratio==.1 and p.fneg_weight==1. and p.neg_tanh_gamma==4. and p.positive_target_mode=='binary'
 for n,v in [('anchor_ids_path',''),('anchor_hold_weight',0.),('replay_bank_path',''),('replay_weight',0.),('deriv_bank_path',''),('deriv_weight',0.),('midnear_enabled',False),('density_weight',0.),('correlation_weight',0.)]:
  if hasattr(p,n):setattr(p,n,v)
 assert not getattr(p,'_card011_schedule',None);p.clip_grad_norm=1.;p._card013_radii=radii;p._card012_identity=ident;p._checkpoint_step_targets=set(checkpoints);p._card054_lmc=None
def validate_ckpt(ck,ident):
 import torch
 import card088_resume as V
 from card088_exposure import validate as validate_exposure
 assert_identity(ck['card012_identity'],ident);validate_exposure(ck['train_stats']);step=ck['global_step'];assert isinstance(step,int) and 0<step<=ident['dose'];V.validate_resume_payload(ck,step,n_nodes=ident['n_nodes']);assert ck['config']['learning_rate']==ident['lr'] and ck['config']['lr_schedule']=='constant' and ck['config']['replay_weight']==0. and not ck['config']['replay_enabled']
 assert ck['model']['proj_out.weight'].shape==(3,2048) and ck['model']['proj_out.bias'].shape==(3,);original=torch.load(warm(ident['arm']),map_location='cpu',weights_only=False)['model_state'];assert set(original)==set(ck['model']) and all(original[k].shape==ck['model'][k].shape and torch.isfinite(ck['model'][k]).all() for k in original);return step


def weight_sha(graph):
 with np.load(graph) as z:return hashlib.sha256(z['weights'].tobytes()).hexdigest()
def endpoints_sha(graph):
 with np.load(graph) as z:return {k:hashlib.sha256(z[k].tobytes()).hexdigest() for k in ['sources','targets']}
def assert_identity(actual,expected):
 assert set(actual)==set(expected),'card088 identity mismatch: keys'
 for k in sorted(expected):assert actual[k]==expected[k],'card088 identity mismatch: '+k

def require_release():
 import os
 p=O/'card088-release.json'
 assert p.is_file(), 'NO ROOT GPU RELEASE'
 r=read(p)
 assert r.get('PASS') is True and r.get('card')=='088', 'invalid root release'
 assert r['runtime_sha']==source_check(), 'release runtime mismatch'
 assert os.environ.get('CARD088_RELEASE_SHA')==sha(p), 'missing root controller release binding'
 assert all(sha(p)==h for p,h in r['files'].items()), 'release input mismatch'
 return r
