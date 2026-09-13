"""Card084 isolated production configuration and immutable continuation contracts."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card073-finishing');TD=R.parent/'card084-train';CHAMP=R.parent/'card023-train/model-actual3d.pt'
ARMS=['quadratic','pseudo_huber'];N=2000000;DIM=1536;DOSE=60000;BATCH=16384;SEED=42;LR={a:.0001 for a in ARMS};SNAPS=[20000,40000,60000];PREP_STEPS=2000;PREP_BATCH=2048;PREP_SEED=65068
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
 m=read(R/'card084-runtime-sha.json');assert all(sha(R/n)==h for n,h in m.items()),'card084 runtime changed';return sha(R/'card084-runtime-sha.json')
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
def identity(arm,dose=DOSE,nodes=N,graph=None,radii_path=None,warm_path=None,noise_q_path=None,calibration=False):
 assert arm in ARMS;graph=Path(graph or D/'edges-fixed15.npz');rp=Path(radii_path or D/'radii.npy');wp=Path(warm_path or warm(arm))
 if warm_path is None:
  prepared=read(TD/arm/'preparation.json');assert prepared['READY'] and prepared['prepared_sha']==sha(wp) and prepared['runtime_sha']==source_check(),'prepared-start identity mismatch'
 attraction = attraction_identity(arm, calibration=calibration)
 return {**attraction,'card':'084','arm':arm,'phase':'graph','output_dim':3,'initialization_sha':sha(TD/arm/'preparation.json'),'parent_sha':read(TD/arm/'preparation.json')['parent_sha'],'n_nodes':nodes,'dose':dose,'lr':LR[arm],'lr_schedule':'constant','batch_size':BATCH,'pos_ratio':.1,'seed':SEED,'warm_sha':sha(wp),'original_init_sha':sha(CHAMP),'scale_manifest_sha':sha(O/'card023-data-manifest.json'),'input_manifest_sha':sha(D/'inputs-manifest.json'),'graph_sha':sha(graph),'radii_sha':sha(rp),'radii_values_sha':hashlib.sha256(np.ascontiguousarray(np.load(rp),dtype='f4').tobytes()).hexdigest(),'runtime_manifest_sha':source_check(),'rankneg_window':0,'negative_policy':'degree_conditional_cdf','noise_q_path':str(noise_q_path or O/'degree-noise-readiness'/'uniform.npy'),'noise_q_sha':sha(noise_q_path or O/'degree-noise-readiness'/'uniform.npy'),'base_negative_weight':1.0,'precision':'device_fp16','gpu_resident_vram_budget_gb':14.,'fneg_weight':1.,'neg_tanh_gamma':4.,'fneg_lo':.1,'fneg_hi':.4,'kernel_a':read(O/'card083-kernel.json')['a'],'kernel_b':read(O/'card083-kernel.json')['b'],'positive_target_mode':'binary','replay_weight':0.,'lmc_weight':0.,'weight_decay':.01,'grad_clip':1.,'all_auxiliary_losses_off':False}
def configure(p,ident,radii,checkpoints):
 validate_identity(ident)
 p._card084_attraction={k:ident[k] for k in ('family','coefficient','delta')}
 p.fneg_weight=ident['fneg_weight'];p.neg_tanh_gamma=ident['neg_tanh_gamma'];p.fneg_lo=ident['fneg_lo'];p.fneg_hi=ident['fneg_hi'];p.a=ident['kernel_a'];p.b=ident['kernel_b'];p.model=None;p.n_components=3;p.hidden_dim=2048;p.learning_rate=ident['lr'];p.lr_schedule='constant';p.warmup_steps=0;p.n_epochs=100000;p._max_train_steps=ident['dose'];p.rankneg_window=ident['rankneg_window'];p._rankneg_scale=None;p._card079_base_negative_weight=ident['base_negative_weight'];p._card081_q_path=ident['noise_q_path'];p.batch_size=BATCH;p.gpu_resident_vram_budget_gb=14.;p.x_residency='auto';p.required_input_pipeline='device'
 assert p.architecture=='residual_bottleneck' and not p.use_dropout and p.pos_ratio==.1 and p.fneg_weight==ident['fneg_weight'] and p.neg_tanh_gamma==ident['neg_tanh_gamma'] and p.positive_target_mode=='binary'
 for n,v in [('anchor_ids_path',''),('anchor_hold_weight',0.),('replay_bank_path',''),('replay_weight',0.),('deriv_bank_path',''),('deriv_weight',0.),('midnear_enabled',False),('density_weight',0.),('correlation_weight',0.)]:
  if hasattr(p,n):setattr(p,n,v)
 assert not getattr(p,'_card011_schedule',None);p.clip_grad_norm=1.;p._card013_radii=radii;p._card012_identity=ident;p._checkpoint_step_targets=set(checkpoints);p._card054_lmc=None
def validate_ckpt(ck,ident):
 import torch
 import card084_resume as V
 assert ck['card012_identity']==ident,'card084 admission-identity mismatch'
 validate_identity(ident)
 step=ck['global_step'];assert isinstance(step,int) and 0<step<=ident['dose'];V.validate_resume_payload(ck,step,n_nodes=ident['n_nodes']);assert ck['config']['learning_rate']==ident['lr'] and ck['config']['lr_schedule']=='constant' and ck['config']['replay_weight']==0. and not ck['config']['replay_enabled']
 assert isinstance(ck.get('card081_noise_pairs'),int) and ck['card081_noise_pairs']>=0
 assert ck['model']['proj_out.weight'].shape==(3,2048) and ck['model']['proj_out.bias'].shape==(3,);original=torch.load(warm(ident['arm']),map_location='cpu',weights_only=False)['model_state'];assert set(original)==set(ck['model']) and all(original[k].shape==ck['model'][k].shape and torch.isfinite(ck['model'][k]).all() for k in original);return step

DELTA=0.9139534189451107
CAL=O/'card084-calibration.json'
BANK=O/'attraction-shape-training-diagnostic/training-pairs.npz'
BANK_SHA='cdbcbadeaeb4c3a08dc737ddf430cb981523541e3da9d561bdd96825c0003ffe'

def attraction_identity(arm, *, calibration=False):
 import math
 assert sha(BANK)==BANK_SHA, 'delta bank changed'
 result={'family':arm,'delta':DELTA,'coefficient':0.,'calibration_sha':None,
         'calibration_batches_sha':None,'bank_sha':BANK_SHA,
         'attraction_normalization':'positive_mean','calibration_only':calibration}
 if not calibration:
  c=read(CAL);assert c['PASS'] and c['runtime_sha']==source_check()
  assert c['original_head_sha']==sha(CHAMP) and c['delta']==DELTA
  assert c['input_manifest_sha']==sha(D/'inputs-manifest.json')
  assert c['graph_sha']==sha(D/'edges-fixed15.npz') and c['radii_sha']==sha(D/'radii.npy')
  assert c['uniform_q_sha']==sha(O/'degree-noise-readiness/uniform.npy') and c['bank_sha']==BANK_SHA
  assert len(c['batches'])==8 and [b['index'] for b in c['batches']]==list(range(8))
  assert all(sha(b['path'])==b['sha'] for b in c['batches']), 'calibration batch content changed'
  assert hashlib.sha256(''.join(b['sha'] for b in c['batches']).encode()).hexdigest()==c['batches_sha']
  assert c['sampler_parity']['PASS'] and c['sampler_parity']['n_distinct_ordered_batches']==8
  assert len(set(c['sampler_parity']['ordered_pair_hashes']))==8 and c['all_eight_model_states_unchanged']
  assert set(c['zero_step_counters'])=={'optimizer_steps_attempted','optimizer_steps_succeeded','positive_lr_optimizer_steps','executed_iters'}
  assert all(v==0 for v in c['zero_step_counters'].values())
  assert all(sha(b['path'])==b['sha'] for b in c['sampler_parity']['baseline_batches'])
  a=c['arms'][arm];v=a['coefficient'];ratios=a['raw_ratios']
  assert len(ratios)==8 and all(math.isfinite(x) and x>0 for x in ratios)
  assert max(ratios)/min(ratios)<=10 and math.isfinite(v) and v>0
  assert v==0.1/float(np.median(ratios)), 'coefficient formula mismatch'
  result.update(coefficient=v,calibration_sha=sha(CAL),calibration_batches_sha=c['batches_sha'])
 return result

def validate_identity(ident):
 import math
 assert ident['family'] in ARMS and ident['family']==ident['arm'], 'family identity'
 assert ident['delta']==DELTA and ident['bank_sha']==BANK_SHA, 'delta/bank identity'
 assert ident['dose']>0 and ident['dose']<=DOSE and ident['lr']==.0001, 'dose/LR identity'
 assert ident['attraction_normalization']=='positive_mean'
 v=ident['coefficient'];assert math.isfinite(v) and v>=0, 'coefficient identity'
 if not ident.get('calibration_only') and not ident.get('canary_off_control'):
  assert v>0 and ident['calibration_sha'] and ident['calibration_batches_sha'], 'calibration identity'
 assert ident['fneg_weight']==1. and ident['neg_tanh_gamma']==4., 'both-on recipe'

def require_gpu_stage():
 # Root external flock wrappers own both leases; chain exports this bound release hash.
 import os
 release=O/'card084-release.json'
 assert release.exists(), 'NO ROOT GPU RELEASE'
 r=read(release);assert r['PASS'] and r['card']=='084'
 assert os.environ.get('CARD084_RELEASE_SHA')==sha(release), 'GPU must run through released chain'
 assert r['runtime_sha']==source_check(), 'root release runtime mismatch'
 assert r['limits']=={'card_gpu_s':4200,'per_arm_gpu_s':1800,'global_vram_gib':30,'rss_gib':32,'deadline':'2026-09-13T23:50:55Z'}
 assert all(sha(p)==h for p,h in r['files'].items()), 'root release file drift'
