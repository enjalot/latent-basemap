"""Matched continuation of the frozen12M head with one final-activation intervention."""
import sys,time,json,hashlib
from pathlib import Path
import numpy as np
from _paths import ensure_paths
ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP
ROOT=Path(__file__).resolve().parents[2];OC=Path('/data/latent-basemap/sandbox/overseer-codex');SB=OC.parent
DATA=Path('/data/latent-basemap/substrates/card019-activation');OUT=SB/'card019-train'
HEAD=SB/'monet-random-dino-12m-pca768/champion-bs16k/model.pt';ARMS=['relu','leaky'];ACT={'relu':'relu','leaky':'leaky_relu_slope_0p01'};SNAPS=[20000,40000,60000]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,o):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');q.replace(p)
def state_sha(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def fit(arm,steps=60000,canary=False,resume_from=None,ckpt_steps=None,identity_override=None):
 assert arm in ARMS and torch.cuda.is_available();torch.set_num_threads(4)
 m=json.loads((DATA/'manifest.json').read_text());assert m['complete'] and sha(HEAD)==m['head_sha']
 assert all(sha(DATA/n)==h for n,h in m['files'].items())
 rt=json.loads((ROOT/'card019-runtime-sha.json').read_text());assert all(sha(ROOT/n)==h for n,h in rt.items())
 for name,mod in list(sys.modules.items()):
  if name.startswith('basemap.') and getattr(mod,'__file__',None):assert Path(mod.__file__).resolve().is_relative_to(ROOT)
 init=torch.load(HEAD,map_location='cpu',weights_only=False)['model_state_dict']
 identity={'card':'019','arm':arm,'activation':ACT[arm],'negative_slope':.01 if arm=='leaky' else 0.,'steps':steps,'lr':.0001,'lr_schedule':'constant','batch_size':16384,'rankneg_window':75000,'seed':42,'head_sha':sha(HEAD),'warm_named_sha':state_sha(init),'data_manifest_sha':sha(DATA/'manifest.json'),'runtime':rt,'canary':canary}
 dest=OUT/(arm if not canary else 'canary-'+arm);dest.mkdir(parents=True,exist_ok=True)
 if not canary:
  assert steps==60000 and not (dest/'complete.json').exists()
  if resume_from:assert json.loads((dest/'admission.json').read_text())==identity
  else:write(dest/'admission.json',identity)
 p=ParametricUMAP.load(str(HEAD),device='cuda');p.model=None;p.final_activation=ACT[arm]
 p.learning_rate=.0001;p.lr_schedule='constant';p.warmup_steps=0;p.n_epochs=10000;p._max_train_steps=steps;p.batch_size=16384;p.rankneg_window=75000;p.x_residency='auto';p.required_input_pipeline='device'
 for attr,value in [('anchor_ids_path',''),('anchor_hold_weight',0.),('replay_bank_path',''),('replay_weight',0.),('deriv_bank_path',''),('deriv_weight',0.)]:
  if hasattr(p,attr):setattr(p,attr,value)
 assert p.a==1.9328 and p.b==.7905 and p.fneg_weight==1. and p.neg_tanh_gamma==4. and p.positive_target_mode=='binary'
 p._card012_identity=identity if identity_override is None else identity_override
 p._checkpoint_step_targets=ckpt_steps if ckpt_steps is not None else ([] if canary else SNAPS)
 torch.manual_seed(42);np.random.seed(42);torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
 X=np.asarray(np.load(DATA/'substrate.f16.npy',mmap_mode='r'),dtype='f4');t=time.monotonic()
 p.fit(X,precomputed_edges_path=str(DATA/'edges-fixed15.npz'),random_state=42,verbose=False,warm_start_state=init,
       checkpoint_every_epochs=0 if canary else 1,checkpoint_dir=str(dest/'ckpt'),resume_from=resume_from,
       **({} if canary else {'snapshot_steps':SNAPS,'snapshot_dir':str(dest)}))
 st=dict(p._train_stats);assert st['positive_lr_optimizer_steps']==st['executed_iters']==steps
 assert st['lr_used_min']==st['lr_used_max']==.0001 and p._pipeline_info['x_residency']=='device_fp16'
 assert all(torch.isfinite(v).all() for v in p.model.state_dict().values())
 result={'status':'TRAINED','identity':identity,'successful_steps':steps,'endpoint_named_sha':state_sha(p.model.state_dict()),'warm_parameter_sha':p.warm_start_sha256,'train_stats':st,'stage_wall_s':time.monotonic()-t,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'pipeline':p._pipeline_info}
 if canary:return result
 p.save(str(dest/'model.pt'));result['endpoint_file_sha']=sha(dest/'model.pt')
 assert all(sha(ROOT/n)==h for n,h in rt.items());write(dest/'complete.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['identity','train_stats']},indent=2),flush=True)
 return result
if __name__=='__main__':
 import os
 fit(sys.argv[1],resume_from=os.environ.get('CARD019_RESUME') or None)
