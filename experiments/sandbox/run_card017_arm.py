"""Fixed full-D fresh-head recipe; sampling strategy is the only intervention."""
import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
from _paths import ensure_paths
ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP
SUBD=Path('/data/latent-basemap/substrates/card017-support')
OC=Path('/data/latent-basemap/sandbox/overseer-codex'); ARMS=['uniform','geometric','hybrid']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def write(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)

ROOT=Path(__file__).resolve().parents[2]; SB=OC.parent; OUT=SB/'card017-train'
CHAMPION=SB/'dino-arrival-t0/champion-bs16k/model.pt'; INIT=Path('/data/latent-basemap/substrates/card010-adaptive/init-card010.pt')
SNAPS=[20000,30000,40000,60000]


def state_sha(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()


def configuration(steps):
 p=ParametricUMAP.load(str(CHAMPION),device='cuda');p.model=None;p.n_components=2
 p.learning_rate=.001;p.lr_schedule='constant';p.batch_size=16384;p.warmup_steps=0;p.n_epochs=10000;p._max_train_steps=steps;p.rankneg_window=75000
 for a,v in [('anchor_ids_path',''),('anchor_hold_weight',0.),('replay_bank_path',''),('replay_weight',0.),('deriv_bank_path',''),('deriv_weight',0.)]:
  if hasattr(p,a):setattr(p,a,v)
 for attr in ['_card013_radii','_card011_stages','_card012_refresh']:
  assert not hasattr(p,attr),'unexpected opt-in intervention'
 return p


def fit(arm,steps=60000,canary=False):
 assert torch.cuda.is_available();torch.set_num_threads(4)
 folder=SUBD/arm;draw=json.loads((SUBD/'draw-manifest.json').read_text()); graph=json.loads((SUBD/'graph-manifest.json').read_text())
 assert draw['complete'] and graph['complete'] and graph['draw_manifest_sha']==sha(SUBD/'draw-manifest.json')
 assert all(sha(folder/n)==h for n,h in draw['arms'][arm]['files'].items())
 assert all(sha(folder/n)==h for n,h in graph['graphs'][arm]['files'].items())
 runtime=json.loads((ROOT/'card017-runtime-sha.json').read_text());assert all(sha(ROOT/n)==h for n,h in runtime.items())
 for name,module in list(sys.modules.items()):
  if name.startswith('basemap.') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(ROOT)
 obj=torch.load(INIT,map_location='cpu',weights_only=False);init=obj['model_state'];assert obj['init_state_sha256']=='589895f037d406ae'
 identity={'card':'017','arm':arm,'steps':steps,'n_components':2,'lr':.001,'lr_schedule':'constant','batch_size':16384,'rankneg_window':75000,'seed':42,
           'draw_manifest_sha':sha(SUBD/'draw-manifest.json'),'graph_manifest_sha':sha(SUBD/'graph-manifest.json'),'shared_init_parameter_order_sha':'589895f037d406ae',
           'shared_init_named_sha':state_sha(init),'runtime':runtime,'canary':canary}
 dest=OUT/arm;dest.mkdir(parents=True,exist_ok=True)
 if not canary:
  assert steps==60000 and not (dest/'complete.json').exists(),'do not overwrite completed head'
  write(dest/'admission.json',identity)
 p=configuration(steps)
 torch.manual_seed(42);np.random.seed(42);torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
 X=np.asarray(np.load(folder/'substrate.f16.npy',mmap_mode='r'),dtype='f4')
 start=time.monotonic()
 p.fit(X,precomputed_edges_path=str(folder/'edges-fixed15.npz'),random_state=42,verbose=False,warm_start_state=init,
       **({} if canary else {'snapshot_steps':SNAPS,'snapshot_dir':str(dest)}))
 stats=dict(p._train_stats)
 assert stats['executed_iters']==stats['positive_lr_optimizer_steps']==steps
 assert stats['lr_used_min']==stats['lr_used_max']==.001
 assert p.warm_start_sha256=='589895f037d406ae'
 assert all(torch.isfinite(t).all() for t in p.model.state_dict().values())
 endpoint=state_sha(p.model.state_dict())
 if canary:return {'arm':arm,'steps':steps,'endpoint_named_sha':endpoint,'warm_parameter_sha':p.warm_start_sha256,'stats':stats}
 xy=np.asarray(p.transform(X,batch_size=8192),dtype='f4');assert xy.shape==(300000,2) and np.isfinite(xy).all()
 np.save(dest/'train-xy.npy',xy);p.save(str(dest/'model.pt'))
 result={'status':'TRAINED','identity':identity,'successful_steps':steps,'endpoint_named_sha':endpoint,'endpoint_file_sha':sha(dest/'model.pt'),
         'warm_parameter_sha':p.warm_start_sha256,'train_stats':stats,'stage_wall_s':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
 assert all(sha(ROOT/n)==h for n,h in runtime.items())
 write(dest/'complete.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['identity','train_stats']},indent=2),flush=True)
 return result


if __name__=='__main__':
 a=sys.argv[1];assert a in ARMS;fit(a)
