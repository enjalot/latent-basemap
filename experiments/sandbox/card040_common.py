"""Immutable Card040 configuration and frozen-body inference helpers. No core.fit changes."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import numpy as np,torch
from torch import nn
R=Path(__file__).resolve().parents[2];SB=R.parent;O=SB/'overseer-codex';DATA=SB/'card040-data';TRAIN=SB/'card040-train';BODY=SB/'monet-random-dino-12m-pca768/champion-bs16k/model.pt';LEAKY=SB/'card019-train/leaky/model.pt';PCA=Path('/data2/monet/random-dino-6m/pca768-model.npz');DRAW=Path('/data/latent-basemap/substrates/card019-activation/draw_ids.npy');SOURCE=DRAW.parent/'draw_source.npy';C=O/'card021-confirmation';SEED=40040;DOSE=20000;BATCH=4096;LR=.001;WD=.01;CLIP=1.;SLOPE=.0001;ALPHA=.01;FORWARD_BATCH=256;ARMS=['post','pre'];STEPS=[5000,10000,15000,20000]
BODY_SHA='7c38430a492a1ff57d83df603d637e28e9c20533759bd3d181c0e5e5c76b7f14';LEAKY_SHA='4aacf7fe51afcfb0fcd8273ff94bd1bb0c984163806e25f639feee1901dba6b7';PCA_SHA='35b08ea002a5937408e488237d351bad0c54d88ca65b49ae81d3205c83ef51a1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def state_sha(sd):
 h=hashlib.sha256()
 for k in sorted(sd):h.update(k.encode());h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def runtime_check():
 m=json.loads((R/'card040-runtime-sha.json').read_text());assert all(sha(R/n)==v for n,v in m.items()),'runtime manifest mismatch';return sha(R/'card040-runtime-sha.json')
def loaded_check():
 paths={}
 for n,m in list(sys.modules.items()):
  if not n.startswith('basemap') or not getattr(m,'__file__',None):continue
  p=Path(m.__file__).resolve();assert p.is_relative_to(R),(n,p);paths[n]={'path':str(p),'sha':sha(p)}
 return paths
def raw(ids):
 ids=np.asarray(ids,'i8');assert ids.ndim==1 and (ids>=0).all() and (ids<103816750).all();x=np.empty((len(ids),1536),'f4')
 for lo,hi,name in [(0,19344847,'pool-20m'),(19344847,103816750,'pool-complement-88m')]:
  ix=np.flatnonzero((ids>=lo)&(ids<hi))
  if len(ix):x[ix]=np.load(Path('/data2/monet')/name/'dino1536.f16.npy',mmap_mode='r')[ids[ix]-lo]
 assert np.isfinite(x).all() and (np.linalg.norm(x,axis=1)>0).all();return x
class Readout(nn.Module):
 def __init__(self):
  super().__init__();self.fc1=nn.Linear(2048,64);self.act=nn.GELU();self.fc2=nn.Linear(64,2);nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
 def forward(self,x):return self.fc2(self.act(self.fc1(x)))
def load_body(which='teacher',device='cpu'):
 from basemap.pumap.parametric_umap.core import ParametricUMAP
 p=BODY if which=='teacher' else LEAKY;assert sha(p)==(BODY_SHA if which=='teacher' else LEAKY_SHA);u=ParametricUMAP.load(str(p),device=device);m=u.model.float().eval().requires_grad_(False);assert m.proj_in.in_features==768 and m.proj_out.in_features==2048 and m.proj_out.out_features==2
 saved=torch.load(p,map_location='cpu',weights_only=False)
 if which=='teacher':assert saved.get('final_activation','relu')=='relu' and isinstance(m.up[1],nn.ReLU)
 else:
  assert saved['final_activation']=='leaky_relu_slope_0p01','unexpected continuation activation'
  # This isolated legacy loader omits the saved final_activation field. Restore its declared module, without altering any weights.
  m.up[1]=nn.LeakyReLU(negative_slope=.01)
  assert isinstance(m.up[1],nn.LeakyReLU) and m.up[1].negative_slope==.01
 assert all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k]) for k,v in m.state_dict().items()),'body weights changed during activation restoration'
 loaded_check();return m
@torch.inference_mode()
def frozen_views(m,x):
 held={};h=m.up[0].register_forward_hook(lambda module,args,z:held.update(pre=z))
 try:t=m(x)
 finally:h.remove()
 z=held['pre'];f=m.proj_out(torch.nn.functional.leaky_relu(z,SLOPE));assert torch.isfinite(z).all() and torch.isfinite(t).all() and torch.isfinite(f).all();return t,f,z

def validate_inputs():
 assert sha(BODY)==BODY_SHA and sha(LEAKY)==LEAKY_SHA and sha(PCA)==PCA_SHA
 m=json.loads((DATA/'inputs.json').read_text());assert m['PASS'] and m['teacher_sha']==BODY_SHA and m['leaky_sha']==LEAKY_SHA and m['PCA_sha']==PCA_SHA
 for n,h in m['files'].items():assert sha(DATA/n)==h,('input hash',n)
 assert sha(DATA/'init.pt')==m['init_file_sha']
 for n,row in m['source_fingerprints'].items():
  s=Path(n).stat();assert {k:getattr(s,k) for k in row}==row,('source fingerprint',n)
 return m

def validate_bank():
 m=json.loads((DATA/'bank.json').read_text());assert m['PASS'] and m['input_manifest_sha']==sha(DATA/'inputs.json') and m['teacher_sha']==BODY_SHA and m['leaky_sha']==LEAKY_SHA and m['runtime_manifest_sha']==runtime_check()
 for n,h in m['files'].items():assert sha(DATA/n)==h,('bank hash',n)
 assert m['inactive_n']>=40 and m['active_n']>=40 and np.isfinite(m['target_scale']) and m['target_scale']>0
 d=np.load(DATA/'delta.f32.npy');c=np.load(DATA/'target-xy.f32.npy');f=np.load(DATA/'fixed-xy.f32.npy');assert d.shape==c.shape==f.shape==(300000,2) and np.isfinite(d).all()
 assert np.array_equal(d,(c.astype('f8')-f.astype('f8')).astype('f4')), 'residual target identity'
 assert m['target_scale']==float(np.sqrt(np.mean(np.square(d.astype('f8'))))), 'target RMS contract'
 positive=np.load(DATA/'positive-final.npy');inactive=np.load(DATA/'inactive-rows.npy');active=np.load(DATA/'active-rows.npy');assert positive.shape==(300000,) and np.array_equal(inactive,np.flatnonzero(positive==0)) and np.array_equal(active,np.flatnonzero(positive>0)) and len(inactive)==m['inactive_n'] and len(active)==m['active_n'], 'bank stratum identity'
 assert m['precision']=='float32' and m['TF32'] is False and m['batch_size']==FORWARD_BATCH
 return m

def device_setup():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');assert torch.cuda.is_available()
def identity(basis,dose=DOSE):
 assert basis in ARMS;bank=json.loads((DATA/'bank.json').read_text());init=torch.load(DATA/'init.pt',map_location='cpu',weights_only=False)['model']
 return {'basis':basis,'dose':dose,'batch':BATCH,'seed':SEED,'LR':LR,'WD':WD,'clip':CLIP,'precision':'float32_noAMP','architecture':'2048_64_GELU_2','init_sha':state_sha(init),'bank_sha':sha(DATA/'bank.json'),'input_manifest_sha':sha(DATA/'inputs.json'),'normalization_sha':sha(DATA/'normalization.npz'),'target_scale':bank['target_scale'],'runtime_manifest_sha':runtime_check()}
def resources():
 import resource
 free,total=torch.cuda.mem_get_info();used=(total-free)/2**30;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert used<30 and rss<12288,(used,rss);return {'global_gpu_gib':used,'max_rss_mib':rss,'allocated_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
