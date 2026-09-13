"""Constant-norm sparse inactive-row repair, one original backbone forward."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex';sys.path.insert(0,str(R))
from basemap.pumap.parametric_umap.core import ParametricUMAP
BATCH=256;HEADS=['ordinary2m','half4m']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def verify():
 m=read(R/'card061-runtime-sha.json');assert all(sha(R/k)==v for k,v in m.items());release=read(O/'card061-engineering-release.json');assert release['PASS'] and all(sha(p)==v for p,v in release['files'].items());return sha(R/'card061-runtime-sha.json')
def load(name,device):
 panel=read(O/'card057-panel/manifest.json');row=panel['heads'][name];assert sha(row['model']['path'])==row['model']['sha'];m=ParametricUMAP.load(row['model']['path'],device=device).model.float().eval().requires_grad_(False)
 assert m.proj_in.in_features==1536 and m.proj_out.out_features==3 and isinstance(m.up[1],torch.nn.ReLU)
 for key,mod in list(sys.modules.items()):
  if key.startswith('basemap') and getattr(mod,'__file__',None):assert Path(mod.__file__).resolve().is_relative_to(R),'nonisolated basemap import'
 c=read(O/'card059-score.json')['heads'][name]['constant_norm'];assert np.isfinite(c) and 0<c<=.001*row['radius'];return m,c,row['radius']
def kernel(teacher,z,weight,c,collect=False):
 inactive=z.amax(1)<=0;idx=torch.nonzero(inactive).flatten()
 if not idx.numel():return teacher,torch.zeros((len(z),3),dtype=torch.float64,device=z.device) if collect else None,inactive
 v=z[idx].double()@weight.double().T;n=torch.linalg.vector_norm(v,dim=1);delta=v*(c/n.clamp_min(torch.finfo(torch.float64).tiny))[:,None];delta=torch.where((n>0)[:,None],delta,torch.zeros_like(delta));y=teacher.clone();y[idx]=(teacher[idx].double()+delta).float();full=None
 if collect:full=torch.zeros((len(z),3),dtype=torch.float64,device=z.device);full[idx]=delta
 return y,full,inactive
@torch.inference_mode()
def project(m,x,c,mode='repair',collect=False):
 assert mode in ['teacher','repair'];held={};hook=None;out=[];teacher_out=[];offsets=[];mask=[];pre=[];device=next(m.parameters()).device
 if mode=='repair':hook=m.up[0].register_forward_hook(lambda _,args,z:held.update(z=z))
 try:
  for lo in range(0,len(x),BATCH):
   xx=torch.from_numpy(np.array(x[lo:lo+BATCH],dtype='f4',copy=True)).to(device);teacher=m(xx)
   if mode=='repair':
    zz=held.pop('z');y,d,b=kernel(teacher,zz,m.proj_out.weight,c,collect=collect)
    if collect:teacher_out.append(teacher.cpu().numpy());offsets.append(d.cpu().numpy());mask.append(b.cpu().numpy());pre.append(zz.cpu().numpy())
   else:y=teacher
   out.append(y.cpu().numpy())
 finally:
  if hook is not None:hook.remove()
 result={'xy':np.concatenate(out)}
 if collect:result.update(teacher=np.concatenate(teacher_out),delta64=np.concatenate(offsets),inactive=np.concatenate(mask),preactivation=np.concatenate(pre))
 assert all(np.isfinite(v).all() for v in result.values());return result
def device_setup():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');assert torch.cuda.is_available()
