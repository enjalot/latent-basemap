"""Card020: explicitly migrate one frozen 20K parent; all child resumes strict."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import hashlib,json,time,sys
import numpy as np
import torch
import card016_model,run_card016_arm
from card016_model import CompactProjector,loss_for
from run_card016_arm import train_step
ROOT=Path(__file__).resolve().parents[2]; SB=Path('/data/latent-basemap/sandbox'); OC=SB/'overseer-codex'; DATA=SB/'card016-data';POOL=SB/'card012-pool';OUT=SB/'card020-train';PARENT=SB/'card016-train/compact_l1/step-20000.pt';SNAPS=[40000,80000]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def objsha(value):
 h=hashlib.sha256()
 def add(x):
  if torch.is_tensor(x):h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=lambda k:str(k)):h.update(str(k).encode());add(x[k])
  elif isinstance(x,(tuple,list)):
   for v in x:add(v)
  else:h.update(repr(x).encode())
 add(value);return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def base():return json.loads((ROOT/'card020-admission-base.json').read_text())
def identity(canary=False):
 b=base();return {'card':'020','parent':b,'total_successful_steps':80000,'parent_steps':20000,'batch':8192,'lr':.001,'objective':'compact_l1','precision':'bf16_hidden_matmuls_FP32_output_loss_parameters','seed':16016,'canary':bool(canary),'runtime_manifest_sha':sha(ROOT/'card020-runtime-sha.json')}
def validate_inputs():
 assert all(Path(mod.__file__).resolve().is_relative_to(ROOT) for mod in [card016_model,run_card016_arm]),'loaded helper path mismatch'
 b=base();assert sha(PARENT)==b['parent_file_sha'],'parent file mismatch';assert sha(DATA/'manifest.json')==b['data_manifest_sha'],'data manifest mismatch';assert sha(POOL/'pool_X.f16.npy')==b['input_file_sha'],'input file mismatch'
 m=json.loads((DATA/'manifest.json').read_text());assert all(sha(DATA/k)==v for k,v in m['files'].items()),'data files mismatch'
 runtime=json.loads((ROOT/'card020-runtime-sha.json').read_text());assert all(sha(ROOT/k)==v for k,v in runtime.items()),'runtime mismatch'
 return m

def init_state(device,ident,resume=None):
 if ident!=identity(bool(ident.get('canary'))):raise ValueError('admission configuration mismatch')
 model=CompactProjector().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);gen=torch.Generator(device=device)
 if resume is None:
  ck=torch.load(PARENT,map_location='cpu',weights_only=False);b=ident['parent']
  if sha(PARENT)!=b['parent_file_sha'] or ck['successful_steps']!=20000:raise ValueError('parent boundary mismatch')
  for key,field in [('model_state_dict','parent_model_sha'),('optimizer_state_dict','parent_optimizer_sha'),('batch_rng','parent_batch_rng_sha'),('cpu_rng','parent_cpu_rng_sha'),('cuda_rng','parent_cuda_rng_sha')]:
   if objsha(ck[key])!=b[field]:raise ValueError('parent state mismatch '+field)
  if ck['identity']!=b['parent_identity']:raise ValueError('parent identity mismatch')
  for st in ck['optimizer_state_dict']['state'].values():
   if float(st['step'])!=20000:raise ValueError('parent optimizer dose mismatch')
  migrated=True
 else:
  ck=torch.load(resume,map_location='cpu',weights_only=False)
  if ck.get('identity')!=ident:raise ValueError('child admission identity mismatch')
  if not 20000<ck['successful_steps']<=80000:raise ValueError('child boundary mismatch')
  migrated=False
 model.load_state_dict(ck['model_state_dict'],strict=True);opt.load_state_dict(ck['optimizer_state_dict']);assert all(g['lr']==.001 for g in opt.param_groups)
 if str(device)=='cpu' and resume is None and ident['canary']:gen.manual_seed(16016) # CPU-only algebra canary; CUDA RNG is device-specific.
 else:gen.set_state(ck['batch_rng'].cpu())
 torch.set_rng_state(ck['cpu_rng'].cpu())
 if str(device).startswith('cuda'):torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
 return model,opt,gen,int(ck['successful_steps']),{'migrated_parent':migrated,'model_sha':objsha(model.state_dict()),'optimizer_sha':objsha(opt.state_dict()),'batch_rng_sha':objsha(gen.get_state())}

def checkpoint(path,model,opt,gen,step,ident,elapsed=0):
 value={'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'batch_rng':gen.get_state(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,'successful_steps':step,'identity':ident,'elapsed_fit_s':elapsed}
 tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def gpu_data():
 raw=np.load(POOL/'pool_X.f16.npy',mmap_mode='r');free,total=torch.cuda.mem_get_info();required=raw.nbytes+32*(1<<20)+6*(1<<30)
 assert free>=required,'global VRAM headroom: input bank plus6GiB required'
 X=torch.empty(raw.shape,dtype=torch.float16,device='cuda')
 for s in range(0,len(raw),8192):X[s:s+8192].copy_(torch.from_numpy(np.array(raw[s:s+8192],copy=True)))
 target=torch.from_numpy(np.load(DATA/'targets_normalized.npy')).cuda();rows=torch.from_numpy(np.load(DATA/'fit_rows.npy')).cuda();grid=torch.from_numpy(np.load(DATA/'density_grid.npy')).cuda()
 free,total=torch.cuda.mem_get_info();assert (total-free)/2**30<30,'global used VRAM>=30GiB after input load'
 return X,target,rows,grid

def validate_endpoint():
 ad=json.loads((OUT/'admission.json').read_text());done=json.loads((OUT/'complete.json').read_text());assert ad==identity() and done['identity']==ad
 parent=base();assert done['start_receipt']['model_sha']==parent['parent_model_sha'] and done['start_receipt']['optimizer_sha']==parent['parent_optimizer_sha'] and done['start_receipt']['batch_rng_sha']==parent['parent_batch_rng_sha']
 hashes=[]
 for step in SNAPS:
  p=OUT/f'step-{step}.pt';c=torch.load(p,map_location='cpu',weights_only=False);assert c['identity']==ad and c['successful_steps']==step and p.stat().st_mtime>=(OUT/'admission.json').stat().st_mtime
  assert all(torch.isfinite(v).all() for v in c['model_state_dict'].values())
  for st in c['optimizer_state_dict']['state'].values():
   assert float(st['step'])==step and all(bool(torch.isfinite(v).all()) for v in st.values() if torch.is_tensor(v))
  assert all(g['lr']==.001 for g in c['optimizer_state_dict']['param_groups']);assert c['batch_rng'].numel()>0 and c['cuda_rng'].numel()>0
  hashes.append(objsha(c['model_state_dict']))
 assert len(set(hashes))==2 and hashes[-1]==done['endpoint_state_sha'] and done['successful_steps']==80000
 e=torch.load(OUT/'model.pt',map_location='cpu',weights_only=False);assert e['identity']==ad and e['successful_steps']==80000 and objsha(e['model_state_dict'])==hashes[-1] and sha(OUT/'model.pt')==done['endpoint_file_sha']
 manifest=json.loads((DATA/'manifest.json').read_text());assert e['center']==manifest['center'] and e['span']==manifest['span'],'endpoint native conversion mismatch'
 assert 0<=done['peak_allocated_gib']<30 and 0<=done['global_used_gib']<30 and done['input_file_sha']==parent['input_file_sha']
 return {'PASS':True,'successful_steps':80000,'parent_file_sha':parent['parent_file_sha'],'endpoint_state_sha':hashes[-1]}

def train():
 assert torch.cuda.is_available();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 m=validate_inputs();ident=identity();OUT.mkdir(exist_ok=True);ad=OUT/'admission.json'
 if ad.exists():assert json.loads(ad.read_text())==ident
 else:write(ad,ident)
 assert not (OUT/'complete.json').exists();X,Y,rows,grid=gpu_data();ckpts=[OUT/f'step-{s}.pt' for s in SNAPS if (OUT/f'step-{s}.pt').exists()];model,opt,gen,step,start=init_state('cuda',ident,ckpts[-1] if ckpts else None)
 if ckpts:start=json.loads((OUT/'start-receipt.json').read_text())
 else:write(OUT/'start-receipt.json',start)
 torch.cuda.reset_peak_memory_stats();t=time.monotonic();curve=[];dev=torch.from_numpy(np.load(DATA/'dev_rows.npy')).cuda()
 while step<80000:
  train_step(model,opt,gen,X,Y,rows,grid,'compact_l1');step+=1
  if step in SNAPS:
   checkpoint(OUT/f'step-{step}.pt',model,opt,gen,step,ident,time.monotonic()-t)
   with torch.inference_mode():
    res=torch.cat([model(X.index_select(0,dev[s:s+8192]).float())-Y.index_select(0,dev[s:s+8192]) for s in range(0,len(dev),8192)])
    curve.append({'step':step,'dev_mae':float(res.abs().mean()),'dev_mse':float(res.square().mean()),'elapsed_s':time.monotonic()-t})
   write(OUT/f'dev-{step}.json',curve[-1]);print(json.dumps(curve[-1]),flush=True)
 torch.save({'model_state_dict':model.state_dict(),'identity':ident,'successful_steps':step,'center':m['center'],'span':m['span']},OUT/'model.pt')
 free,total=torch.cuda.mem_get_info();global_used=(total-free)/2**30;assert global_used<30,'global used VRAM>=30GiB after training'
 done={'global_used_gib':global_used,'status':'TRAINED','identity':ident,'successful_steps':step,'start_receipt':start,'endpoint_state_sha':objsha(model.state_dict()),'endpoint_file_sha':sha(OUT/'model.pt'),'input_file_sha':sha(POOL/'pool_X.f16.npy'),'fit_s':time.monotonic()-t,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30};write(OUT/'complete.json',done);validate_inputs();write(OC/'card020-validation.json',validate_endpoint())

if __name__=='__main__':train()
