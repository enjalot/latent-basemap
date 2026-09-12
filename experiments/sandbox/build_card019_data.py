"""Recover original raw rows, apply the frozen deployed PCA contract, then store training fp16."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OC=Path('/data/latent-basemap/sandbox/overseer-codex')
DATA=Path('/data/latent-basemap/substrates/card019-activation');OLD=Path('/data/latent-basemap/substrates/card010-adaptive')
PCA=Path('/data2/monet/random-dino-6m/pca768-model.npz');RAW=Path('/data2/monet/pool-20m/dino1536.f16.npy')
HEAD=Path('/data/latent-basemap/sandbox/monet-random-dino-12m-pca768/champion-bs16k/model.pt')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,o):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');q.replace(p)
def main():
 t=time.monotonic();torch.set_num_threads(4);DATA.mkdir(exist_ok=True)
 assert not (DATA/'manifest.json').exists(),'preserve admitted bundle'
 ids=np.load(OLD/'draw_ids.npy');sources=np.load(OLD/'draw_source.npy');raw=np.load(RAW,mmap_mode='r');pca=np.load(PCA)
 assert ids.shape==(300000,) and len(np.unique(ids))==300000 and ids.min()>=0 and ids.max()<len(raw)
 checks={}
 for name in ['ref_idx.npy','val_idx.npy']:
  checks['excludes_'+name]=not np.isin(ids,np.load(Path('/data2/monet/eval-common-v2')/name)).any()
 assert all(checks.values())
 mean=torch.from_numpy(pca['mean']);comp=torch.from_numpy(pca['components'])
 out=np.lib.format.open_memmap(DATA/'substrate.f16.npy',mode='w+',dtype='f2',shape=(len(ids),768))
 errors=[]
 with torch.inference_mode():
  for lo in range(0,len(ids),4096):
   x=torch.from_numpy(np.asarray(raw[ids[lo:lo+4096]],dtype='f4'));z=torch.nn.functional.normalize((x-mean)@comp,dim=1)
   assert torch.isfinite(z).all() and torch.all(torch.linalg.vector_norm(z,dim=1)>.999)
   out[lo:lo+len(z)]=z.numpy();errors.append(float(np.abs(z.numpy()-out[lo:lo+len(z)].astype('f4')).max()))
 out.flush();del out
 for name in ['draw_ids.npy','draw_source.npy','edges-fixed15.npz']:
  (DATA/name).symlink_to(OLD/name)
 panel=np.load(OC/'band-investigation-20260911/pile-audit-v1/frozen-panel.npz');pile=panel['full_ids'][panel['role']=='pile']
 checks['no_pile_update_rows']=not np.isin(ids,pile).any()
 assert all(checks.values())
 m={'complete':True,'n_rows':len(ids),'input_dim':768,'contract':'Original raw fp16 source castFP32; (X-mean)@components then L2; training storageFP16. No raw pre-normalization. Deployment eval uses FP32 PCA inputs without training quantization.',
    'checks':checks,'head_sha':sha(HEAD),'pca_sha':sha(PCA),'head':str(HEAD),'pca':str(PCA),
    'source':{'path':str(RAW),'shape':list(raw.shape),'bytes':RAW.stat().st_size,'mtime_ns':RAW.stat().st_mtime_ns},
    'source_counts':{str(s):int((sources==s).sum()) for s in np.unique(sources)},'max_training_fp16_roundtrip_error':max(errors),
    'files':{n:sha(DATA/n) for n in ['substrate.f16.npy','draw_ids.npy','draw_source.npy','edges-fixed15.npz']},'builder_sha':sha(__file__),'cpu_wall_s':time.monotonic()-t}
 write(DATA/'manifest.json',m);write(OC/'card019-data-manifest.json',m);print(json.dumps(m,indent=2))
if __name__=='__main__':main()
