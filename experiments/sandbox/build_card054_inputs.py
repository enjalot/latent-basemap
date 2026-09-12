"""Training-only point/landmark bank; exact stored-input distances, no GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np,json,time,hashlib,torch
from scipy.spatial.distance import cdist
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card054-landmark');S=D.parent/'card052-global-pca'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();D.mkdir(exist_ok=True);assert not (D/'inputs-manifest.json').exists();old=json.loads((S/'inputs-manifest.json').read_text());assert old['PASS'];files=['train.f16.npy','edges-fixed15.npz','train-ids.npy','radii.npy','init.pt','global-pairs-before-outcomes.npz']
 for name in files:
  assert sha(S/name)==old['files'][name],name
  if not (D/name).exists():os.link(S/name,D/name)
 x=np.load(D/'train.f16.npy',mmap_mode='r');ids=np.load(D/'train-ids.npy');pool=np.random.default_rng(54054).choice(len(x),200000,replace=False);eligible=np.setdiff1d(np.arange(len(x)),pool);land=np.random.default_rng(54055).choice(eligible,64,replace=False);px=np.array(x[pool],copy=True);lx=np.array(x[land],copy=True);dist=np.empty((len(pool),len(land)),'f4')
 for i in range(0,len(pool),4096):dist[i:i+4096]=cdist(px[i:i+4096].astype('f8'),lx.astype('f8')).astype('f4')
 assert np.isfinite(dist).all() and (dist>0).all();np.savez(D/'landmark-bank.npz',pool_X=px,landmark_X=lx,distances=dist,pool_local=pool,landmark_local=land,pool_ids=ids[pool],landmark_ids=ids[land]);files+=['landmark-bank.npz']
 r={'PASS':True,'n_train':300000,'dim':1536,'files':{name:sha(D/name) for name in files},'parent_sha':old['parent_sha'],'init_state_sha256':old['init_state_sha256'],'parent_manifest_sha':sha(S/'inputs-manifest.json'),'builder_sha':sha(__file__),'seeds':{'pool':54054,'landmark':54055,'per_step':54056},'weight':.02,'distance_quantiles':np.quantile(dist,[0,.01,.5,.99,1]).tolist(),'cpu_s':time.monotonic()-start,'scope':'ExactFP16 inputs promoted toFP64 for raw Euclidean distance thenstoredFP32; no renormalization.200Kpool/64complement landmarks, trainingonly.'};(D/'inputs-manifest.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['PASS','cpu_s','distance_quantiles']},indent=2))
if __name__=='__main__':main()
