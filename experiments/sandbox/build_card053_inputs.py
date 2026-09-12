"""CPU-only train-radius transforms; unchanged control keeps exact parent radius bytes."""
from pathlib import Path
import os,json,time,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import numpy as np,torch
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card053-sparse-scale');S=D.parent/'card033-scale4m';P=R.parent/'card033-train/model-actual3d.pt'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def state_sha(sd):
 h=hashlib.sha256()
 for k in sorted(sd):h.update(k.encode());h.update(np.ascontiguousarray(sd[k].numpy()).tobytes())
 return h.hexdigest()
def main():
 start=time.monotonic();D.mkdir(exist_ok=True);assert not (D/'inputs-manifest.json').exists(),'immutable existing input bundle';m=json.loads((S/'manifest.json').read_text());assert m['PASS'] and m['n']==4000000
 mapping={'train.f16.npy':'substrate.f16.npy','edges-fixed15.npz':'edges-fixed15.npz','train-ids.npy':'draw_ids.npy','train-source.npy':'draw_source.npy','r-half.npy':'r_actual.npy','r-raw.npy':'r_raw.npy'}
 for dest,src in mapping.items():
  assert sha(S/src)==m['files'][src],src
  if not (D/dest).exists():os.link(S/src,D/dest)
 raw=np.load(D/'r-raw.npy');base=np.maximum(raw/m['radius']['train_p95'],1e-6);half=np.load(D/'r-half.npy');assert np.array_equal(np.sqrt(base).astype('f4'),half),'parent radius is not exact half'
 sorted_r=np.sort(base);p=(np.searchsorted(sorted_r,base,side='left')+np.searchsorted(sorted_r,base,side='right'))/(2*len(base));u=np.clip((p-.7)/.3,0,1);w=1-3*u*u+2*u*u*u
 quarter=np.sqrt(half).astype('f4');taper=(half.astype('f8')**w).astype('f4');taper[p<=.7]=half[p<=.7]
 for name,x in [('r-quarter.npy',quarter),('r-taper.npy',taper),('training-percentile.npy',p),('taper-weight.npy',w)]:np.save(D/name,x)
 assert np.isfinite(taper).all() and (taper>0).all();assert np.max(np.abs(np.log(taper.astype('f8')))-np.abs(np.log(half.astype('f8'))))<2e-7
 obj=torch.load(P,map_location='cpu',weights_only=False);sd=obj['model_state_dict'];assert obj['input_dim']==1536 and obj['n_components']==3;torch.save({'model_state':sd},D/'init.pt')
 files={name:sha(D/name) for name in list(mapping)+['r-quarter.npy','r-taper.npy','training-percentile.npy','taper-weight.npy','init.pt']}
 result={'PASS':True,'n_train':4000000,'dim':1536,'files':files,'parent_sha':sha(P),'init_state_sha256':state_sha(sd),'original_manifest_sha':sha(S/'manifest.json'),'builder_sha':sha(__file__),'parent_radius_exact_bytes':True,'train_p95':m['radius']['train_p95'],'definition':'Use exact stored half radius h for unchanged control. quarter=sqrt(h), taper=h**smoothstep_weight(training empirical midrank of original raw/p95), preserving exact h through percentile0.7.','radius_quantiles':{a:np.quantile(x,[0,.1,.5,.7,.9,.95,.99,1]).tolist() for a,x in [('half',half),('quarter',quarter),('taper',taper)]},'rows_attenuated':int((w<1).sum()),'cpu_wall_s':time.monotonic()-start}
 (D/'inputs-manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['PASS','cpu_wall_s','radius_quantiles']},indent=2))
if __name__=='__main__':main()
