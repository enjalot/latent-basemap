"""Independent scalar weights, stored endpoints, actual CPU production consumption."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import sys,json,hashlib,resource,time
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card089-soft-reciprocity')
from card089_weights import weights,ARMS
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler

def main():
 start=time.monotonic();torch.set_num_threads(2);checks={};rng=np.random.default_rng(89089)
 for m in range(16):
  mask=np.zeros((32,15),bool)
  for row in mask:row[rng.choice(15,m,replace=False)]=True
  a,_=weights(mask,'reciprocal');b,_=weights(mask,'rank_count_control')
  for i in range(32):
   for j in range(15):
    assert a[i,j]==np.float32((4 if mask[i,j] else 1)*15/(15+3*m));assert b[i,j]==np.float32((4 if j<m else 1)*15/(15+3*m))
  assert np.array_equal(np.sort(a,1),np.sort(b,1)) and np.array_equal(a.sum(1,dtype='f8'),b.sum(1,dtype='f8'));checks['scalar_m'+str(m)]=True
 mask=np.load(O/'reciprocity-readiness-20260913/mutual-mask.npy',mmap_mode='r');ids=np.load(R.parent/'dino-arrival-t0/knn_indices.npy',mmap_mode='r');dist=np.load(R.parent/'dino-arrival-t0/knn_dists.npy',mmap_mode='r');assert dist.shape==ids.shape
 with np.load('/data/latent-basemap/substrates/card086-directed-membership/all_one-edges.npz') as z:
  src=z['sources'];dst=z['targets'];assert np.array_equal(dst.reshape(-1,15),ids) and np.array_equal(src.reshape(-1,15),np.broadcast_to(np.arange(len(ids))[:,None],ids.shape));assert (z['weights']==1).all()
 for arm in ARMS:
  with np.load(D/f'{arm}-edges.npz') as z:
   assert np.array_equal(z['sources'],src) and np.array_equal(z['targets'],dst);w=z['weights'].reshape(-1,15)
   for lo in range(0,len(ids),8192):
    hi=min(lo+8192,len(ids));m=np.asarray(mask[lo:hi]);count=m.sum(1);fav=m if arm=='reciprocal' else np.arange(15)<count[:,None];expected=np.where(fav,60.,15.)/(15+3*count[:,None]);assert np.array_equal(w[lo:hi],expected.astype('f4'));assert np.isfinite(dist[lo:hi]).all() and (np.diff(dist[lo:hi],axis=1)>=0).all()
  checks[arm+'_full_actual_endpoints_weights_rank_order']=True
 n=512;src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.array([rng.choice(np.delete(np.arange(n),i),15,False) for i in range(n)],dtype='i4');mut=(dst[dst]==np.arange(n)[:,None,None]).any(2);samplers=[];mass={};seen={a:[] for a in ARMS}
 for arm in ARMS:
  w,fav=weights(mut,arm);mass[arm]=float(w[mut].sum(dtype='f8')/w.sum(dtype='f8'));s=DeviceEdgeSampler(DeviceArrayDataset(np.arange(n,dtype='f4')[:,None],device='cpu'),src,dst.ravel(),w.ravel(),n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,device='cpu');s._stash_ids=True;samplers.append(s)
 for ep in range(16):
  for s in samplers:iter(s)
  for batch in range(len(samplers[0])):
   for arm,s in zip(ARMS,samplers):
    pos=s.pos_idx;out=next(s);idx=s.perm[pos:min(pos+s.num_pos,s.n_pos)];seen[arm].extend(idx.tolist());k=len(idx);assert np.array_equal(s._last_all_src[:k],src[idx]) and np.array_equal(s._last_all_dst[:k],dst.ravel()[idx]);assert (out[-1][:k]==1).all()
   a,b=samplers;assert torch.equal(a.gen.get_state(),b.gen.get_state()) and torch.equal(a._last_all_src[k:],b._last_all_src[k:]) and torch.equal(a._last_all_dst[k:],b._last_all_dst[k:])
 actual={a:float(mut.ravel()[seen[a]].mean()) for a in ARMS};assert all(abs(actual[a]-mass[a])<.01 for a in ARMS);assert seen[ARMS[0]]!=seen[ARMS[1]];checks['actual16epoch_CPU_consumption_noise_parity']=True
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert rss<8192
 r={'PASS':True,'checks':checks,'expected_reciprocal_fraction':mass,'observed_reciprocal_fraction':actual,'draws_per_arm':len(seen[ARMS[0]]),'cpu_wall_s':time.monotonic()-start,'max_rss_MiB':rss,'scope':'Stored entire30M endpoint/weight/rank audit and synthetic512-node production CPU sampling. No real feature/query/model/GPU access.'};(O/'card089-weights-cpu.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
