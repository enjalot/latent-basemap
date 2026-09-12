"""CPU admission of existing2M support, directed15 graph and correctly interpreted radii."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json,time,hashlib,datetime as dt
import numpy as np
SRC=Path('/data/latent-basemap/substrates/dino-arrival-t0');OLD=Path('/data/latent-basemap/sandbox/dino-arrival-t0')
OUT=Path('/data/latent-basemap/substrates/card018-scale2m');OUT.mkdir(parents=True,exist_ok=True)
OC=Path('/data/latent-basemap/sandbox/overseer-codex');SEAL=Path('/data2/monet/eval-common-v2')


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def write(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)


def norm(x):
 x=np.asarray(x,dtype='f4');return x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)


def exact_panel(X,panel,k=15):
 q=norm(X[panel]);bs=16384;best_s=np.full((len(panel),k),-np.inf,dtype='f4');best_i=np.zeros((len(panel),k),dtype='i8')
 for s in range(0,len(X),bs):
  sim=q@norm(X[s:s+bs]).T
  here=(panel>=s)&(panel<s+sim.shape[1]);sim[np.flatnonzero(here),panel[here]-s]=-np.inf
  ix=np.argpartition(-sim,k-1,axis=1)[:,:k];ss=np.take_along_axis(sim,ix,axis=1)
  all_s=np.concatenate([best_s,ss],1);all_i=np.concatenate([best_i,ix+s],1)
  take=np.argsort(-all_s,axis=1,kind='stable')[:,:k]
  best_s=np.take_along_axis(all_s,take,1);best_i=np.take_along_axis(all_i,take,1)
 return best_s,best_i


def main():
 start=time.monotonic();assert not (OUT/'manifest.json').exists(),'do not overwrite admitted inputs'
 X=np.load(SRC/'substrate.f16.npy',mmap_mode='r');ids=np.load(SRC/'draw_idx.npy');source=np.load(SRC/'subsets.npy',allow_pickle=True).astype(str)
 knn=np.load(OLD/'knn_indices.npy',mmap_mode='r');dist=np.load(OLD/'knn_dists.npy',mmap_mode='r');n=len(X)
 checks={'shape':X.shape==(2000000,1536) and knn.shape==dist.shape==(2000000,15),'unique_draw':len(np.unique(ids))==n,
         'no_self':not (knn==np.arange(n)[:,None]).any(),'neighbor_bounds':int(knn.min())>=0 and int(knn.max())<n,
         'unique_neighbors':not (np.sort(knn,axis=1)[:,1:]==np.sort(knn,axis=1)[:,:-1]).any(),'dist_finite_nonnegative':bool(np.isfinite(dist).all() and (dist>=0).all())}
 expected=json.loads((SRC/'manifest.json').read_text())['by_source']
 counts={str(k):int(v) for k,v in zip(*np.unique(source,return_counts=True))};checks['source_counts']=counts==expected
 for file in ['ref_idx.npy','val_idx.npy']:checks['excluded_'+file]=not np.isin(ids,np.load(SEAL/file)).any()
 checks['fresh_reserve_excluded']=not np.isin(ids,np.load(OC.parent/'card012-pool/confirm_reserved.npy')).any()
 assert all(checks.values()),checks
 raw=np.sqrt(2*np.asarray(dist,dtype='f8').mean(1));p95=float(np.percentile(raw,95));r=raw/p95
 floor_count=int((r<1e-6).sum());checks['floor_fraction']=floor_count/n<=.001
 r=np.maximum(r,1e-6).astype('f4');checks['radii_finite_positive']=bool(np.isfinite(r).all() and (r>0).all())
 rng=np.random.default_rng(18018);panel=rng.choice(n,1024,replace=False)
 q=norm(X[panel]);neighbors=norm(X[knn[panel].reshape(-1)]).reshape(1024,15,1536)
 actual_sq=np.square(neighbors-q[:,None,:]).sum(2,dtype='f8');actual=np.sqrt(actual_sq.mean(1))
 nz=actual>1e-8;relative=np.abs(raw[panel][nz]-actual[nz])/actual[nz]
 checks['radius_convention']=float(np.percentile(relative,95))<=.001
 stored=np.asarray(dist[panel],dtype='f8');cos=actual_sq/2
 np.savez(OUT/'radius-audit.npz',panel_local=panel,panel_ids=ids[panel],stored_cosine_distances=stored,recomputed_euclidean_squared=actual_sq,
          raw_radius_cached=raw[panel],raw_radius_recomputed=actual,relative_nonzero=relative)
 # Separate exact normalized-full-D neighbor audit, fixed before viewing results.
 exact_ids=panel[:128];t=time.monotonic();sim,truth=exact_panel(X,exact_ids)
 recalls=np.array([len(set(a)&set(b))/15 for a,b in zip(knn[exact_ids],truth)])
 np.savez(OUT/'neighbor-audit.npz',panel_local=exact_ids,panel_ids=ids[exact_ids],stored_neighbors=knn[exact_ids],exact_neighbors=truth,exact_similarities=sim,recall15=recalls)
 print('neighbor audit',float(recalls.mean()),'wall',time.monotonic()-t,flush=True)
 checks['neighbor_fidelity']=float(recalls.mean())>=.98
 assert all(checks.values()),checks
 src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.asarray(knn).reshape(-1)
 np.savez(OUT/'edges-fixed15.npz',sources=src,targets=dst,weights=np.ones(len(src),dtype='f4'),n_nodes=np.int64(n))
 np.save(OUT/'r_raw.npy',raw);np.save(OUT/'r_actual.npy',r);np.save(OUT/'draw_ids.npy',ids);np.save(OUT/'draw_source.npy',source)
 # Link the exact existing substrate; never duplicate or silently re-normalize its training payload.
 target=OUT/'substrate.f16.npy'
 if not target.exists():target.symlink_to(SRC/'substrate.f16.npy')
 paths=[SRC/'manifest.json',SRC/'draw_idx.npy',SRC/'substrate.f16.npy',OLD/'knn_indices.npy',OLD/'knn_dists.npy']
 result={'schema':'card018-data','complete':True,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'n':n,'dimensions':1536,'source_counts':counts,
         'radius':{'definition':'sqrt(mean(2*stored_cosine_distance))/train_p95','p95':p95,'floor':1e-6,'floor_count':floor_count,'floor_fraction':floor_count/n,
                   'radius_relative_error_p95':float(np.percentile(relative,95)),'radius_relative_error_max':float(relative.max()),'audit_zero_radius_rows':int((~nz).sum()),
                   'quantiles':np.percentile(r,[0,1,10,50,90,95,99,100]).tolist()},
         'neighbor_audit':{'panel_n':128,'recall15_mean':float(recalls.mean()),'recall15_min':float(recalls.min()),'contract':'normalized stored fullD features; CPU exhaustive blocked inner products; ID self excluded',
                           'historical_search':'inspected image_map_pipeline.knn exhaustively merged fp32 inner products over stored fp16 feature chunks; historical TF32 setting not independently reconstructed'},
         'PASS':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'inputs':{str(p):sha(p) for p in paths},
         'files':{name:sha(OUT/name) for name in ['substrate.f16.npy','edges-fixed15.npz','r_raw.npy','r_actual.npy','draw_ids.npy','draw_source.npy','radius-audit.npz','neighbor-audit.npz']},
         'builder_sha':sha(__file__),'cpu_wall_s':time.monotonic()-start}
 write(OUT/'manifest.json',result);write(OC/'card018-data-manifest.json',result);print(json.dumps({k:result[k] for k in ['PASS','radius','neighbor_audit','cpu_wall_s']},indent=2),flush=True)

if __name__=='__main__':main()
