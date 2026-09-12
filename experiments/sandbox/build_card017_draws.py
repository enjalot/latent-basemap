"""CPU-only frozen source-quota sampling on full-D features via a PCA plaid cover."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
from pathlib import Path
import sys,json,hashlib,time,datetime as dt
import numpy as np
from sklearn.decomposition import PCA
from card017_sketch import cover,sample_cover,canary
SB=Path('/data/latent-basemap/sandbox'); OC=SB/'overseer-codex'; POOL=SB/'card012-pool'; D16=SB/'card016-data'
OUT=Path('/data/latent-basemap/substrates/card017-support'); OUT.mkdir(parents=True,exist_ok=True)
SOURCES=['laion','coyo','commoncatalog-cc-by','megalith10m','cc12m']; ARMS=['uniform','geometric','hybrid']; SEED=17017


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def write(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)


def norm(x):
 x=np.asarray(x,dtype='f4');return x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)


def main():
 start=time.monotonic(); assert not (OUT/'draw-manifest.json').exists(),'do not overwrite admitted draw'
 proof=canary(); assert all(proof.values()),proof
 write(OC/'card017-sketch-canary.json',{'PASS':True,'checks':proof,'source':{n:sha(Path(__file__).parent/n) for n in ['card017_sketch.py','build_card017_draws.py']}})
 parent=json.loads((D16/'manifest.json').read_text()); assert parent['complete'] and parent['audit']['PASS']
 assert all(sha(D16/n)==h for n,h in parent['files'].items())
 pm=json.loads((OC/'card012-pool-manifest.json').read_text()); assert sha(OC/'card012-pool-manifest.json')==parent['pool_manifest_sha']
 X=np.load(POOL/'pool_X.f16.npy',mmap_mode='r'); ids=np.load(POOL/'pool_ids.npy'); src=np.load(POOL/'pool_source.npy',allow_pickle=True).astype(str)
 fit=np.load(D16/'fit_rows.npy'); dev=np.load(D16/'dev_rows.npy')
 # Verify actual full feature contents, not only inherited file names.
 h=hashlib.sha256()
 for i in range(0,len(X),4096): h.update(np.ascontiguousarray(X[i:i+4096]).tobytes())
 assert h.hexdigest()[:16]==pm['X_sha']
 assert len(fit)==950000 and len(dev)==50000 and not np.isin(fit,dev).any()
 rows={s:fit[src[fit]==s] for s in SOURCES}; assert all(len(r)==190000 for r in rows.values())
 rng=np.random.default_rng(SEED)
 pca_rows=np.concatenate([rng.choice(rows[s],10000,replace=False) for s in SOURCES])
 np.save(OUT/'pca_fit_pool_rows.npy',pca_rows); np.save(OUT/'support_dev_pool_rows.npy',dev)
 pca=PCA(n_components=100,svd_solver='randomized',iterated_power=5,random_state=SEED)
 pca.fit(norm(X[pca_rows])); np.savez(OUT/'pca.npz',mean=pca.mean_,components=pca.components_,explained_variance_ratio=pca.explained_variance_ratio_)
 print('PCA fitted; variance',float(pca.explained_variance_ratio_.sum()),flush=True)
 draws={a:[] for a in ARMS}; diag={}
 for j,s in enumerate(SOURCES):
  pr=rows[s]; reduced=[]
  for at in range(0,len(pr),8192): reduced.append(pca.transform(norm(X[pr[at:at+8192]])))
  reduced=np.concatenate(reduced); np.save(OUT/f'pca-{s}.npy',reduced)
  c=cover(reduced,60000); del reduced
  np.savez(OUT/f'cover-{s}.npz',pool_rows=pr,labels=c['labels'],counts=c['counts'],cells=c['cells'])
  u=np.random.default_rng(SEED+100+j).choice(len(pr),60000,replace=False)
  g=sample_cover(c['labels'],60000,SEED+200+j)
  hg=sample_cover(c['labels'],30000,SEED+300+j)
  remaining=np.setdiff1d(np.arange(len(pr)),hg,assume_unique=True)
  hu=np.random.default_rng(SEED+400+j).choice(remaining,30000,replace=False)
  for a,ix in [('uniform',u),('geometric',g),('hybrid',np.r_[hg,hu])]:draws[a].append(pr[ix])
  diag[s]={'occupied_boxes':len(c['counts']),'unit':c['unit'],'scale':c['scale'],'search_trace':c['trace'],
           'occupancy_quantiles':np.percentile(c['counts'],[0,25,50,75,90,99,100]).tolist(),
           'singleton_boxes':int((c['counts']==1).sum()),'draw_occupied_boxes':{a:int(len(np.unique(c['labels'][ix]))) for a,ix in [('uniform',u),('geometric',g),('hybrid',np.r_[hg,hu])]}}
  print(s,diag[s]['occupied_boxes'],diag[s]['draw_occupied_boxes'],flush=True)
 arm_info={}
 for a in ARMS:
  r=np.concatenate(draws[a]); r=r[np.argsort(ids[r],kind='stable')]
  assert len(r)==len(np.unique(r))==300000 and not np.isin(r,dev).any()
  ad=OUT/a;ad.mkdir(exist_ok=True)
  np.save(ad/'pool_rows.npy',r);np.save(ad/'draw_ids.npy',ids[r]);np.save(ad/'draw_source.npy',src[r])
  dest=np.lib.format.open_memmap(ad/'substrate.f16.npy',mode='w+',dtype='f2',shape=(len(r),1536))
  for at in range(0,len(r),8192):dest[at:at+8192]=norm(X[r[at:at+8192]]).astype('f2')
  dest.flush();del dest
  arm_info[a]={'files':{n:sha(ad/n) for n in ['pool_rows.npy','draw_ids.npy','draw_source.npy','substrate.f16.npy']},'n':300000,'sources':{s:int((src[r]==s).sum()) for s in SOURCES}}
 result={'schema':'card017-draw-manifest','complete':True,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'parent_data_sha':sha(D16/'manifest.json'),
         'seed':SEED,'pca':{'fit_rows':50000,'dimensions':100,'whiten':False,'variance_fraction':float(pca.explained_variance_ratio_.sum()),'files':{n:sha(OUT/n) for n in ['pca.npz','pca_fit_pool_rows.npy','support_dev_pool_rows.npy']}},
         'arms':arm_info,'cover':diag,'source':{n:sha(Path(__file__).parent/n) for n in ['card017_sketch.py','build_card017_draws.py']},'cpu_wall_s':time.monotonic()-start}
 write(OUT/'draw-manifest.json',result);write(OC/'card017-draw-manifest.json',result)
 print('DONE',result['cpu_wall_s'],flush=True)


if __name__=='__main__': main()
