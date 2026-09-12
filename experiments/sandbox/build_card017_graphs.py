"""CPU full-D fixed15 graphs and support diagnostics; strict common ANN fidelity."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
from pathlib import Path
import json,time
import numpy as np
import faiss
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from build_card017_draws import OUT,POOL,OC,ARMS,SOURCES,sha,write,norm
faiss.omp_set_num_threads(4)


def remove_self(I,S):
 n=len(I); ok=(I>=0)&(I!=np.arange(n)[:,None]); keep=ok&(np.cumsum(ok,axis=1)<=15)
 assert (keep.sum(1)==15).all()
 ids=I[keep].reshape(n,15).astype('i4'); sim=S[keep].reshape(n,15)
 assert not (np.sort(ids,axis=1)[:,1:]==np.sort(ids,axis=1)[:,:-1]).any()
 return ids,sim


def build_arm(arm,nprobe,devX,dev_ids):
 start=time.monotonic(); folder=OUT/arm
 X=norm(np.load(folder/'substrate.f16.npy')); n,d=X.shape
 p=folder/'ivf.index'
 if p.exists(): index=faiss.read_index(str(p))
 else:
  index=faiss.IndexIVFFlat(faiss.IndexFlatIP(d),d,2048,faiss.METRIC_INNER_PRODUCT)
  index.cp.seed=42;index.cp.max_points_per_centroid=39;index.train(X);index.add(X);faiss.write_index(index,str(p))
 assert index.ntotal==n and index.d==1536;index.nprobe=nprobe
 print(arm,'graph search nprobe',nprobe,flush=True)
 S,I=index.search(X,16); knn,sim=remove_self(I,S);del S,I
 np.save(folder/f'knn-nprobe{nprobe}.npy',knn);np.save(folder/f'knn-sim-nprobe{nprobe}.npy',sim)
 panel=np.random.default_rng(17017).choice(n,512,replace=False)
 exact=faiss.IndexFlatIP(d);exact.add(X)
 ES,EI=exact.search(X[panel],16)
 eok=EI!=panel[:,None];ekeep=eok&(np.cumsum(eok,axis=1)<=15)
 truth=EI[ekeep].reshape(512,15)
 hits=(knn[panel,:,None]==truth[:,None,:]).any(2).mean()
 ds,di=index.search(devX,1); exact_s,exact_i=exact.search(devX[dev_ids],1)
 coverage_recall=float((di[dev_ids,0]==exact_i[:,0]).mean())
 np.savez(folder/f'audit-nprobe{nprobe}.npz',train_panel_local=panel,train_truth15=truth,train_ann15=knn[panel],
          dev_panel_local=dev_ids,dev_exact_i=exact_i[:,0],dev_exact_s=exact_s[:,0],dev_ann_i=di[dev_ids,0],dev_ann_s=ds[dev_ids,0])
 np.savez(folder/f'coverage-nprobe{nprobe}.npz',neighbor_local=di[:,0],similarity=ds[:,0],distance=np.sqrt(np.maximum(0,2-2*ds[:,0])))
 diagnostics={'arm':arm,'nprobe':nprobe,'train_recall15':float(hits),'support_recall1':coverage_recall,'cpu_wall_s':time.monotonic()-start,
              'numerical_PASS':bool(hits>=.98 and coverage_recall>=.98),'duplicate_first_neighbor_fraction':float((sim[:,0]>=1-1e-7).mean())}
 write(folder/f'graph-audit-nprobe{nprobe}.json',diagnostics)
 print(diagnostics,flush=True);return diagnostics


def finish(arm,nprobe):
 folder=OUT/arm; knn=np.load(folder/f'knn-nprobe{nprobe}.npy');n=len(knn)
 src=np.repeat(np.arange(n,dtype='i4'),15);dst=knn.reshape(-1)
 assert (src!=dst).all() and dst.min()>=0 and dst.max()<n
 graph=coo_matrix((np.ones(len(src),dtype='u1'),(src,dst)),shape=(n,n)).tocsr()
 nc,labels=connected_components(graph,directed=False); giant=float(np.bincount(labels).max()/n)
 keys=src.astype('i8')*n+dst; rev=dst.astype('i8')*n+src
 reciprocity=float(np.isin(rev,keys,assume_unique=True).mean())
 s=np.load(folder/'draw_source.npy').astype(str);label=np.array([SOURCES.index(x) for x in s],dtype='i4')
 exposure=np.bincount(label[src]*5+label[dst],minlength=25).reshape(5,5)
 assert giant>=.99,f'{arm} giant {giant} <.99'
 np.savez(folder/'edges-fixed15.npz',sources=src,targets=dst,weights=np.ones(len(src),dtype='f4'),n_nodes=np.int64(n))
 return {'giant_fraction':giant,'components':int(nc),'reciprocity':reciprocity,'source_order':SOURCES,'source_exposure_counts':exposure.tolist(),
         'files':{name:sha(folder/name) for name in ['edges-fixed15.npz',f'knn-nprobe{nprobe}.npy',f'knn-sim-nprobe{nprobe}.npy',f'audit-nprobe{nprobe}.npz',f'coverage-nprobe{nprobe}.npz']}}


def main():
 start=time.monotonic(); assert not (OUT/'graph-manifest.json').exists(),'do not overwrite admitted graphs'
 draw=json.loads((OUT/'draw-manifest.json').read_text());assert draw['complete']
 for a in ARMS: assert all(sha(OUT/a/n)==h for n,h in draw['arms'][a]['files'].items())
 raw=np.load(POOL/'pool_X.f16.npy',mmap_mode='r'); dev=np.load(OUT/'support_dev_pool_rows.npy'); devX=norm(raw[dev])
 dev_panel=np.random.default_rng(170170).choice(len(dev),512,replace=False)
 all_diagnostics={}; admitted=None
 for probe in [96,192]:
  diag={a:build_arm(a,probe,devX,dev_panel) for a in ARMS};all_diagnostics[str(probe)]=diag
  if all(v['numerical_PASS'] for v in diag.values()):admitted=probe;break
 assert admitted is not None,'declared common nprobe repair did not reach search fidelity'
 final={a:finish(a,admitted) for a in ARMS}
 result={'schema':'card017-graph-manifest','complete':True,'common_nprobe':admitted,'nlist':2048,'clustering_max_points_per_centroid':39,'clustering_seed':42,'metric':'IP on unit-normalized stored full1536D fp16 features cast toFP32',
         'neighbors':'15 directed real neighbors, no self, unique indices; binary weights','draw_manifest_sha':sha(OUT/'draw-manifest.json'),
         'audit':all_diagnostics,'graphs':final,'builder_sha':sha(__file__),'cpu_wall_s':time.monotonic()-start}
 write(OUT/'graph-manifest.json',result);write(OC/'card017-graph-manifest.json',result)
 print('GRAPHS_ADMITTED',admitted,result['cpu_wall_s'],flush=True)


if __name__=='__main__':main()
