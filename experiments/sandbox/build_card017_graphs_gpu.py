"""Exact FP32 GPU inner-product graphs, fixed inputs and bounded batched memory."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import time,json,hashlib
import numpy as np
import torch
import faiss
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from run_card017_arm import SUBD,OC,ARMS,sha,write
POOL=OC.parent/'card012-pool';SOURCES=['laion','coyo','commoncatalog-cc-by','megalith10m','cc12m']
BATCH=1024


def norm(x):
 x=np.asarray(x,dtype='f4');return x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)


@torch.inference_mode()
def search(index,q,k,self_rows=None,batch=BATCH):
 sims=[];inds=[]
 for at in range(0,len(q),batch):
  x=q[at:at+batch].to(index.device)
  s=x@index.T
  if self_rows is not None:s[torch.arange(len(x),device=index.device),self_rows[at:at+batch].to(index.device)]=-torch.inf
  values,where=torch.topk(s,k,dim=1,largest=True,sorted=True)
  sims.append(values.cpu().numpy());inds.append(where.cpu().numpy())
 return np.concatenate(sims),np.concatenate(inds)


def finish(arm):
 folder=SUBD/arm;knn=np.load(folder/'knn-nprobe0.npy');n=len(knn)
 src=np.repeat(np.arange(n,dtype='i4'),15);dst=knn.reshape(-1)
 assert (src!=dst).all() and dst.min()>=0 and dst.max()<n
 assert not (np.sort(knn,axis=1)[:,1:]==np.sort(knn,axis=1)[:,:-1]).any()
 graph=coo_matrix((np.ones(len(src),dtype='u1'),(src,dst)),shape=(n,n)).tocsr()
 nc,labels=connected_components(graph,directed=False);giant=float(np.bincount(labels).max()/n)
 keys=src.astype('i8')*n+dst;rev=dst.astype('i8')*n+src
 reciprocity=float(np.isin(rev,keys,assume_unique=True).mean())
 ss=np.load(folder/'draw_source.npy').astype(str);sl=np.array([SOURCES.index(x) for x in ss],dtype='i4')
 exposure=np.bincount(sl[src]*5+sl[dst],minlength=25).reshape(5,5)
 assert giant>=.99,f'giant {giant}<.99'
 np.savez(folder/'edges-fixed15.npz',sources=src,targets=dst,weights=np.ones(len(src),dtype='f4'),n_nodes=np.int64(n))
 return {'giant_fraction':giant,'components':int(nc),'reciprocity':reciprocity,'source_order':SOURCES,'source_exposure_counts':exposure.tolist(),
         'files':{n:sha(folder/n) for n in ['edges-fixed15.npz','knn-nprobe0.npy','knn-sim-nprobe0.npy','audit-nprobe0.npz','coverage-nprobe0.npz']}}


def canary():
 rng=np.random.default_rng(17017);x=norm(rng.normal(size=(4096,1536)).astype('f4'));q=x[:32]
 X=torch.from_numpy(x).cuda();Q=torch.from_numpy(q).cuda();selfids=torch.arange(32)
 s,i=search(X,Q,15,selfids,batch=7)
 exact=q.astype('f8')@x.astype('f8').T;exact[np.arange(32),np.arange(32)]=-np.inf
 truth=np.argsort(-exact,axis=1,kind='stable')[:,:15]
 checks={'fp64_reference_neighbors':np.array_equal(i,truth),'finite_nonself':np.isfinite(s).all() and not (i==np.arange(32)[:,None]).any()}
 s2,i2=search(X,Q,15,selfids,batch=32)
 checks['chunk_neighbor_identity']=np.array_equal(i,i2)
 # Duplicate-neighbor tie: self is excluded, the distinct equal vector remains.
 X[1]=X[0];_,ti=search(X,X[:1],1,torch.tensor([0]));checks['duplicate_not_self']=int(ti[0,0])==1
 checks={k:bool(v) for k,v in checks.items()}
 write(OC/'card017-exact-search-canary.json',{'PASS':all(checks.values()),'checks':checks,'source_sha':sha(__file__)})
 assert all(checks.values()),checks
 del X,Q;torch.cuda.empty_cache()


def main():
 torch.set_num_threads(4);faiss.omp_set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert torch.cuda.is_available();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 assert not (SUBD/'graph-manifest.json').exists(),'do not overwrite admitted graph'
 draw=json.loads((SUBD/'draw-manifest.json').read_text());assert draw['complete']
 runtime=json.loads((Path(__file__).resolve().parents[2]/'card017-runtime-sha.json').read_text())
 assert all(sha(Path(__file__).resolve().parents[2]/n)==h for n,h in runtime.items())
 canary()
 raw=np.load(POOL/'pool_X.f16.npy',mmap_mode='r');dev=np.load(SUBD/'support_dev_pool_rows.npy');devX=norm(raw[dev]);dev_panel=np.random.default_rng(170170).choice(len(dev),512,replace=False)
 diagnostics={};final={}
 for a in ARMS:
  t=time.monotonic();folder=SUBD/a
  assert all(sha(folder/n)==h for n,h in draw['arms'][a]['files'].items())
  x=norm(np.load(folder/'substrate.f16.npy'));X=torch.from_numpy(x).cuda();n=len(x)
  if a=='uniform':
   search(X,X[:1024],15,torch.arange(1024));torch.cuda.synchronize();pt=time.monotonic()
   search(X,X[:4096],15,torch.arange(4096));torch.cuda.synchronize();sec=time.monotonic()-pt
   estimate=sec/4096*(3*300000+3*len(devX))+120
   pre={'4096_query_s':sec,'expected_all_graph_and_support_s_plus_120':estimate,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'batch':BATCH,'precision':'FP32; TF32 and autocast off'}
   write(OC/'card017-exact-search-preflight.json',pre);print('PREFLIGHT',pre,flush=True)
   assert estimate<850,'exact GPU graph forecast exceeds fixed900s stage cap'
  sim,knn=search(X,X,15,torch.arange(n));np.save(folder/'knn-nprobe0.npy',knn.astype('i4'));np.save(folder/'knn-sim-nprobe0.npy',sim)
  ds,di=search(X,torch.from_numpy(devX),1)
  np.savez(folder/'coverage-nprobe0.npz',neighbor_local=di[:,0],similarity=ds[:,0],distance=np.sqrt(np.maximum(0,2-2*ds[:,0])))
  del X;torch.cuda.empty_cache()
  # Exhaustive independent CPU FAISS audit on the declared512-row panels.
  panel=np.random.default_rng(17017).choice(n,512,replace=False);exact=faiss.IndexFlatIP(1536);exact.add(x)
  es,ei=exact.search(x[panel],16);ok=ei!=panel[:,None];keep=ok&(np.cumsum(ok,axis=1)<=15);truth=ei[keep].reshape(512,15)
  recall=float((knn[panel,:,None]==truth[:,None,:]).any(2).mean())
  es,ei=exact.search(devX[dev_panel],1);cov=float((di[dev_panel,0]==ei[:,0]).mean())
  np.savez(folder/'audit-nprobe0.npz',train_panel_local=panel,train_truth15=truth,train_ann15=knn[panel],dev_panel_local=dev_panel,dev_exact_i=ei[:,0],dev_exact_s=es[:,0],dev_ann_i=di[dev_panel,0],dev_ann_s=ds[dev_panel,0])
  diag={'arm':a,'nprobe':None,'backend':'torch_fp32_exact','train_recall15':recall,'support_recall1':cov,'numerical_PASS':recall>=.98 and cov>=.98,
        'duplicate_first_neighbor_fraction':float((sim[:,0]>=1-1e-7).mean()),'stage_wall_s':time.monotonic()-t}
  write(folder/'graph-audit-exact.json',diag);assert diag['numerical_PASS'],diag
  diagnostics[a]=diag;final[a]=finish(a);print(a,'EXACT_DONE',diag,flush=True)
  del exact,x,knn,sim
 assert torch.cuda.max_memory_allocated()/2**30<30
 result={'schema':'card017-graph-manifest','complete':True,'backend':'torch_fp32_exact','common_nprobe':0,
         'compatibility_note':'file suffix nprobe0 denotes exact exhaustive search, not an IVF configuration','nlist':None,
         'metric':'IP on unit-normalized stored full1536D fp16 features cast toFP32; FP32 matmul, TF32/autocast off',
         'neighbors':'15 directed real neighbors, self excluded before topk, unique indices; binary weights',
         'draw_manifest_sha':sha(SUBD/'draw-manifest.json'),'audit':{'0':diagnostics},'graphs':final,'builder_sha':sha(__file__),
         'exclusive_stage_wall_s':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
 write(SUBD/'graph-manifest.json',result);write(OC/'card017-graph-manifest.json',result);print('EXACT_GRAPHS_ADMITTED',result['exclusive_stage_wall_s'],flush=True)

if __name__=='__main__':main()
