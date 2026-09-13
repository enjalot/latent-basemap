"""Independent training-ID-only reciprocity/union-find audit. CPU <=8GiB/2threads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,time,resource
import numpy as np
from numba import njit
O=Path('/data/latent-basemap/sandbox/overseer-codex');D=O/'reciprocity-readiness-20260913'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
@njit
def components(a,mask):
 n=len(a);parent=np.arange(n);size=np.ones(n,np.int64)
 for i in range(n):
  for c in range(a.shape[1]):
   if not mask[i,c]:continue
   x=i;y=a[i,c]
   while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
   while y!=parent[y]:parent[y]=parent[parent[y]];y=parent[y]
   if x!=y:
    if size[x]<size[y]:x,y=y,x
    parent[y]=x;size[x]+=size[y]
 return size[parent==np.arange(n)]
def summary(s,n):return {'components':len(s),'largest_component':int(s.max()),'largest_fraction':float(s.max()/n),'singleton_components':int((s==1).sum()),'vertices_in_components_lt16':int(s[s<16].sum()),'largest10':np.sort(s)[-10:][::-1].tolist()}
def main():
 start=time.monotonic();r=json.loads((D/'result.json').read_text());assert sha(O/'diagnose_training_reciprocity.py')==r['source_sha']
 assert all(sha(p)==h for p,h in r['inputs'].items());assert all(sha(D/p)==h for p,h in r['outputs'].items())
 a=np.load('/data/latent-basemap/sandbox/dino-arrival-t0/knn_indices.npy',mmap_mode='r');mask=np.load(D/'mutual-mask.npy',mmap_mode='r');degree=np.load(D/'mutual-degree.npy');n,k=a.shape
 # Independent scalar-key reverse lookup, no neighbor-of-neighbor broadcast.
 keys=(np.arange(n,dtype='i8')[:,None]*n+a).ravel();keys.sort()
 for lo in range(0,n,8192):
  hi=min(lo+8192,n);reverse=a[lo:hi].astype('i8')*n+np.arange(lo,hi)[:,None];idx=np.searchsorted(keys,reverse);good=idx<len(keys);got=good&(keys[np.minimum(idx,len(keys)-1)]==reverse);assert np.array_equal(got,mask[lo:hi]),'reverse-key mismatch'
 del keys
 assert np.array_equal(mask.sum(1),degree);assert np.bincount(degree,minlength=16).tolist()==r['mutual_degree_histogram'];assert int(mask.sum())==r['reciprocal_directed_edges'];assert int((degree==0).sum())==r['zero_mutual_nodes']
 observed={}
 for name,m in [('original_weak_components',np.ones(a.shape,bool)),('strict_mutual_components',mask)]:
  observed[name]=summary(components(a,m),n);assert observed[name]==r[name],name
 sources=np.load('/data/latent-basemap/substrates/card018-scale2m/draw_source.npy',allow_pickle=True).astype(str);radius=np.load('/data/latent-basemap/substrates/card018-scale2m/r_raw.npy');dec=np.searchsorted(np.percentile(radius,np.arange(10,100,10)),radius,side='right');indegree=np.bincount(a.ravel(),minlength=n)
 for group,selectors in [('source_strata',{s:sources==s for s in np.unique(sources)}),('training_radius_deciles',{str(i+1):dec==i for i in range(10)})]:
  for name,sel in selectors.items():
   v=r[group][name];actual={'n':int(sel.sum()),'reciprocal_directed_edge_fraction':degree[sel].sum()/(15*sel.sum()),'zero_mutual_fraction':(degree[sel]==0).mean(),'mean_mutual_degree':degree[sel].mean(),'mean_indegree':indegree[sel].mean(),'indegree_p99':np.quantile(indegree[sel],.99)};assert all(np.isclose(actual[x],v[x],rtol=0,atol=1e-12) for x in actual)
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert rss<8192
 out={'PASS':True,'source_sha':sha(__file__),'root_result_sha':sha(D/'result.json'),'input_bindings':r['inputs'],'output_bindings':r['outputs'],'full_reverse_key_edges_checked':n*k,'independent_union_find_components':observed,'all_source_radius_summaries_match':True,'cpu_wall_s':time.monotonic()-start,'max_rss_MiB':rss,'scope':'Full existing training neighbor IDs/masks/source/radius only; independent sorted reverse keys and union-find; no features,queries,models,GPU. Connectivity is descriptive, not semantic or quality evidence.'};p=O/'card089-runner-diagnostic-audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
