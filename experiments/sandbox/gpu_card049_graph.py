"""Exhaustive FP32 CLIP training graph and query truth; fixed FP64 audit before admission."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,time,json,resource
import numpy as np
import torch
import torch.nn.functional as F
import card049_common as C

def top15(x,ref,self_exclude=False):
 ids=np.empty((len(x),15),'i4');scores=np.empty((len(x),15),'f4')
 for lo in range(0,len(x),512):
  hi=min(len(x),lo+512);sim=x[lo:hi]@ref.T
  if self_exclude:sim[torch.arange(hi-lo,device='cuda'),torch.arange(lo,hi,device='cuda')]=-torch.inf
  v,j=torch.topk(sim,15,dim=1,largest=True,sorted=True);ids[lo:hi]=j.cpu().numpy();scores[lo:hi]=v.cpu().numpy();del sim
  if lo%32768==0:print('neighbors',lo,len(x),flush=True)
 return ids,scores

def audit(bank,query,ids,scores,chosen,self_exclude=False):
 # Independent CPU FP64 blocked exhaustive calculation; no GPU topk reuse.
 out=[];refs=np.asarray(bank,dtype='f8');refs/=np.linalg.norm(refs,axis=1,keepdims=True)
 for start in range(0,len(chosen),8):
  ix=chosen[start:start+8];q=np.asarray(query[ix],dtype='f8');q/=np.linalg.norm(q,axis=1,keepdims=True);sim=q@refs.T
  if self_exclude:sim[np.arange(len(ix)),ix]=-np.inf
  for j,row in enumerate(ix):
   exact=np.argpartition(sim[j],-15)[-15:];found=ids[row];boundary=float(sim[j,exact].min());got=sim[j,found];missing=np.setdiff1d(exact,found);material=int(np.count_nonzero(got<boundary-4e-6));gap=max(0.,boundary-float(got.min()));entry={'row':int(row),'recall15':len(np.intersect1d(found,exact))/15,'material_misses':material,'max_boundary_shortfall':gap,'score_max_abs':float(np.max(np.abs(got-scores[row])))};assert material==0,entry;assert entry['score_max_abs']<4e-6,entry;out.append(entry)
 return out

def main():
 start=time.monotonic();C.source_check();inputs=C.input_check();D=C.D
 if (D/'graph-manifest.json').exists():C.graph_check();print('existing graph validated');return
 assert torch.cuda.is_available();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 train=np.load(D/'train.f16.npy',mmap_mode='r');ref=np.load(D/'reference.f16.npy',mmap_mode='r');query=np.load(D/'query.f16.npy',mmap_mode='r');x=F.normalize(torch.tensor(np.asarray(train),device='cuda').float(),dim=1);xr=F.normalize(torch.tensor(np.asarray(ref),device='cuda').float(),dim=1);q=F.normalize(torch.tensor(np.asarray(query),device='cuda').float(),dim=1)
 ki,ks=top15(x,x,True);qi,qs=top15(q,xr);assert not (ki==np.arange(C.N)[:,None]).any();assert all(len(np.unique(j))==15 for j in ki);assert all(len(np.unique(j))==15 for j in qi)
 sel=np.load(D/'neighbor-audit-selection.npz');ta=audit(train,train,ki,ks,sel['train_local'],True);qa=audit(ref,query,qi,qs,sel['query_local']);np.savez(D/'neighbor-audit.npz',train_indices=ki[sel['train_local']],train_scores=ks[sel['train_local']],query_indices=qi[sel['query_local']],query_scores=qs[sel['query_local']],train_local=sel['train_local'],query_local=sel['query_local'])
 d2=np.maximum(0.,2.-2.*ks.astype('f8'));radius=np.sqrt(d2.mean(1));p95=float(np.percentile(radius,95));assert p95>0 and np.isfinite(radius).all();floor=float(np.mean(radius/p95<1e-6));assert floor<=.1;half=np.sqrt(np.maximum(radius/p95,1e-6)).astype('f4');shuffled=half.copy();groups=np.load(D/'train-source.npy');rng=np.random.default_rng(49049)
 for g in np.unique(groups):
  ix=np.flatnonzero(groups==g);shuffled[ix]=half[rng.permutation(ix)];assert np.array_equal(np.sort(half[ix]),np.sort(shuffled[ix]))
 assert np.isfinite(half).all() and (half>0).all() and not np.array_equal(half,shuffled)
 np.save(D/'train-knn-indices.npy',ki);np.save(D/'train-knn-scores.npy',ks);np.save(D/'truth.npy',qi);np.save(D/'truth-scores.npy',qs);np.save(D/'encoder-radius.npy',np.sqrt(np.maximum(0.,2.-2.*qs.astype('f8')).mean(1)));np.save(D/'train-radius.npy',radius);np.save(D/'r-half.npy',half);np.save(D/'r-shuffled-half.npy',shuffled);np.savez(D/'edges-fixed15.npz',sources=np.repeat(np.arange(C.N,dtype='i4'),15),targets=ki.reshape(-1),weights=np.ones(C.N*15,'f4'),n_nodes=C.N)
 names=['train-knn-indices.npy','train-knn-scores.npy','truth.npy','truth-scores.npy','encoder-radius.npy','train-radius.npy','r-half.npy','r-shuffled-half.npy','edges-fixed15.npz','neighbor-audit.npz'];free,total=torch.cuda.mem_get_info();vram=(total-free)/2**30;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert vram<30 and rss<32768
 result={'PASS':True,'status':'GRAPH_TRUTH_READY','input_manifest_sha':C.sha(D/'inputs-manifest.json'),'files':{n:C.sha(D/n) for n in names},'training_audit':ta,'query_audit':qa,'train_p95':p95,'floor_fraction':floor,'source_radius_quantiles':{str(g):np.percentile(half[groups==g],[0,25,50,75,100]).tolist() for g in np.unique(groups)},'search':'Exhaustive GPU FP32 normalized stored inputs, TF32 disabled. Fixed18-query/128-training CPU FP64 audits; material shortfall tolerance4e-6. High-dimensional ties use persisted topk membership, not averaged boundary membership.','global_vram_GiB':vram,'max_rss_MiB':rss,'wall_s':time.monotonic()-start,'builder_sha':C.sha(Path(__file__)),'runtime_sha':C.source_check()};C.write(D/'graph-manifest.json',result);print(json.dumps({k:result[k] for k in ['PASS','train_p95','floor_fraction','wall_s','global_vram_GiB']},indent=2))
if __name__=='__main__':main()
