"""Card088 graph contracts and exact tiled extra-neighbor search; no import-time work."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
R=Path(__file__).resolve().parents[2];O=R.parent/'overseer-codex'
D=Path('/data/latent-basemap/substrates/card088-positive-support')
OLD=R.parent/'dino-arrival-t0';INPUT=Path('/data/latent-basemap/substrates/card018-scale2m')
N=2000000;DIM=1536;K=45;TOL=4e-6

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def weights(arm,n):
 assert arm in ('original15','mixture'),'unknown support arm'
 w=np.empty((n,60),dtype='f4');w[:,:15]=6 if arm=='original15' else 3;w[:,15:]=0 if arm=='original15' else 1;return w

def validate_rows(old,extra,n):
 old=np.asarray(old);extra=np.asarray(extra)
 assert old.shape==(n,15) and extra.shape==(n,45),'support shape'
 assert np.issubdtype(old.dtype,np.integer) and np.issubdtype(extra.dtype,np.integer),'support integer IDs'
 both=np.concatenate([old,extra],axis=1)
 assert (both>=0).all() and (both<n).all(),'support bounds'
 assert not (both==np.arange(n)[:,None]).any(),'support self'
 assert (np.diff(np.sort(both,axis=1),axis=1)>0).all(),'support duplicates/original overlap'
 return both

def deterministic_top(scores,ids,k):
 """Exact score ties resolved by smaller ID; assumes bounded candidate width."""
 ids=ids.expand_as(scores) if ids.ndim==1 else ids
 byid=torch.argsort(ids,dim=1,stable=True)
 si=torch.gather(scores,1,byid);ii=torch.gather(ids,1,byid)
 byscore=torch.argsort(si,dim=1,descending=True,stable=True)[:,:k]
 return torch.gather(si,1,byscore),torch.gather(ii,1,byscore)

def block_top(sim,offset,k):
 """Fast topk with explicit complete boundary-tie resolution,never epsilon jitter."""
 width=sim.shape[1];kk=min(k,width)
 v,idx=torch.topk(sim,min(k+1,width),dim=1,sorted=True);idx=idx+offset
 sc,ii=deterministic_top(v[:,:kk],idx[:,:kk],kk)
 tied=torch.nonzero(v[:,kk-1]==v[:,kk],as_tuple=False).flatten() if width>k else torch.empty(0,dtype=torch.long,device=sim.device)
 for row in tied.tolist():
  cut=v[row,kk-1];above=torch.nonzero(sim[row]>cut).flatten();equal=torch.nonzero(sim[row]==cut).flatten()
  # nonzero yields ascending local IDs; include entire tie only to choose lowest.
  take=torch.cat([above,equal[:kk-len(above)]])
  ss,jj=deterministic_top(sim[row,take][None],(take+offset)[None],kk);sc[row]=ss[0];ii[row]=jj[0]
 return sc,ii,int(len(tied))

def search(db,rows,old,*,block=65536,k=45,dtype=torch.float32,check=lambda:None):
 """Exhaustive same-support search outside exact original15 AND self.

Original list need not equal current numerical top15. It is NEVER regenerated.
"""
 rows=torch.as_tensor(rows,device=db.device,dtype=torch.long);old=torch.as_tensor(old,device=db.device,dtype=torch.long)
 assert old.shape==(len(rows),15)
 query=torch.nn.functional.normalize(db[rows].to(dtype),dim=1)
 assert bool(torch.isfinite(query).all()) and bool((torch.linalg.vector_norm(query,dim=1)>0).all()),'invalid query features'
 best=torch.empty((len(rows),0),device=db.device,dtype=dtype);bid=torch.empty((len(rows),0),device=db.device,dtype=torch.long);tie_rows=0
 for lo in range(0,len(db),block):
  hi=min(lo+block,len(db));ref=torch.nn.functional.normalize(db[lo:hi].to(dtype),dim=1)
  assert bool(torch.isfinite(ref).all()) and bool((torch.linalg.vector_norm(ref,dim=1)>0).all()),'invalid reference features'
  sim=query@ref.T;exclude=torch.cat([rows[:,None],old],dim=1);mask=(exclude>=lo)&(exclude<hi);rr,cc=torch.nonzero(mask,as_tuple=True);sim[rr,exclude[rr,cc]-lo]=-torch.inf
  sc,ii,t=block_top(sim,lo,k);tie_rows+=t
  best,bid=deterministic_top(torch.cat([best,sc],1),torch.cat([bid,ii],1),min(k,best.shape[1]+sc.shape[1]));check()
 assert bool(torch.isfinite(best).all()),'insufficient eligible candidates'
 assert not bool((bid[:,:,None]==old[:,None,:]).any()) and not bool((bid==rows[:,None]).any()),'excluded candidate returned'
 return bid,best,{'block_boundary_tied_query_count':tie_rows}

def validate_bundle(directory=D):
 directory=Path(directory);m=read(directory/'manifest.json')
 assert m['PASS'] is True and m['n']==N and m['k_extra']==45,'card088 graph manifest schema'
 assert m['runtime_sha']==sha(R/'card088-runtime-sha.json'),'card088 graph runtime identity'
 assert m['graph_builder_sha']==sha(R/'experiments/sandbox/build_card088_graph.py'),'card088 graph builder identity'
 assert all(sha(directory/p)==h for p,h in m['files'].items()),'card088 data artifact hash mismatch'
 assert all(sha(p)==h for p,h in m['inputs'].items()),'card088 data input hash mismatch'
 return m
