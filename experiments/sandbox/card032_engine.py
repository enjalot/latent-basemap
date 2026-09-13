"""Isolated streaming/top-neighbor/checkpoint primitives; no global GPU side effects."""
from pathlib import Path
import json,os,hashlib
import numpy as np
import torch

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def initial(nq,tiers,device):return {n:(torch.full((nq,16),-torch.inf,device=device),torch.full((nq,16),-1,dtype=torch.long,device=device)) for n in tiers}
def update(state,similarity,ids,ranks):
 for n,(scores,oldids) in state.items():
  take=torch.nonzero((ranks>=0)&(ranks<n)).flatten()
  if not len(take):continue
  new,ix=torch.topk(similarity[:,take],min(16,len(take)),dim=1);newids=ids[take][ix];joined=torch.cat([scores,new],1);js=torch.cat([oldids,newids],1);best,order=torch.topk(joined,16,dim=1);state[n]=(best,torch.gather(js,1,order))
 return state

def save_checkpoint(directory,cursor,state,outputs,identity,history):
 directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
 for x in outputs.values():x.flush()
 p=directory/f'state-{cursor}.npz';tmp=p.with_suffix('.tmp');arrays={}
 for n,(s,i) in state.items():arrays[f'scores_{n}']=s.cpu().numpy();arrays[f'ids_{n}']=i.cpu().numpy()
 with tmp.open('wb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);atomic(directory/'progress.json',{'cursor':cursor,'identity':identity,'state':p.name,'state_sha':sha(p),'history':history})
 # Keep predecessor as well as current checkpoint; a kill before progress replace cannot invalidate the predecessor.
 states=sorted(directory.glob('state-*.npz'),key=lambda p:int(p.stem.split('-')[1]))
 for old in states[:-2]:old.unlink()
def load_checkpoint(directory,identity,outputs,device):
 directory=Path(directory);p=json.loads((directory/'progress.json').read_text())
 if p['identity']!=identity:raise ValueError('card032 admission-identity mismatch')
 if sha(directory/p['state'])!=p['state_sha']:raise ValueError('card032 state hash mismatch')
 for h in p['history']:
  lo,hi=h['lo'],h['hi']
  for a,m in outputs.items():
   if hashlib.sha256(np.ascontiguousarray(m[lo:hi]).tobytes()).hexdigest()!=h['sha'][a]:raise ValueError('card032 completed-output hash mismatch')
 z=np.load(directory/p['state']);ns=[int(n.split('_')[1]) for n in z.files if n.startswith('scores_')];state={n:(torch.from_numpy(z[f'scores_{n}']).to(device),torch.from_numpy(z[f'ids_{n}']).to(device)) for n in ns};return p['cursor'],state,p['history']
