"""1536-D real core resume twins and radius=1 control; tiny cyclic graph, no quality claim."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc
import numpy as np,torch
import card053_common as C
from card053_fit import fit

def same(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 start=time.monotonic();C.source_check();C.input_check();C.graph_check();checks=[];hashes={};streams={};keys=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale']
 with tempfile.TemporaryDirectory(dir=str(C.R.parent)) as td:
  td=Path(td);n=512;X=np.asarray(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4');graph=td/'edges.npz';src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.array([(i+j)%n for i in range(n) for j in range(1,16)],dtype='i4');np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n)
  for arm in C.ARMS:
   rd=None
   if C.RAD[arm]:rd=td/(arm+'.npy');np.save(rd,np.load(C.D/C.RAD[arm])[:n])
   kw={'X':X,'graph':graph,'radius_path':rd,'checkpoints':[2,4,7,9,18]}
   p,full,report=fit(arm,18,td/arm,**kw);hashes[arm]=report['state_sha'];probe=torch.load(td/arm/'ckpts/ckpt-step2.pt',map_location='cpu',weights_only=False);streams[arm]={k:probe[k] for k in ['loader_gen','loader_perm','loader_rank_of_node','loader_node_at_rank']};del probe;del p;gc.collect();torch.cuda.empty_cache()
   candidates=[]
   for path in (td/arm/'ckpts').glob('ckpt-step*.pt'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if 0<ck['global_step']<18 and 0<ck['loader_pos_idx']<len(ck['loader_perm']):candidates.append((ck['global_step'],path))
   assert candidates,'no genuine mid-epoch checkpoint';mid=min(candidates)[1];p,res,_=fit(arm,18,td/(arm+'-mid'),resume=mid,**kw);assert all(same(full[k],res[k]) for k in keys),'mid-epoch state differs';checks.append(arm+' genuine mid-epoch full-state twin');del p,res;gc.collect();torch.cuda.empty_cache()
   eps=[]
   for path in (td/arm/'ckpts').glob('ckpt-epoch*.pt'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if 0<ck['global_step']<18 and ck['epoch']<full['epoch']:eps.append((ck['global_step'],path))
   assert eps,'no checkpoint before later epoch';ep=min(eps)[1];p,res,_=fit(arm,18,td/(arm+'-epoch'),resume=ep,**kw);assert all(same(full[k],res[k]) for k in keys),'epoch state differs';checks.append(arm+' epoch resume across later boundary');del p,res;gc.collect();torch.cuda.empty_cache()
   for tag,override in [('wrong_arm',{'arm':'wrong'}),('wrong_radius',{'radii_sha':'0'*64}),('wrong_input',{'input_manifest_sha':'0'*64}),('wrong_LR',{'lr':.001})]:
    try:fit(arm,18,td/(arm+'-'+tag),resume=mid,ident_override=override,**kw)
    except AssertionError as exc:assert 'card053 admission-identity mismatch' in str(exc);checks.append(arm+' '+tag+' rejects before restore')
    else:raise AssertionError('wrong identity accepted')
  assert all(same(streams[C.ARMS[0]][k],streams[a][k]) for a in C.ARMS for k in streams[a]);checks.append('attempted sampler streams at step2 equal')
  assert len(set(hashes.values()))==3,'radius interventions not exposed';checks.append('three radius arms actually diverge')
 result={'PASS':True,'checks':checks,'n_checks':len(checks),'endpoint_hashes':hashes,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'scope':'Actual1536-D device core fit on512-row cyclic15 graph,18 successful updates. State twins, default-off control and specific identity failures; no quality claim.'};C.write(C.O/'card053-device-canary.json',result);print(result,flush=True)
if __name__=='__main__':main()
