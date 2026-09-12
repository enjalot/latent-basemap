"""Actual replay+radius 1536D continuation/resume controls. No heldout queries."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc
import numpy as np,torch
import card052_common as C
from card052_fit import fit

def same(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 start=time.monotonic();C.source_check();C.input_check();C.graph_check();checks=[];hashes={};keys=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen'];stream={}
 with tempfile.TemporaryDirectory(dir=str(C.R.parent),prefix='card052-canary-') as td:
  td=Path(td);n=512;X=np.asarray(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4');graph=td/'edges.npz';src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.array([(i+j)%n for i in range(n) for j in range(1,16)],dtype='i4');np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n);rp=td/'radii.npy';np.save(rp,np.load(C.D/'radii.npy')[:n]);bank=td/'bank.npz';b=np.load(C.D/'pca-bank.npz');np.savez(bank,**{k:b[k][:512] for k in ['replay_X','replay_targets','replay_ids']});del b
  for arm in C.ARMS:
   kw={'X':X,'graph':graph,'radius_path':rp,'bank_path':bank,'checkpoints':[2,4,7,9,18]};p,full,report=fit(arm,18,td/arm,**kw);hashes[arm]=report['state_sha'];stream[arm]={k:full[k] for k in ['torch_rng','cuda_rng','loader_gen','loader_perm','loader_rank_of_node','loader_node_at_rank']};del p;gc.collect();torch.cuda.empty_cache()
   candidates=[]
   for path in (td/arm/'ckpts').glob('ckpt-step*.pt'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if 0<ck['global_step']<18 and 0<ck['loader_pos_idx']<len(ck['loader_perm']):candidates.append((ck['global_step'],path))
   assert candidates;mid=min(candidates)[1];p,res,_=fit(arm,18,td/(arm+'-mid'),resume=mid,**kw);assert all(same(full[k],res[k]) for k in keys),'mid-resume differs';checks.append(arm+' mid-epoch fullstate twin');del p,res;gc.collect();torch.cuda.empty_cache()
   eps=[]
   for path in (td/arm/'ckpts').glob('ckpt-epoch*.pt'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if 0<ck['global_step']<18 and ck['epoch']<full['epoch']:eps.append((ck['global_step'],path))
   assert eps;p,res,_=fit(arm,18,td/(arm+'-epoch'),resume=min(eps)[1],**kw);assert all(same(full[k],res[k]) for k in keys),'epoch-resume differs';checks.append(arm+' epoch boundary twin');del p,res;gc.collect();torch.cuda.empty_cache()
   for tag,override in [('wrong_arm',{'arm':'wrong'}),('wrong_bank',{'replay_bank_sha':'0'*64}),('wrong_weight',{'replay_weight':.999}),('wrong_input',{'input_manifest_sha':'0'*64})]:
    try:fit(arm,18,td/(arm+'-'+tag),resume=mid,ident_override=override,**kw)
    except AssertionError as e:assert 'card052 admission-identity mismatch' in str(e);checks.append(arm+' '+tag+' rejects')
    else:raise AssertionError('bad identity accepted')
   if arm=='ordinary':
    p,off,_=fit(arm,18,td/'disabled-bank',disabled_bank_control=True,**kw);assert all(same(full[k],off[k]) for k in keys),'disabled bank changes baseline';checks.append('disabled bank bit-identical');del p,off;gc.collect();torch.cuda.empty_cache()
  assert len(set(hashes.values()))==3;checks.append('all three endpoint hashes diverge')
  assert all(same(stream['ordinary'][k],stream[a][k]) for a in C.ARMS for k in stream[a]);checks.append('replay independent graph/global RNG')
 C.write(C.O/'card052-device-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'endpoints':hashes,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'scope':'Real1536D core fit,512rows,18successfulsteps, replay/radius active, fullstate epoch andstep resume; wrong identities reject. Not a numericalquality result.'});print('DEVICE PASS',len(checks),flush=True)
if __name__=='__main__':main()
