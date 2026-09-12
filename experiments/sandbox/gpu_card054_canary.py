"""Actual landmark callback and original-core off-path controls; no quality data."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc,subprocess
import numpy as np,torch
import card054_common as C
from card054_fit import fit

def same(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 start=time.monotonic();C.source_check();C.input_check();C.graph_check();checks=[];hashes={};stream={};keys=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen']
 with tempfile.TemporaryDirectory(dir=str(C.R.parent),prefix='card054-canary-') as path:
  td=Path(path);n=512;X=np.asarray(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4');np.save(td/'X.npy',X);graph=td/'edges.npz';src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.array([(i+j)%n for i in range(n) for j in range(1,16)],dtype='i4');np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n);rp=td/'radii.npy';np.save(rp,np.load(C.D/'radii.npy')[:n]);bank=td/'bank.npz';b=np.load(C.D/'landmark-bank.npz');np.savez(bank,pool_X=b['pool_X'][:512],landmark_X=b['landmark_X'],distances=b['distances'][:512]);del b
  for arm in C.ARMS:
   kw={'X':X,'graph':graph,'radius_path':rp,'bank_path':bank,'checkpoints':[2,4,7,9,18]};p,full,report=fit(arm,18,td/arm,**kw);hashes[arm]=report['state_sha'];probe=torch.load(td/arm/'ckpts/ckpt-step2.pt',map_location='cpu',weights_only=False);stream[arm]={k:probe[k] for k in ['torch_rng','cuda_rng','loader_gen','loader_perm','loader_rank_of_node','loader_node_at_rank']};del p,probe;gc.collect();torch.cuda.empty_cache()
   mids=[];eps=[]
   for f in (td/arm/'ckpts').glob('*.pt'):
    ck=torch.load(f,map_location='cpu',weights_only=False)
    if 0<ck['global_step']<18:
     if ck['step_checkpoint'] and 0<ck['loader_pos_idx']<len(ck['loader_perm']):mids.append((ck['global_step'],f))
     if not ck['step_checkpoint'] and ck['epoch']<full['epoch']:eps.append((ck['global_step'],f))
   assert mids and eps
   for tag,f in [('mid',min(mids)[1]),('epoch',min(eps)[1])]:
    p,res,rr=fit(arm,18,td/(arm+'-'+tag),resume=f,**kw);assert all(same(full[k],res[k]) for k in keys),arm+' '+tag+' differs';checks.append(arm+' '+tag+' fullstate resume');
    if arm=='lmc':
     shared=rr['lmc_stats']['probes'];assert shared and all(shared[k]==report['lmc_stats']['probes'][k] for k in shared);checks.append(tag+' stateless row/loss probes same after resume')
    del p,res;gc.collect();torch.cuda.empty_cache()
   for tag,override in [('wrong_arm',{'arm':'wrong'}),('wrong_bank',{'lmc_bank_sha':'0'*64}),('wrong_weight',{'lmc_weight':.9})]:
    try:fit(arm,18,td/(arm+'-'+tag),resume=min(mids)[1],ident_override=override,**kw)
    except AssertionError as e:assert 'card054 admission-identity mismatch' in str(e);checks.append(arm+' '+tag+' rejects')
    else:raise AssertionError('wrong identity accepted')
   if arm=='ordinary':
    torch.save(full,td/'new054-off.pt');subprocess.run(['/home/enjalot/code/latent-basemap/.venv/bin/python',str(C.R/'experiments/sandbox/card054_original_off_control.py'),str(td)],check=True,timeout=120);old=torch.load(td/'old052-end.pt',map_location='cpu',weights_only=False);assert all(same(full[k],old[k]) for k in keys),'defaultoff differs from original052 core';checks.append('actual original052 versus054 defaultoff fullstate bit-identical');del old
   else:assert report['lmc_stats']['attempted_calls']>=18 and report['lmc_stats']['invalid_variance_calls']==0;checks.append('real LMC exposed with finite nondegenerate losses')
  assert len(set(hashes.values()))==2;checks.append('two exposed endpoint hashes differ');assert all(same(stream['ordinary'][k],stream['lmc'][k]) for k in stream['ordinary']);checks.append('graph/global RNG unchanged at common early probe')
 C.write(C.O/'card054-device-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'endpoints':hashes,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'scope':'Real1536D fit with512-row graph and actual landmark callback,18successes; original052 core offcontrol in separate process, midpoint andepoch resume, wrongidentities, stateless sample/loss probes.'});print('PASS',len(checks),flush=True)
if __name__=='__main__':main()
