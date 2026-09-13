import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc
import numpy as np,torch
import card090_common as C
from card090_fit import fit
from card090_state import compare
from card090_canary_admission import admitted_canary

def main():
 start=time.monotonic();checks=[];hashes={};cache={}
 with admitted_canary(cache),tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card090-canary-') as path:
  td=Path(path);n=512;X=np.array(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4');src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.array([(i+j)%n for i in range(n) for j in range(1,16)],dtype='i4');graph=td/'edges.npz';np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n);rp=td/'radii.npy';np.save(rp,np.load(C.D/'radii.npy')[:n])
  for arm in C.ARMS:
   kw={'X':X,'graph':graph,'radius_path':rp,'checkpoints':[2,4,7,9,18]};p,full,r=fit(arm,18,td/arm,**kw);hashes[arm]=r['state_sha'];del p;gc.collect();torch.cuda.empty_cache();mids=[];epochs=[]
   for path in (td/arm/'ckpts').glob('*.pt'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if 0<ck['global_step']<18:
     if ck['step_checkpoint'] and 0<ck['loader_pos_idx']<len(ck['loader_perm']):mids.append((ck['global_step'],path))
     if not ck['step_checkpoint'] and ck['epoch']<full['epoch']:epochs.append((ck['global_step'],path))
   assert mids and epochs,'missing genuine MID/EPOCH'
   for tag,path in [('mid',min(mids)[1]),('epoch',min(epochs)[1])]:
    p,res,_=fit(arm,18,td/(arm+'-'+tag),resume=path,**kw);compare(full,res);checks.append(arm+' '+tag+' fullstate');del p,res;gc.collect();torch.cuda.empty_cache()
   for tag,override in [('seed',{'seed':45}),('policy',{'rankneg_window':17}),('parent',{'parent_sha':'0'*64}),('dose',{'dose':19}),('arm',{'arm':'wrong'}),('warm',{'warm_sha':'0'*64}),('lr',{'lr':.00037})]:
    try:fit(arm,18,td/(arm+'-reject-'+tag),resume=min(mids)[1],ident_override=override,**kw)
    except AssertionError as e:assert str(e)=='card090 admission-identity mismatch',str(e);checks.append(arm+' wrong '+tag)
    else:raise AssertionError('wrong identity accepted')
   print(arm,'MID/EPOCH/identity PASS',flush=True)
  assert hashes['ranked43']!=hashes['ranked44'] and hashes['uniform43']!=hashes['uniform44'],'seed ineffective'
  checks.append('seed-dependent endpoints in both policies')
 C.write(C.O/'card090-graph-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'endpoints':hashes,'cache':cache,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.D/'inputs-manifest.json'),'scope':'Actual1536D residual2048 3D model,512-node synthetic fixed15 graph, ranked window511 (legacy small fixture convention), four arms18updates; all durable numerical state, MID/EPOCH; not full2M quality or policy-to-policy RNG parity.'})
if __name__=='__main__':main()
