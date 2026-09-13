import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc
import numpy as np,torch
import card085_common as C
from card085_fit import fit
from gpu_card060_canary import same
start=time.monotonic();C.require_release();C.source_check();C.input_check();checks=[];hashes={};keys=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen']
with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card085-graph-canary-') as path:
 td=Path(path);n=512;X=np.array(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4');src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.array([(i+j)%n for i in range(n) for j in range(1,16)],dtype='i4');graph=td/'edges.npz';np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n);rp=td/'radii.npy';np.save(rp,np.load(C.D/'radii.npy')[:n])
 for arm in C.ARMS:
  kw={'warm_path':C.warm('fresh'),'X':X,'graph':graph,'radius_path':rp,'checkpoints':[1,2,4,7,9,18]};p,full,r=fit(arm,18,td/arm,**kw);hashes[arm]=r['state_sha'];first=torch.load(td/arm/'ckpts/ckpt-step1.pt',map_location='cpu',weights_only=False);assert all(float(v['step'])==1 for v in first['optimizer']['state'].values()) and first['train_stats']['positive_lr_optimizer_steps']==1;checks.append(arm+' optimizer reset first successful step');del p;gc.collect();torch.cuda.empty_cache();mids=[];epochs=[]
  for path in (td/arm/'ckpts').glob('*.pt'):
   ck=torch.load(path,map_location='cpu',weights_only=False)
   if 0<ck['global_step']<18:
    if ck['step_checkpoint'] and 0<ck['loader_pos_idx']<len(ck['loader_perm']):mids.append((ck['global_step'],path))
    if not ck['step_checkpoint'] and ck['epoch']<full['epoch']:epochs.append((ck['global_step'],path))
  assert mids and epochs
  for tag,path in [('mid',min(mids)[1]),('epoch',min(epochs)[1])]:
   p,res,_=fit(arm,18,td/(arm+'-'+tag),resume=path,**kw);assert all(same(full[k],res[k]) for k in keys),arm+' '+tag+' fullstate differs';checks.append(arm+' '+tag+' fullstate resume');del p,res;gc.collect();torch.cuda.empty_cache()
  for tag,override in [('arm',{'arm':'wrong'}),('warm',{'warm_sha':'0'*64}),('scale',{'scale_manifest_sha':'0'*64}),('dimension',{'output_dim':8}),('learning_rate',{'lr':.00037}),('policy',{'rankneg_window':123}),('dose',{'dose':19}),('graph',{'graph_sha':'0'*64}),('source',{'source_files':{}}),('init',{'original_untrained_sha':'0'*64}),('parent',{'parent_sha':'0'*64})]:
   try:fit(arm,18,td/(arm+'-reject-'+tag),resume=min(mids)[1],ident_override=override,**kw)
   except AssertionError as e:assert str(e)=='card085 identity mismatch: '+next(iter(override));checks.append(arm+' wrong '+tag+' rejected')
   else:raise AssertionError('wrong graph identity accepted')
 assert len(hashes)==2;checks.append('uniform small-graph endpoint validated')
C.write(C.O/'card085-graph-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'endpoints':hashes,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'history_cpu_sha':C.sha(C.O/'card085-history-compatibility.json'),'scope':'Real1536D fit,512 nodes,full radii; midpoint and epoch resumes preserve model+optimizer+loader/RNG state; wrong warm/scale/arm reject. No assertion of identical sampled exposure across model-dependent rank orders or AMP skips.'});print('PASS',len(checks),flush=True)
