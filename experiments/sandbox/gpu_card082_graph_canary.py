import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc
import numpy as np,torch
import card082_common as C
from card082_fit import fit
from gpu_card060_canary import same
start=time.monotonic();C.source_check();C.input_check();checks=[];hashes={};keys=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen','card081_noise_pairs']
with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card082-graph-canary-') as path:
 td=Path(path);n=512;X=np.array(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4');src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.array([(i+j)%n for i in range(n) for j in range(1,16)],dtype='i4');graph=td/'edges.npz';np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n);rp=td/'radii.npy';np.save(rp,np.load(C.D/'radii.npy')[:n]);qp=td/'uniform-q.npy';np.save(qp,np.full(n,1/n));kw={'X':X,'graph':graph,'radius_path':rp,'noise_q_path':qp,'checkpoints':[2,4,7,9,18]}
 for arm in C.ARMS:
  p,full,r=fit(arm,18,td/arm,**kw);hashes[arm]=r['state_sha'];del p;gc.collect();torch.cuda.empty_cache();mids=[];epochs=[]
  for path in (td/arm/'ckpts').glob('*.pt'):
   ck=torch.load(path,map_location='cpu',weights_only=False)
   if 0<ck['global_step']<18:
    if ck['step_checkpoint'] and 0<ck['loader_pos_idx']<len(ck['loader_perm']):mids.append((ck['global_step'],path))
    if not ck['step_checkpoint'] and ck['epoch']<full['epoch']:epochs.append((ck['global_step'],path))
  assert mids and epochs
  for tag,path in [('mid',min(mids)[1]),('epoch',min(epochs)[1])]:
   p,res,_=fit(arm,18,td/(arm+'-'+tag),resume=path,**kw);assert all(same(full[k],res[k]) for k in keys),arm+' '+tag+' fullstate differs';checks.append(arm+' '+tag+' exact fullstate resume');del p,res;gc.collect();torch.cuda.empty_cache()
  for tag,override in [('arm',{'arm':'wrong'}),('warm',{'warm_sha':'0'*64}),('learning_rate',{'lr':.00037}),('q_hash',{'noise_q_sha':'0'*64}),('band',{'fneg_weight':.5}),('cap',{'neg_tanh_gamma':2.}),('band_threshold',{'fneg_hi':.5}),('kernel',{'kernel_b':.7})]:
   try:fit(arm,18,td/(arm+'-reject-'+tag),resume=min(mids)[1],ident_override=override,**kw)
   except AssertionError as e:assert 'card082 admission-identity mismatch' in str(e);checks.append(arm+' wrong '+tag+' rejected')
   else:raise AssertionError('wrong graph identity accepted')
 p,base,r=fit('no_band',18,td/'registered_control',ident_override={'fneg_weight':1.,'neg_tanh_gamma':4.},**kw);assert r['fneg_telemetry']['final_band_hit_frac_of_neg']>0;hashes['control']=r['state_sha'];del p;gc.collect();torch.cuda.empty_cache()
 assert len(set(hashes.values()))==3;checks.append('both ablations diverge from registered uniform control with actual band exposure')
 # Repeat control under the other arm label: attribution metadata does not alter fit.
 p,repeat,_=fit('no_cap',18,td/'control_repeat',ident_override={'fneg_weight':1.,'neg_tanh_gamma':4.},**kw);assert all(same(base[k],repeat[k]) for k in keys);checks.append('control model optimizer sampler RNG identical across arm labels');del p
C.write(C.O/'card082-graph-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'endpoints':hashes,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'scope':'Real1536D512-node/radii/uniform-CDF fit. Two ablations and registered control differ; exact mid+epoch resumes; recipe/kernel/threshold/q identity rejects. No claim that loss clipping bounds coordinate gradients.'});print('PASS',len(checks),flush=True)
