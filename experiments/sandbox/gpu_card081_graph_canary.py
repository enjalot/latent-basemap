import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc
import numpy as np,torch
import card081_common as C
from card081_fit import fit
from gpu_card060_canary import same
start=time.monotonic();C.source_check();C.input_check();checks=[];hashes={};keys=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen','card081_noise_pairs']
with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card081-graph-canary-') as path:
 td=Path(path);n=512;X=np.array(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4');src=np.repeat(np.arange(n,dtype='i4'),15);dst=np.array([(i+j)%n for i in range(n) for j in range(1,16)],dtype='i4');graph=td/'edges.npz';np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n);rp=td/'radii.npy';np.save(rp,np.load(C.D/'radii.npy')[:n]);qs={'uniform':np.full(n,1/n),'degree':np.arange(1,n+1,dtype='f8')/sum(range(1,n+1))};qs['shuffled']=qs['degree'][::-1].copy()
 for arm in C.ARMS:
  qp=td/(arm+'-q.npy');np.save(qp,qs[arm]);kw={'X':X,'graph':graph,'radius_path':rp,'noise_q_path':qp,'checkpoints':[2,4,7,9,18]};p,full,r=fit(arm,18,td/arm,**kw);hashes[arm]=r['state_sha'];del p;gc.collect();torch.cuda.empty_cache();mids=[];epochs=[]
  for path in (td/arm/'ckpts').glob('*.pt'):
   ck=torch.load(path,map_location='cpu',weights_only=False)
   if 0<ck['global_step']<18:
    if ck['step_checkpoint'] and 0<ck['loader_pos_idx']<len(ck['loader_perm']):mids.append((ck['global_step'],path))
    if not ck['step_checkpoint'] and ck['epoch']<full['epoch']:epochs.append((ck['global_step'],path))
  assert mids and epochs
  for tag,path in [('mid',min(mids)[1]),('epoch',min(epochs)[1])]:
   p,res,_=fit(arm,18,td/(arm+'-'+tag),resume=path,**kw);assert all(same(full[k],res[k]) for k in keys),arm+' '+tag+' fullstate differs';checks.append(arm+' '+tag+' fullstate resume incl noise exposure');del p,res;gc.collect();torch.cuda.empty_cache()
  for tag,override in [('arm',{'arm':'wrong'}),('warm',{'warm_sha':'0'*64}),('scale',{'scale_manifest_sha':'0'*64}),('learning_rate',{'lr':.00037}),('policy',{'rankneg_window':123}),('q_hash',{'noise_q_sha':'0'*64})]:
   try:fit(arm,18,td/(arm+'-reject-'+tag),resume=min(mids)[1],ident_override=override,**kw)
   except AssertionError as e:assert 'card081 admission-identity mismatch' in str(e);checks.append(arm+' wrong '+tag+' rejected')
   else:raise AssertionError('wrong graph identity accepted')
 # Explicitly absent attribute vs configured None must have no branch/RNG/model effects.
 kw={'X':X,'graph':graph,'radius_path':rp,'checkpoints':[2,4,7,9,18],'ident_override':{'noise_q_path':None,'noise_q_sha':None}}
 p,off,_=fit('uniform',18,td/'none',**kw);del p;gc.collect();torch.cuda.empty_cache()
 configure=C.configure
 def unset(*a,**k):
  configure(*a,**k);delattr(a[0],'_card081_q_path')
 C.configure=unset
 try:p,absent,_=fit('uniform',18,td/'absent',**kw)
 finally:C.configure=configure
 del p;gc.collect();torch.cuda.empty_cache()
 assert all(same(off[k],absent[k]) for k in keys);checks.append('default off attr absent versus None model optimizer sampler RNG bit-identical')
 assert len(set(hashes.values()))==3 and C.state_sha(off['model'])!=hashes['uniform'];checks.append('all three active endpoints diverge; new uniform stream differs from legacy')
 # GPU empirical law on an independent small probability vector.
 from basemap.pumap.parametric_umap.degree_noise import ConditionalCDF
 q=np.array([.1,.2,.3,.4]);d=ConditionalCDF(q,'cuda');g=torch.Generator(device='cuda').manual_seed(81081);i,j=d.draw(1000000,g);v=torch.bincount(i*4+j,minlength=16).cpu().numpy().reshape(4,4)/len(i);t=q[:,None]*q[None,:]/(1-q[:,None]);np.fill_diagonal(t,0);assert np.max(abs(v-t))<.0015;checks.append('GPU million-pair ordered conditional law and no self')
C.write(C.O/'card081-graph-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'endpoints':hashes,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'scope':'Real1536D fit,512 nodes,radii active; every arm mid+epoch exact resume incl sampler exposure; wrong identities reject. CPU and GPU probability laws checked separately. Cross-arm exposure can diverge following AMP skips.'});print('PASS',len(checks),flush=True)
