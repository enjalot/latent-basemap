import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc,hashlib
import numpy as np,torch
import card089_common as C
from card089_fit import fit
from gpu_card060_canary import same
from card089_weights import weights as make_weights
from card089_canary_admission import admitted_canary
from canary_cleanup_faults import device_controls
start=time.monotonic();C.require_release();C.source_check();C.input_check();checks=[];hashes={};keys=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen','mn_gen','mn_rng','dens_gen','dens_rng','hold_gen','hold_rng','deriv_gen','global_step','epoch','train_stats']
cache={}
with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card089-graph-canary-') as path, admitted_canary(cache):
 td=Path(path);n=512;X=np.array(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4');src=np.repeat(np.arange(n,dtype='i4'),15);rng=np.random.default_rng(89089);neighbors=np.array([rng.choice(np.delete(np.arange(n),i),15,replace=False) for i in range(n)],dtype='i4');dst=neighbors.ravel();mutual=(neighbors[neighbors]==np.arange(n)[:,None,None]).any(2);graph=td/'edges.npz';np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n);rp=td/'radii.npy';np.save(rp,np.load(C.D/'radii.npy')[:n])
 for arm in C.ARMS:
  graph=td/f'{arm}-edges.npz';weights=make_weights(mutual,arm)[0].ravel();np.savez(graph,sources=src,targets=dst,weights=weights,n_nodes=n)
  kw={'X':X,'graph':graph,'radius_path':rp,'checkpoints':[1,2,4,7,9,18],'mutual_mask':mutual};p,full,r=fit(arm,18,td/arm,**kw);hashes[arm]=r['state_sha'];first=torch.load(td/arm/'ckpts/ckpt-step1.pt',map_location='cpu',weights_only=False);assert all(float(v['step'])==1 for v in first['optimizer']['state'].values()) and first['train_stats']['positive_lr_optimizer_steps']==1;checks.append(arm+' optimizer reset first successful step');del p;gc.collect();torch.cuda.empty_cache();mids=[];epochs=[]
  for path in (td/arm/'ckpts').glob('*.pt'):
   ck=torch.load(path,map_location='cpu',weights_only=False)
   if 0<ck['global_step']<18:
    if ck['step_checkpoint'] and 0<ck['loader_pos_idx']<len(ck['loader_perm']):mids.append((ck['global_step'],path))
    if not ck['step_checkpoint'] and ck['epoch']<full['epoch']:epochs.append((ck['global_step'],path))
  assert mids and epochs
  for tag,path in [('mid',min(mids)[1]),('epoch',min(epochs)[1])]:
   p,res,_=fit(arm,18,td/(arm+'-'+tag),resume=path,**kw);assert all(same(full[k],res[k]) for k in keys),arm+' '+tag+' fullstate differs';checks.append(arm+' '+tag+' fullstate resume');del p,res;gc.collect();torch.cuda.empty_cache()
  for tag,override in [('protocol',{'protocol_sha':'bad'}),('quality',{'quality_prereg_sha':'bad'}),('mask',{'mutual_mask_sha':'bad'}),('actualmask',{'actual_mutual_mask_values_sha':'bad'}),('ratio',{'raw_ratio':2}),('favored',{'favored_policy':'wrong'}),('arm',{'arm':'wrong'}),('warm',{'warm_sha':'0'*64}),('scale',{'scale_manifest_sha':'0'*64}),('dimension',{'output_dim':8}),('learning_rate',{'lr':.00037}),('policy',{'rankneg_window':123}),('dose',{'dose':19}),('graph',{'graph_sha':'0'*64}),('weights',{'weight_values_sha':'0'*64}),('weighted_flag',{'weighted_edge_sampling':False}),('reference',{'reference_sha':'0'*64}),('data',{'data_manifest_sha':'0'*64}),('parent',{'parent_sha':'0'*64})]:
   try:fit(arm,18,td/(arm+'-reject-'+tag),resume=min(mids)[1],ident_override=override,**kw)
   except AssertionError as e:assert str(e)=='card089 identity mismatch: '+next(iter(override));checks.append(arm+' wrong '+tag+' rejected')
   else:raise AssertionError('wrong graph identity accepted')
 forced=device_controls(fit,C,C.ARMS[-1],kw,td,keys);checks.extend(forced['checks'])
 from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler
 samplers=[];observed={a:[] for a in C.ARMS};expected={a:float(make_weights(mutual,a)[0][mutual].sum(dtype='f8')/make_weights(mutual,a)[0].sum(dtype='f8')) for a in C.ARMS}
 for arm in C.ARMS:
  weights=make_weights(mutual,arm)[0].ravel()
  sampler=DeviceEdgeSampler(DeviceArrayDataset(X,device='cuda'),src,dst,weights,n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,uniform_with_replacement=False,device='cuda');sampler._stash_ids=True;samplers.append(sampler)
 for epoch in range(16):
  for sampler in samplers:iter(sampler)
  for attempt in range(len(samplers[0])):
   positions=[s.pos_idx for s in samplers];labels=[next(s)[2] for s in samplers];npos=len(labels[0])-samplers[0].num_neg
   assert all(bool((v[:npos]==1).all() and (v[npos:]==0).all()) for v in labels)
   for arm,s,pos in zip(C.ARMS,samplers,positions):observed[arm].extend(s.perm[pos:min(pos+s.num_pos,s.n_pos)].cpu().tolist())
   a,b=samplers;assert torch.equal(a._last_all_src[npos:],b._last_all_src[npos:]) and torch.equal(a._last_all_dst[npos:],b._last_all_dst[npos:]) and torch.equal(a.gen.get_state(),b.gen.get_state())
 actual={a:float(mutual.ravel()[observed[a]].mean()) for a in C.ARMS};assert all(len(observed[a])==122880 and abs(actual[a]-expected[a])<.01 for a in C.ARMS),'actual weight exposure mismatch';assert observed[C.ARMS[0]]!=observed[C.ARMS[1]],'identical positive IDs'
 checks.append('CUDA actual122880 draws reciprocal exposure and positive IDs')
 checks.append('CUDA matched-attempt negative IDs and RNG parity across sixteen real sampler epochs')
 assert len(set(hashes.values()))==2,'trained endpoints identical';checks.append('weighted small-graph endpoints validated')
C.write(C.O/'card089-graph-canary.json',{'PASS':True,'actual_fit_forced_cleanup':forced,'canary_admission':cache,'checks':checks,'n_checks':len(checks),'endpoints':hashes,'expected_reciprocal_fraction':expected,'observed_reciprocal_fraction':actual,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.GD/'manifest.json'),'scope':'Real1536D fit on512-node synthetic random support and corresponding original radii; not full2M resume proof; midpoint and epoch resumes preserve model+optimizer+loader/RNG state; wrong warm/scale/arm reject. No assertion of identical sampled exposure across model-dependent rank orders or AMP skips.'});print('PASS',len(checks),flush=True)
