import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc,hashlib
import numpy as np,torch
import card088_common as C
from card088_fit import fit
from card088_graph import weights as make_weights
from card088_sampler import matched_sampler
from gpu_card060_canary import same
start=time.monotonic();C.require_release();C.source_check();C.input_check();checks=[];hashes={};training_exposure={};keys=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen','mn_gen','mn_rng','dens_gen','dens_rng','hold_gen','hold_rng','deriv_gen','global_step','epoch','train_stats']
with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card088-graph-canary-') as path:
 td=Path(path);n=512;X=np.array(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4');src=np.repeat(np.arange(n,dtype='i4'),60);dst=np.array([(i+j)%n for i in range(n) for j in range(1,61)],dtype='i4');graph=td/'edges.npz';np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n);rp=td/'radii.npy';np.save(rp,np.load(C.D/'radii.npy')[:n])
 for arm in C.ARMS:
  graph=td/f'{arm}-edges.npz';weights=make_weights(arm,n).ravel();np.savez(graph,sources=src,targets=dst,weights=weights,n_nodes=n)
  kw={'X':X,'graph':graph,'radius_path':rp,'checkpoints':[1,2,4,7,9,18]};p,full,r=fit(arm,18,td/arm,**kw);hashes[arm]=r['state_sha'];training_exposure[arm]=r['support_fractions'];first=torch.load(td/arm/'ckpts/ckpt-step1.pt',map_location='cpu',weights_only=False);assert all(float(v['step'])==1 for v in first['optimizer']['state'].values()) and first['train_stats']['positive_lr_optimizer_steps']==1;checks.append(arm+' optimizer reset first successful step');del p;gc.collect();torch.cuda.empty_cache();mids=[];epochs=[]
  for path in (td/arm/'ckpts').glob('*.pt'):
   ck=torch.load(path,map_location='cpu',weights_only=False)
   if 0<ck['global_step']<18:
    if ck['step_checkpoint'] and 0<ck['loader_pos_idx']<len(ck['loader_perm']):mids.append((ck['global_step'],path))
    if not ck['step_checkpoint'] and ck['epoch']<full['epoch']:epochs.append((ck['global_step'],path))
  assert mids and epochs
  for tag,path in [('mid',min(mids)[1]),('epoch',min(epochs)[1])]:
   p,res,_=fit(arm,18,td/(arm+'-'+tag),resume=path,**kw);assert all(same(full[k],res[k]) for k in keys),arm+' '+tag+' fullstate differs';checks.append(arm+' '+tag+' fullstate resume');del p,res;gc.collect();torch.cuda.empty_cache()
  for tag,override in [('protocol',{'protocol_sha':'0'*64}),('quality',{'quality_prereg_sha':'0'*64}),('arm',{'arm':'wrong'}),('warm',{'warm_sha':'0'*64}),('scale',{'scale_manifest_sha':'0'*64}),('dimension',{'output_dim':8}),('learning_rate',{'lr':.00037}),('policy',{'rankneg_window':123}),('dose',{'dose':19}),('graph',{'graph_sha':'0'*64}),('weights',{'weight_values_sha':'0'*64}),('weighted_flag',{'weighted_edge_sampling':False}),('reference',{'reference_sha':'0'*64}),('data',{'data_manifest_sha':'0'*64}),('parent',{'parent_sha':'0'*64}),('law',{'positive_support_law':[1,1]}),('logical_epoch',{'logical_epoch_draws':123}),('cdf_side',{'cdf_side':'left'}),('columns',{'endpoint_columns':15}),('endpoints',{'endpoints_sha':{'sources':'bad','targets':'bad'}}),('positive_rng',{'positive_rng_policy':'shared'}),('negative_rng',{'negative_rng_policy':'shared'})]:
   try:fit(arm,18,td/(arm+'-reject-'+tag),resume=min(mids)[1],ident_override=override,**kw)
   except AssertionError as e:assert str(e)=='card088 identity mismatch: '+next(iter(override));checks.append(arm+' wrong '+tag+' rejected')
   else:raise AssertionError('wrong graph identity accepted')
 from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler
 from card088_offpath import restoration_control
 offpath=restoration_control(X,src,dst,make_weights('mixture',n).ravel(),n,'cuda');checks.append('actual CUDA base sampler offpath IDs RNG restoration')
 exposure={a:{'near':0,'extra':0} for a in C.ARMS};positive_hashers={a:hashlib.sha256() for a in C.ARMS}
 with matched_sampler():
  samplers=[]
  for arm in C.ARMS:
   weights=make_weights(arm,n).ravel()
   sampler=DeviceEdgeSampler(DeviceArrayDataset(X,device='cuda'),src,dst,weights,n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,uniform_with_replacement=False,device='cuda');sampler._stash_ids=True;samplers.append(sampler)
  for epoch in range(16):
   for sampler in samplers:iter(sampler)
   for attempt in range(len(samplers[0])):
    labels=[next(s)[2] for s in samplers];npos=len(labels[0])-samplers[0].num_neg
    assert all(bool((v[:npos]==1).all() and (v[npos:]==0).all()) for v in labels)
    for arm,s in zip(C.ARMS,samplers):
     indices=s._card088_last_edge_idx;near=int((indices%60<15).sum());exposure[arm]['near']+=near;exposure[arm]['extra']+=len(indices)-near;positive_hashers[arm].update(s._last_all_src[:npos].cpu().numpy().tobytes());positive_hashers[arm].update(s._last_all_dst[:npos].cpu().numpy().tobytes())
    a,b=samplers;assert torch.equal(a._last_all_src[npos:],b._last_all_src[npos:]) and torch.equal(a._last_all_dst[npos:],b._last_all_dst[npos:]) and torch.equal(a.gen.get_state(),b.gen.get_state())
 checks.append('CUDA matched-attempt negative IDs and RNG parity across sixteen real sampler epochs')
 assert all(sum(v.values())==122880 for v in exposure.values()),'fixed exposure panel dose mismatch'
 positive_hashes={a:h.hexdigest() for a,h in positive_hashers.items()}
 assert exposure['original15']['extra']==0,'control extra45 exposure'
 fraction=exposure['mixture']['near']/sum(exposure['mixture'].values());assert abs(fraction-.5)<.01,'mixture actual exposure fraction'
 assert positive_hashes['original15']!=positive_hashes['mixture'],'positive endpoint IDs identical'
 assert hashes['original15']!=hashes['mixture'],'trained endpoint hashes identical'
 assert all(v['extra45']==0 for v in training_exposure['original15'].values()),'trained control extra exposure'
 checks.extend(['actual control zero extra45','fixed122880-draw mixture half mass','positive endpoint IDs differ','trained endpoint hashes differ'])
C.write(C.O/'card088-graph-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'endpoints':hashes,'sampler_exposure':exposure,'mixture_near_fraction':fraction,'positive_ID_hashes':positive_hashes,'training_support_fractions':training_exposure,'offpath':offpath,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.GD/'manifest.json'),'scope':'Real1536D fit on512-node synthetic ring support with corresponding original radii only; not full2M device proof; midpoint and epoch resumes preserve model+optimizer+loader/RNG state; wrong warm/scale/arm reject. No assertion of identical sampled exposure across model-dependent rank orders or AMP skips.'});print('PASS',len(checks),flush=True)
