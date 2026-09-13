"""Tiny synthetic-only exact extra45 and zero-CDF contracts; no data graph build."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,tempfile
from pathlib import Path
import numpy as np,torch
import card088_graph as G
sys.path.insert(0,str(G.R))
from card088_sampler import inverse,matched_sampler
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler

def main():
 torch.set_num_threads(2);checks={};rng=np.random.default_rng(88);n=128
 for label,x in [('random',rng.normal(size=(n,12))),('ties',np.tile([1.,0.],(n,1)))]:
  db=torch.tensor(x,dtype=torch.float16);old=np.array([[(i+j)%n for j in range(70,85)] for i in range(n)],dtype='i4');rows=np.array([0,7,63,127]);reference=db.double().numpy();reference/=np.linalg.norm(reference,axis=1,keepdims=True)
  for block in [17,64,128]:
   ids,sc,_=G.search(db,rows,old[rows],block=block,k=46,dtype=torch.float64)
   for j,row in enumerate(rows):
    eligible=np.setdiff1d(np.arange(n),np.r_[row,old[row]]);ss=reference[eligible]@reference[row];want=eligible[np.lexsort((eligible,-ss))[:46]];assert np.array_equal(ids[j].numpy(),want),label+' independent scalar ordering'
   checks[label+'_chunk_'+str(block)]=True
  # Old15 are deliberately NOT nearest15; preserved outside-set definition.
 assert np.array_equal(old,np.array([[(i+j)%n for j in range(70,85)] for i in range(n)],dtype='i4'));checks['original15_not_recomputed']=True
 w=G.weights('original15',n);v=G.weights('mixture',n);assert np.all(w.sum(1)==90) and np.all(v.sum(1)==90) and np.all(v[:,:15].sum(1)==v[:,15:].sum(1));checks['integer_exact_row_and_half_mass']=True
 src=np.repeat(np.arange(n,dtype='i4'),60);dst=np.array([(i+j)%n for i in range(n) for j in range(1,61)],dtype='i4')
 from card088_offpath import restoration_control
 offpath=restoration_control(np.arange(n,dtype='f4')[:,None],src,dst,v.ravel(),n,'cpu');checks['actual_base_sampler_offpath_restoration']=offpath['PASS']
 def make(weight):
  s=DeviceEdgeSampler(DeviceArrayDataset(np.arange(n,dtype='f4')[:,None],device='cpu'),src,dst,weight.ravel(),n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,device='cpu');s._stash_ids=True;return s
 with matched_sampler():
  samplers=[make(w),make(v)]
  for epoch in range(3):
   for s in samplers:iter(s)
   assert all(s.n_pos==15*n and s.perm.shape==(15*n,) for s in samplers)
   for batch in range(len(samplers[0])):
    labels=[next(s)[2] for s in samplers];a,b=samplers;np_=len(labels[0])-a.num_neg
    assert np_==min(a.num_pos,n*15-batch*a.num_pos);assert torch.equal(a._last_all_src[np_:],b._last_all_src[np_:]) and torch.equal(a._last_all_dst[np_:],b._last_all_dst[np_:]) and torch.equal(a.gen.get_state(),b.gen.get_state())
    assert bool((a._card088_last_edge_idx%60<15).all())
  checks['actual_three_epoch_negative_ID_RNG_tail_parity']=True
  for s in samplers:
   c=s.sample_cdf;u=torch.cat([torch.zeros(1,dtype=torch.float64),torch.unique(c[c<1]),torch.nextafter(torch.ones(1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64))]);idx=inverse(c,u);ww=w if s._card088_control else v;assert np.all(ww.ravel()[idx.numpy()]>0)
  # Explicit leading/internal/trailing zero fixture, exact CDF boundaries.
  c=torch.tensor([0.,.25,.25,1.,1.],dtype=torch.float64);assert torch.equal(inverse(c,torch.tensor([0.,.25,np.nextafter(1.,0.)],dtype=torch.float64)),torch.tensor([1,3,3]));checks['zero_weight_exact_boundary_exclusion']=True
  s=samplers[1];draw=s._draw_idx(100000).numpy();near=float((draw%60<15).mean());assert abs(near-.5)<.008;checks['actual_mixture_draw_mass']=True
  # A new sampler restores the actual logical epoch state including indices>30N.
  a=make(v);iter(a);next(a);saved=(a.perm.clone(),a.pos_idx,a.gen.get_state());expect=next(a);b=make(v);b.perm,b.pos_idx=saved[:2];b.gen.set_state(saved[2]);actual=next(b);assert all(torch.equal(x,y) for x,y in zip(expect,actual));checks['actual_sampler_mid_resume']=True
 def replay(split=None):
  state={'attempted_batches':0};seen=[]
  with matched_sampler('mixture',stats=lambda:state):
   s=make(v)
   while state['attempted_batches']<6:
    if s.perm is None or s.pos_idx>=s.n_pos:iter(s)
    next(s);state['attempted_batches']+=1
    seen.append((s._last_all_src.clone(),s._last_all_dst.clone(),s.gen.get_state().clone()))
    if state['attempted_batches']==split:
     with tempfile.TemporaryDirectory() as path:
      f=Path(path)/'sampler.pt';torch.save({'state':state,'perm':s.perm,'pos':s.pos_idx,'rng':s.gen.get_state()},f);saved=torch.load(f,weights_only=False)
     state=saved['state'];s=make(v);s.perm=saved['perm'];s.pos_idx=saved['pos'];s.gen.set_state(saved['rng'])
  return seen,state
 full,state=replay()
 for split in [1,2]:
  continued,end=replay(split);assert state==end and all(torch.equal(a,b) for left,right in zip(full,continued) for a,b in zip(left,right));checks['independent_positive_epoch_reconstruction_'+str(split)]=True
 with matched_sampler():
  a=make(v);b=make(v);iter(a);iter(b);a._draw_idx(17);next(a);next(b);np_=a.num_pos;assert torch.equal(a._last_all_src[np_:],b._last_all_src[np_:]) and torch.equal(a._last_all_dst[np_:],b._last_all_dst[np_:]);checks['positive_draws_do_not_advance_noise_rng']=True
 try:
  with matched_sampler('mixture'):make(w)
 except AssertionError as e:assert str(e)=='card088 sampler arm law mismatch';checks['wrong_arm_actual_weight_law_rejected']=True
 else:raise AssertionError('wrong arm weights accepted')
 try:G.validate_rows(np.zeros((n,15),dtype='i4'),np.zeros((n,45),dtype='i4'),n)
 except AssertionError as e:assert str(e)=='support self';checks['self_rejected_specific']=True
 else:raise AssertionError('self accepted')
 G.write(G.O/'card088-graph-cpu-contracts.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'observed_mixture_near_fraction':near,'scope':'Synthetic128nodes only: independentFP64 scalar exclusion/ties/chunking, integer mass,actual CPU sampler boundary zeros,three epochs/tails/noise RNG and midresume. No real graph or GPU.'});print('GRAPH CPU PASS',len(checks))
if __name__=='__main__':main()
