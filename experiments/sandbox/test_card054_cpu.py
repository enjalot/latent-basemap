"""Complete CPU bank joins, independent correlation/gradients and selection guards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,time,tempfile
import numpy as np,torch
from scipy.spatial.distance import cdist
import card054_common as C
from card054_lmc import LandmarkLoss,negative_correlation
sys.path.insert(0,str(C.O));import score_card054 as S

def main():
 start=time.monotonic();C.input_check();b=np.load(C.D/'landmark-bank.npz');x=np.load(C.D/'train.f16.npy',mmap_mode='r');ids=np.load(C.D/'train-ids.npy');checks={};pool=b['pool_local'];land=b['landmark_local'];px=b['pool_X'];lx=b['landmark_X'];dd=b['distances'];checks['full_pool_input_join']=np.array_equal(px,x[pool]);checks['landmark_input_join']=np.array_equal(lx,x[land]);checks['global_ids']=np.array_equal(b['pool_ids'],ids[pool]) and np.array_equal(b['landmark_ids'],ids[land]);checks['uniform_pool_seed']=np.array_equal(pool,np.random.default_rng(54054).choice(len(x),200000,replace=False));checks['uniform_landmark_seed']=np.array_equal(land,np.random.default_rng(54055).choice(np.setdiff1d(np.arange(len(x)),pool),64,replace=False));checks['disjoint_pool_landmarks']=not np.isin(pool,land).any();checks['full_distances']=True
 for i in range(0,len(pool),1024):
  dx=px[i:i+1024].astype('f8')[:,None,:]-lx.astype('f8')[None,:,:];d=np.sqrt((dx*dx).sum(2)).astype('f4');checks['full_distances'] &= np.allclose(d,dd[i:i+1024],rtol=0,atol=2e-7)
 seal=Path('/data2/monet/eval-common-v2');checks['all_support_excludes_queries']=not np.isin(np.load(seal/'val_idx.npy'),ids).any();checks['all_support_excludes_reference']=not np.isin(np.load(seal/'ref_idx.npy'),ids).any()
 torch.set_num_threads(2);torch.manual_seed(54054);y=torch.randn(32,3,dtype=torch.float64,requires_grad=True);l=torch.randn(7,3,dtype=torch.float64,requires_grad=True);target=torch.rand(32,7,dtype=torch.float64);d=((y[:,None]-l[None])**2).sum(2).add(1e-12).sqrt();loss,valid=negative_correlation(target,d);g=torch.autograd.grad(loss,(y,l),retain_graph=True);exact=-torch.corrcoef(torch.stack([target.ravel(),d.ravel()]))[0,1];g2=torch.autograd.grad(exact,(y,l));checks['scalar_correlation']=abs(float(loss-exact))<1e-6;checks['both_endpoint_gradients']=all(torch.isfinite(v).all() and torch.linalg.norm(v)>1e-6 for v in g);checks['independent_FP64_gradient']=all(torch.allclose(a,z,atol=2e-7,rtol=2e-5) for a,z in zip(g,g2));wrong,_=negative_correlation(target.roll(1,0),d);checks['wrong_pairing_changes_loss']=abs(float(wrong-loss))>1e-4
 for tag,xx,yy in [('target',torch.ones(32,7),torch.rand(32,7,requires_grad=True)),('map',torch.rand(32,7),torch.ones(32,7,requires_grad=True))]:
  loss,v=negative_correlation(xx,yy);gg=torch.autograd.grad(loss,yy)[0];checks[tag+'_zero_variance_zero_grad']=not bool(v) and float(loss)==0 and torch.equal(gg,torch.zeros_like(gg))
 with tempfile.TemporaryDirectory(dir=str(C.O)) as tmp:
  bp=Path(tmp)/'bank.npz';np.savez(bp,pool_X=px[:512],landmark_X=lx,distances=dd[:512]);h=LandmarkLoss(bp,.02,device='cpu');torch.manual_seed(14);state=torch.get_rng_state().clone();i0=h.indices(0);i1=h.indices(1);checks['independent_global_RNG']=torch.equal(state,torch.get_rng_state());checks['stateless_repeat']=torch.equal(i0,h.indices(0));checks['distinct_steps']=not torch.equal(i0,i1);h2=LandmarkLoss(bp,.02,device='cpu');checks['resume_indices_exact']=torch.equal(h.indices(173),h2.indices(173))
 z={'val_source':np.repeat(np.array(['a','b','c']),100),'decile_zero_based':np.tile(np.arange(10),30),'enc_radius':np.arange(300.)+1}
 for a in S.ARMS:
  for budget in S.BUDGETS:z[f'{a}_B{budget}']=np.ones(300)*.6
  z[a+'_continuity_per_query']=np.ones(30)*.8;z[a+'_map_radius']=z['enc_radius'].copy()
 good={'disjoint_pair_CPD':{'parent':.2,'ordinary':.2,'lmc':.24},'candidate_minus_ordinary_CI95':{'lmc':[.01,.06]}};checks['single_candidate_selection']=S.aggregate(z,{},good)['selected_candidate']=='lmc';bad={**good,'candidate_minus_ordinary_CI95':{'lmc':[-.001,.06]}};checks['primary_CI_AND']=not S.aggregate(z,{},bad)['GATE_PASS'];z['lmc_B250'][z['decile_zero_based']==9]-=.011;checks['sparse_guard_AND']=not S.aggregate(z,{},good)['GATE_PASS'];z['lmc_B250'][:]=.6;z['parent_B2000'][:]=.606;checks['parent_guard_AND']=not S.aggregate(z,{},good)['GATE_PASS']
 r={'PASS':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'cpu_s':time.monotonic()-start,'input_manifest_sha':C.sha(C.D/'inputs-manifest.json'),'test_sha':C.sha(__file__),'scope':'All200Kx64 distances independently direct-summed, complete row/seed/exclusion joins, FP64 scalar/gradient comparison, zero-variance controls, statelessCPU sampler/globalRNG and gatefixtures. NoGPU or qualityoutcomes.'};C.write(C.O/'card054-cpu-canary.json',r);print(r);assert r['PASS']
if __name__=='__main__':main()
