"""Actual CPU sampler/output/Adam/scaler/RNG/offpath and serial CDF contracts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,tempfile,copy
from contextlib import nullcontext
import numpy as np,torch
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R.parent/'overseer-codex'
from card086_serial_cdf import build
from card086_cdf_adapter import fixed_cdf,values_sha
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler
from gpu_card060_canary import same

def run(repaired,split=None,membership=False):
 torch.manual_seed(42);n=512;src=np.repeat(np.arange(n,dtype='i4'),15);dst=((np.arange(n)[:,None]+np.arange(1,16))%n).astype('i4').ravel();w=np.ones(n*15,dtype='f4') if not membership else np.tile(np.array([1,.5,.1,.01,.001,.0001,0,1e-20,1e-35,.2,.3,.4,.5,.6,.7],dtype='f4'),n);cdf,_=build(w);X=np.arange(n,dtype='f4')[:,None]/n
 def make():
  s=DeviceEdgeSampler(DeviceArrayDataset(X,device='cpu'),src,dst,w,n,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,device='cpu');s._stash_ids=True;return s
 seen=[];model=torch.nn.Linear(1,2);opt=torch.optim.AdamW(model.parameters(),lr=.0001);scaler=torch.amp.GradScaler('cpu',init_scale=16)
 with fixed_cdf(cdf,values_sha(cdf),values_sha(w),require_old_exact=not membership) if repaired else nullcontext():
  s=make();attempt=0
  for epoch in range(3):
   iter(s)
   for batch in range(len(s)):
    x,y,labels=next(s);seen.append((s._last_all_src.clone(),s._last_all_dst.clone(),labels.clone(),s.gen.get_state().clone()));opt.zero_grad();loss=((model(x)-model(y)).square().sum(1)*(.5+labels)).mean()
    if batch==4:loss=loss*float('inf')
    scaler.scale(loss).backward();scaler.step(opt);scaler.update();attempt+=1
    if attempt==split:
     with tempfile.TemporaryDirectory() as td:
      p=Path(td)/'state.pt';torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'scaler':scaler.state_dict(),'perm':s.perm,'pos':s.pos_idx,'rng':s.gen.get_state(),'torch_rng':torch.get_rng_state()},p);ck=torch.load(p,weights_only=False)
     s=make();s.perm=ck['perm'];s.pos_idx=ck['pos'];s.gen.set_state(ck['rng']);model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt']);scaler.load_state_dict(ck['scaler']);torch.set_rng_state(ck['torch_rng'])
 return {'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'sampler_rng':s.gen.get_state(),'torch_rng':torch.get_rng_state(),'seen':seen}
def main():
 torch.set_num_threads(2);checks={};method=DeviceEdgeSampler.__init__
 for name,w in [('constant',np.ones(99,'f4')),('mixed',np.array([1,1e-30,0,.1,.5,2,0,1e-5],'f4'))]:
  acc=0.;prefix=[]
  for x in w:acc+=float(x);prefix.append(acc)
  expected=np.array(prefix)/prefix[-1];actual,_=build(w);assert np.array_equal(actual,expected);checks[name+'_independent_python_serial']=True
 old=run(False);new=run(True);assert same(old,new);checks['allone_actual_outputs_Adam_scaler_RNG_exact']=True
 for membership in [False,True]:
  full=run(True,membership=membership)
  for split in [2,5]:assert same(full,run(True,split,membership));checks[str(membership)+'_serialized_'+('MID' if split==2 else 'EPOCH')]=True
 assert DeviceEdgeSampler.__init__ is method and same(old,run(False));checks['offpath_restored_outputs_RNG']=True
 for name,expected in [('cdf','CDF values identity mismatch'),('weights','CDF weight identity mismatch'),('allone','all_one old CDF mismatch STOP')]:
  w=np.ones(30,'f4');cdf,_=build(w);cdf[0]*=.9
  try:
   with fixed_cdf(cdf,'bad' if name=='cdf' else values_sha(cdf),'bad' if name=='weights' else values_sha(w),require_old_exact=True):
    DeviceEdgeSampler(DeviceArrayDataset(np.arange(2,dtype='f4')[:,None],device='cpu'),np.zeros(30,dtype='i4'),np.ones(30,dtype='i4'),w,2,weighted_edge_sampling=True,positive_target_mode='binary',device='cpu')
  except AssertionError as e:assert str(e)==expected;checks[name+'_fault_rejected']=True
  else:raise AssertionError('fault accepted')
 assert DeviceEdgeSampler.__init__ is method;checks['exception_restoration']=True
 p=O/'card086-cdf-repair/cpu-contracts.json';p.write_text(json.dumps({'PASS':True,'checks':checks,'n_checks':len(checks),'scope':'Actual512-node CPU production sampler,output IDs/labels,AdamW/GradScaler overflow,RNG,MID/EPOCH serialization andoffpath;CUDA/full2M model parity notclaimed.'},indent=2)+'\n');print('CDF REPAIR CPU PASS',len(checks))
if __name__=='__main__':main()
