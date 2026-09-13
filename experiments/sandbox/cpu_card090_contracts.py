"""Bounded CPU contracts; synthetic states, no parent/input tensors loaded."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,copy,json,tempfile,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
import card090_common as C
import card090_resume as V
import run_card090_chain as M
from gpu_card090_preflight import estimate

def main():
 torch.set_num_threads(2);checks=[]
 assert "'card':'090'" in Path(C.__file__).read_text()
 assert C.ARMS==['ranked43','uniform43','uniform44','ranked44'];assert [C.seed(a) for a in C.ARMS]==[43,43,44,44];checks.append('arm order and seed mapping')
 for a in C.ARMS:
  assert C.window(a)==(500000 if a.startswith('ranked') else 0)
  assert C.window(a,512)==(511 if a.startswith('ranked') else 0)
 checks.append('full and scoped small ranked windows')
 with patch.dict(os.environ,{'GROUPED_NEGATIVES':'1'}):
  try:C.environment()
  except AssertionError as e:assert str(e)=='Card090 environment policy mismatch'
  else:raise AssertionError('environment accepted')
 checks.append('environment drift rejected')
 with tempfile.TemporaryDirectory() as tmp:
  td=Path(tmp);rp=td/'radii.npy';np.save(rp,np.ones(8,'f4'))
  with patch.multiple(C,TD=td,sha=lambda p:'hash',source_check=lambda:'runtime',read=lambda p:{'READY':True,'prepared_sha':'hash','runtime_sha':'runtime','parent_sha':'parent'}):
   for arm in C.ARMS:
    ident=C.identity(arm,18,8,td/'graph.npz',rp)
    assert ident['card']=='090' and ident['seed']==C.seed(arm) and ident['rankneg_window']==C.window(arm,8) and ident['protocol_sha']=='hash' and ident['quality_prereg_sha']=='hash'
  checks.append('actual identity constructor four seeds/policies and frozen protocols')
 from card090_state import compare,KEYS
 state={k:None for k in KEYS};state['additional_saved_state']=torch.ones(1)
 compare(state,copy.deepcopy(state));bad=copy.deepcopy(state);bad['additional_saved_state'][0]=2
 try:compare(state,bad)
 except AssertionError as e:assert str(e)=='fullstate mismatch: additional_saved_state'
 else:raise AssertionError('unlisted durable state omission')
 checks.append('full checkpoint field set, including additional state, compared exactly')
 old=C.R.parent/'card075-code'
 for p in (old/'basemap').rglob('*.py'):
  assert p.read_bytes()==(C.R/p.relative_to(old)).read_bytes(),'production bytes changed'
 checks.append('every legacy basemap Python source byte unchanged')
 fit=(Path(__file__).with_name('card090_fit.py')).read_text()
 for token in ['torch.manual_seed(C.seed(arm))','torch.cuda.manual_seed_all(C.seed(arm))','np.random.seed(C.seed(arm))','random_state=C.seed(arm)']:assert token in fit
 assert fit.index('C.validate_ckpt')<fit.index('p=ParametricUMAP.load');checks.append('all seed plumbing and pre-restore identity order')
 core=(C.R/'basemap/pumap/parametric_umap/core.py').read_text();assert 'optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)' in core
 param=torch.nn.Parameter(torch.ones(3));opt=torch.optim.AdamW([param],lr=.0001);assert not opt.state and opt.param_groups[0]['weight_decay']==.01;checks.append('legacy fresh AdamW constructor defaults, no inherited moments')
 for arm in C.ARMS:
  n=8;step=2;param=torch.ones(2);g=torch.Generator().manual_seed(12)
  ident={'arm':arm,'seed':C.seed(arm),'gpu_resident_vram_budget_gb':14.,'fneg_weight':1.,'neg_tanh_gamma':4.,'positive_target_mode':'binary','replay_weight':0.,'rankneg_window':C.window(arm,n)}
  ts={'executed_iters':step,'positive_lr_optimizer_steps':step,'optimizer_steps_succeeded':step,'attempted_batches':4,'amp_overflow_skips':1,'nonfinite_loss_skips':1,'nonfinite_gradient_skips':0}
  ck={'schema':'pumap-ckpt-2026-08-30','epoch':0,'train_stats':ts,'card012_identity':ident,'config':{'batch_size':C.BATCH,'random_state':C.seed(arm),'architecture':'residual_bottleneck','replay_enabled':False,'deriv_enabled':False,'n_epochs':100000,'lr_horizon':int(np.ceil(15*n/int(C.BATCH*.1)))*100000},'model':{'p':param},'optimizer':{'param_groups':[{'params':[0],'lr':.0001,'betas':(.9,.999),'eps':1e-8,'weight_decay':.01,'amsgrad':False}],'state':{0:{'step':torch.tensor(float(step)),'exp_avg':param.clone(),'exp_avg_sq':param.clone()}}},'scheduler':{'base_lrs':[.0001],'last_epoch':0,'_step_count':1,'_last_lr':[.0001],'lr_lambdas':[None]},'scaler':{'scale':65536.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000,'_growth_tracker':1},'torch_rng':g.get_state(),'cuda_rng':[torch.zeros(16,dtype=torch.uint8)],'loader_gen':torch.zeros(16,dtype=torch.uint8),'replay_gen':None,'loader_perm':torch.arange(15*n),'loader_pos_idx':20,'loader_batch_no':4,'loader_rank_of_node':torch.arange(n) if C.window(arm,n) else None,'loader_node_at_rank':torch.arange(n) if C.window(arm,n) else None,'rankneg_scale':(2*C.window(arm,n)/n)**.75 if C.window(arm,n) else None}
  ck.update(mn_gen=torch.zeros(16,dtype=torch.uint8),hold_gen=torch.zeros(16,dtype=torch.uint8),dens_gen=None,deriv_gen=None,**{k:np.random.RandomState(1).get_state() for k in ['mn_rng','dens_rng','hold_rng']})
  V.validate_resume_payload(ck,step,n);checks.append(arm+' synthetic complete legacy payload')
  for key in ['seed','rank','counter','scaler','moment']:
   bad=copy.deepcopy(ck)
   if key=='seed':bad['config']['random_state']=42
   if key=='rank':bad['rankneg_scale']=123.
   if key=='counter':bad['train_stats']['attempted_batches']=5
   if key=='scaler':bad['scaler']['scale']=float('nan')
   if key=='moment':bad['optimizer']['state'][0]['exp_avg'][0]=float('nan')
   try:V.validate_resume_payload(bad,step,n)
   except AssertionError:pass
   else:raise AssertionError('corrupted '+key+' accepted')
   checks.append(arm+' corrupt '+key+' rejected')
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'receipt.json';start=10
  def reject(message):
   try:M.validate_stage_receipt(p,start,'runtime','data')
   except AssertionError as e:assert str(e)==message
   else:raise AssertionError('bad receipt accepted')
  reject('missing stage receipt');p.write_text('{}');os.utime(p,ns=(1,1));reject('stale stage receipt')
  for row,message in [({},'stage receipt lacks explicit PASS'),({'PASS':True,'runtime_sha':'wrong'},'stage receipt runtime mismatch'),({'PASS':True,'runtime_sha':'runtime','data_manifest_sha':'wrong'},'stage receipt calibration mismatch')]:
   p.write_text(json.dumps(row));reject(message)
  p.write_text(json.dumps({'PASS':True,'runtime_sha':'runtime','data_manifest_sha':'data'}));M.validate_stage_receipt(p,start,'runtime','data');checks.append('missing stale nonPASS wrong-runtime wrong-data receipts reject, fresh PASS accepts')
 e=estimate({'500':{'seconds':20.,'serialization_s':1.},'3500':{'seconds':70.,'serialization_s':1.}});assert e['complete_arm_s']>1200;checks.append('measured slope setup and repeated serialization reserve')
 C.write(C.O/'card090-readiness/cpu-contracts.json',{'PASS':True,'n_checks':len(checks),'checks':checks,'scope':'CPU synthetic tensors only; actual historical payload and full parent identity/device replay remain root-released gates.'});print('CPU PASS',len(checks))
if __name__=='__main__':main()
