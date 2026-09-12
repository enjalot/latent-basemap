"""CPU algebra checks plus genuine GPU parent/child resume and throughput admission."""
import copy,json,sys,time
from run_card020 import *

def main(device):
 torch.set_num_threads(2);checks={};ident=identity(True);dest=OUT/('canary-'+device);dest.mkdir(parents=True,exist_ok=True);t=time.monotonic()
 if device=='cuda':
  assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;validate_inputs();X,Y,rows,grid=gpu_data();batch=8192;dose=240;mid=120
 else:
  assert device=='cpu';torch.manual_seed(16020);X=torch.randn(64,1536);Y=torch.rand(64,2);rows=torch.arange(64);grid=torch.ones(16,16);batch=8;dose=4;mid=2
 def run(model,opt,gen,start,end,save=False):
  ts=time.monotonic()
  for k in range(start,end):train_step(model,opt,gen,X,Y,rows,grid,'compact_l1',batch=batch)
  if device=='cuda':torch.cuda.synchronize()
  if save:checkpoint(dest/'middle.pt',model,opt,gen,20000+end,ident)
  return time.monotonic()-ts
 m,o,g,s,r=init_state(device,ident);checks['parent_model_hash']=r['model_sha']==base()['parent_model_sha'];checks['parent_optimizer_hash']=r['optimizer_sha']==base()['parent_optimizer_sha']
 if device=='cuda':checks['actual_parent_cuda_batch_rng']=r['batch_rng_sha']==base()['parent_batch_rng_sha']
 first_time=run(m,o,g,0,dose);target={'model':objsha(m.state_dict()),'optimizer':objsha(o.state_dict()),'rng':objsha(g.get_state())}
 del m,o,g
 m,o,g,s,r=init_state(device,ident);run(m,o,g,0,mid,True);middle=torch.load(dest/'middle.pt',map_location='cpu',weights_only=False);checks['genuine_intermediate']=middle['successful_steps']==20000+mid and 0<mid<dose
 del m,o,g
 m,o,g,s,r=init_state(device,ident,dest/'middle.pt');tail_time=run(m,o,g,mid,dose);checks['resumed_model_bitwise']=objsha(m.state_dict())==target['model'];checks['resumed_optimizer_bitwise']=objsha(o.state_dict())==target['optimizer'];checks['resumed_batch_rng_bitwise']=objsha(g.get_state())==target['rng']
 checks['model_changed']=objsha(m.state_dict())!=base()['parent_model_sha']
 for name,mutate,resume in [('wrong_parent',lambda z:z['parent'].update(parent_file_sha='0'*64),None),('wrong_lr',lambda z:z.update(lr=.0002),None),('wrong_child_identity',lambda z:z.update(total_successful_steps=80001),dest/'middle.pt')]:
  z=copy.deepcopy(ident);mutate(z)
  try:init_state(device,z,resume);checks[name+'_reject']=False
  except ValueError as exc:checks[name+'_reject']='mismatch' in str(exc)
 # Corrupt an otherwise valid child's identity to exercise resume-bound check, not outer config check.
 bad=copy.deepcopy(middle);bad['identity']['parent']['parent_file_sha']='f'*64;torch.save(bad,dest/'bad-child.pt')
 try:init_state(device,ident,dest/'bad-child.pt');checks['corrupt_child_identity_reject']=False
 except ValueError as exc:checks['corrupt_child_identity_reject']='child admission identity mismatch' in str(exc)
 result={'PASS':all(checks.values()),'device':device,'checks':checks,'true_parent_boundary':20000,'target_step':20000+dose,'resume_step':20000+mid,'wall_s':time.monotonic()-t,'identity':ident,'scope':'Exact original GPU model/Adam/RNG; full fixed input bank' if device=='cuda' else 'CPU model/Adam parent and algebra resume with separate CPU RNG and synthetic inputs; GPU canary validates actual CUDA RNG/input path'}
 if device=='cuda':
  rate=(dose-mid)/tail_time;result.update({'steady_updates_per_s':rate,'measured_tail_s':tail_time,'remaining_fit_estimate_s':60000/rate,'admission_estimate_s':60000/rate*1.25+30,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30});checks['VRAM_lt30']=result['peak_allocated_gib']<30;result['PASS']=all(checks.values())
 write(OC/f'card020-{device}-canary.json',result);assert result['PASS'],checks;print(json.dumps({k:v for k,v in result.items() if k!='identity'},indent=2))
if __name__=='__main__':main(sys.argv[1])
