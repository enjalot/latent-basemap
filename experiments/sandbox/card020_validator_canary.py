"""CPU synthetic validator controls; never mistaken for trained outputs."""
import copy,tempfile,shutil
import run_card020 as r
from run_card020 import *
def main():
 torch.set_num_threads(2);checks={};original=r.OUT
 with tempfile.TemporaryDirectory(prefix='card020-validator-',dir=OC) as name:
  r.OUT=Path(name);ad=identity();write(r.OUT/'admission.json',ad);parent=torch.load(PARENT,map_location='cpu',weights_only=False);m=json.loads((DATA/'manifest.json').read_text())
  for i,step in enumerate(SNAPS):
   c=copy.deepcopy(parent);c['identity']=ad;c['successful_steps']=step
   for s in c['optimizer_state_dict']['state'].values():s['step'].fill_(step)
   next(iter(c['model_state_dict'].values())).add_(.001*(i+1));torch.save(c,r.OUT/f'step-{step}.pt')
  torch.save({'model_state_dict':c['model_state_dict'],'identity':ad,'successful_steps':80000,'center':m['center'],'span':m['span']},r.OUT/'model.pt')
  b=base();done={'identity':ad,'start_receipt':{'model_sha':b['parent_model_sha'],'optimizer_sha':b['parent_optimizer_sha'],'batch_rng_sha':b['parent_batch_rng_sha']},'successful_steps':80000,'endpoint_state_sha':objsha(c['model_state_dict']),'endpoint_file_sha':sha(r.OUT/'model.pt'),'peak_allocated_gib':5.,'global_used_gib':6.,'input_file_sha':b['input_file_sha']};write(r.OUT/'complete.json',done);checks['valid_fixture']=validate_endpoint()['PASS']
  target=r.OUT/'step-80000.pt';pristine=target.read_bytes()
  def negative(label,mutate):
   z=torch.load(target,map_location='cpu',weights_only=False);mutate(z);torch.save(z,target)
   try:validate_endpoint();checks[label]=False
   except (AssertionError,ValueError):checks[label]=True
   target.write_bytes(pristine)
  negative('wrong_step_reject',lambda z:z.update(successful_steps=79999))
  negative('nonfinite_weights_reject',lambda z:next(iter(z['model_state_dict'].values())).fill_(float('nan')))
  negative('wrong_admission_reject',lambda z:z['identity'].update(lr=.002))
  negative('wrong_optimizer_dose_reject',lambda z:next(iter(z['optimizer_state_dict']['state'].values()))['step'].fill_(79999))
  negative('wrong_snapshot_payload_reject',lambda z:next(iter(z['model_state_dict'].values())).add_(.01))
  endpoint=r.OUT/'model.pt';ep=torch.load(endpoint,map_location='cpu',weights_only=False);ep_original=endpoint.read_bytes();ep['span']*=2;torch.save(ep,endpoint);done['endpoint_file_sha']=sha(endpoint);write(r.OUT/'complete.json',done)
  try:validate_endpoint();checks['wrong_native_conversion_reject']=False
  except AssertionError:checks['wrong_native_conversion_reject']=True
  endpoint.write_bytes(ep_original);done['endpoint_file_sha']=sha(endpoint);write(r.OUT/'complete.json',done)
  old=target.stat().st_mtime;os.utime(target,(0,0))
  try:validate_endpoint();checks['stale_snapshot_reject']=False
  except AssertionError:checks['stale_snapshot_reject']=True
  os.utime(target,(old,old));target.unlink()
  try:validate_endpoint();checks['missing_snapshot_reject']=False
  except FileNotFoundError:checks['missing_snapshot_reject']=True
 r.OUT=original;write(OC/'card020-validator-canary.json',{'PASS':all(checks.values()),'checks':checks,'scope':'Synthetic CPU fixtures only, no actual training claims'});assert all(checks.values());print(checks)
if __name__=='__main__':main()
