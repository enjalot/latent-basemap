"""CPU identity/resource/receipt negative controls. No GPU or ledger mutation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import tempfile,time,json
import card086_common as C
import card086_budget as B
from run_card086_chain import validate_stage_receipt

def reject(f,msg):
 try:f()
 except AssertionError as e:assert str(e)==msg,(str(e),msg);return True
 raise AssertionError('negative control accepted: '+msg)
def main():
 checks={}
 for key in ['cdf_algorithm','cdf_manifest_sha','cdf_values_sha','cdf_file_sha','cdf_normalization','cdf_side','arm','dose','lr','graph_sha','weight_values_sha','endpoints_sha','weighted_edge_sampling','uniform_with_replacement','parent_sha','warm_sha','reference_sha','data_manifest_sha','runtime_manifest_sha','positive_target_mode','negative_policy']:
  checks['wrong_'+key]=reject(lambda:C.assert_identity({key:'old'},{key:'new'}),'card086 identity mismatch: '+key)
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'receipt.json';t=time.time_ns();checks['missing']=reject(lambda:validate_stage_receipt(p,t,'r','d'),'missing stage receipt');good={'PASS':True,'runtime_sha':'r','data_manifest_sha':'d'};p.write_text(json.dumps(good));os.utime(p,ns=(t,t));assert validate_stage_receipt(p,t,'r','d')==good;checks['fresh']=True
  os.utime(p,ns=(t-1,t-1));checks['stale']=reject(lambda:validate_stage_receipt(p,t,'r','d'),'stale stage receipt')
  for k,v,msg in [('PASS',False,'stage receipt lacks explicit PASS'),('runtime_sha','wrong','stage receipt runtime mismatch'),('data_manifest_sha','wrong','stage receipt calibration mismatch')]:
   p.write_text(json.dumps(dict(good,**{k:v})));checks['receipt_'+k]=reject(lambda:validate_stage_receipt(p,0,'r','d'),msg)
 checks['budget']=B.LIMITS['card_gpu_s']==3600 and B.LIMITS['stage_gpu_s']=={'all_one':1600,'membership':1600}
 checks['dose_recipe']=C.DOSE==60000 and C.LR=={'all_one':.0001,'membership':.0001} and C.BATCH==16384
 checks['old_release_rejected']=reject(C.require_release,'release runtime mismatch')
 import torch
 from card086_resume import validate_resume_payload
 ck=torch.load(C.R.parent/'card075-train/uniform/ckpts/ckpt-step60000.pt',map_location='cpu',weights_only=False);ck['card012_identity']=dict(ck['card012_identity'],arm='all_one',weighted_edge_sampling=True,uniform_with_replacement=False)
 checks['legacy_unweighted_state_rejected']=reject(lambda:validate_resume_payload(ck,60000),'actual weighted pipeline mismatch')
 assert all(checks.values());C.write(C.O/'card086-cdf-repair/runtime-contracts.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'scope':'CPU contracts only; actual weighted device resume and numerical mass acceptance pending root release.'});print('CPU CONTRACTS PASS',len(checks))
if __name__=='__main__':main()
