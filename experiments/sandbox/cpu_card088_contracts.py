"""CPU identity/resource/receipt negative controls. No GPU or ledger mutation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import tempfile,time,json
import card088_common as C
import card088_budget as B
from run_card088_chain import validate_stage_receipt

def reject(f,msg):
 try:f()
 except AssertionError as e:assert str(e)==msg,(str(e),msg);return True
 raise AssertionError('negative control accepted: '+msg)
def main():
 checks={}
 for key in ['arm','dose','lr','graph_sha','weight_values_sha','endpoints_sha','weighted_edge_sampling','uniform_with_replacement','parent_sha','warm_sha','reference_sha','data_manifest_sha','runtime_manifest_sha','positive_target_mode','negative_policy','positive_support_law','logical_epoch_draws','cdf_side','endpoint_columns']:
  checks['wrong_'+key]=reject(lambda:C.assert_identity({key:'old'},{key:'new'}),'card088 identity mismatch: '+key)
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'receipt.json';t=time.time_ns();checks['missing']=reject(lambda:validate_stage_receipt(p,t,'r','d'),'missing stage receipt');good={'PASS':True,'runtime_sha':'r','data_manifest_sha':'d'};p.write_text(json.dumps(good));os.utime(p,ns=(t,t));assert validate_stage_receipt(p,t,'r','d')==good;checks['fresh']=True
  os.utime(p,ns=(t-1,t-1));checks['stale']=reject(lambda:validate_stage_receipt(p,t,'r','d'),'stale stage receipt')
  for k,v,msg in [('PASS',False,'stage receipt lacks explicit PASS'),('runtime_sha','wrong','stage receipt runtime mismatch'),('data_manifest_sha','wrong','stage receipt calibration mismatch')]:
   p.write_text(json.dumps(dict(good,**{k:v})));checks['receipt_'+k]=reject(lambda:validate_stage_receipt(p,0,'r','d'),msg)
 checks['budget']=B.LIMITS['card_gpu_s']==4800 and B.LIMITS['stage_gpu_s']=={'original15':1600,'mixture':1600}
 checks['dose_recipe']=C.DOSE==60000 and C.LR=={'original15':.0001,'mixture':.0001} and C.BATCH==16384
 checks['no_release']=reject(C.require_release,'NO ROOT GPU RELEASE')
 import torch
 from card088_resume import validate_resume_payload
 ck=torch.load(C.R.parent/'card075-train/uniform/ckpts/ckpt-step60000.pt',map_location='cpu',weights_only=False);ck['card012_identity']=dict(ck['card012_identity'],arm='mixture',weighted_edge_sampling=True,uniform_with_replacement=False)
 checks['legacy_unweighted_state_rejected']=reject(lambda:validate_resume_payload(ck,60000),'actual weighted pipeline mismatch')
 ck['card012_identity']['arm']='original15';checks['zero_control_resume_rejected']=reject(lambda:validate_resume_payload(ck,60000),'zero control weight in resume PERM')
 assert all(checks.values());C.write(C.O/'card088-cpu-contracts.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'scope':'CPU contracts only; actual weighted device resume and numerical mass acceptance pending root release.'});print('CPU CONTRACTS PASS',len(checks))
if __name__=='__main__':main()
