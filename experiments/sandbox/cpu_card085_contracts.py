"""CPU negative controls; no CUDA, ledger writes or historical artifact mutation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import tempfile,json,time
from pathlib import Path
import card085_common as C
from run_card085_chain import validate_stage_receipt
import card085_budget as B

def rejected(fn,expected):
 try:fn()
 except AssertionError as e:assert str(e)==expected,(str(e),expected);return True
 raise AssertionError('negative control accepted: '+expected)
def main():
 checks={}
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'receipt.json';t=time.time_ns()
  checks['missing_receipt']=rejected(lambda:validate_stage_receipt(p,t,'runtime','history'),'missing stage receipt')
  base={'PASS':True,'runtime_sha':'runtime','history_cpu_sha':'history'};p.write_text(json.dumps(base));os.utime(p,ns=(t,t));assert validate_stage_receipt(p,t,'runtime','history')==base;checks['valid_receipt']=True
  os.utime(p,ns=(t-1,t-1));checks['stale_receipt']=rejected(lambda:validate_stage_receipt(p,t,'runtime','history'),'stale stage receipt')
  for key,bad,msg in [('PASS',False,'stage receipt lacks explicit PASS'),('runtime_sha','wrong','stage receipt runtime mismatch'),('history_cpu_sha','wrong','stage receipt calibration mismatch')]:
   p.write_text(json.dumps(dict(base,**{key:bad})));checks['wrong_receipt_'+key]=rejected(lambda:validate_stage_receipt(p,0,'runtime','history'),msg)
 for key in ['arm','dose','stage_full_dose','lr','lr_schedule','graph_sha','radii_sha','warm_sha','parent_sha','original_untrained_sha','negative_policy','rankneg_window','fneg_weight','neg_tanh_gamma','source_files','runtime_manifest_sha','optimizer_reset','validation_rows','budget_addendum_sha']:
  expected={key:'original'};checks['wrong_identity_'+key]=rejected(lambda:C.assert_identity({key:'wrong'},expected),'card085 identity mismatch: '+key)
 checks['identity_keys']=rejected(lambda:C.assert_identity({}, {'arm':'fresh'}),'card085 identity mismatch: keys')
 checks['full_doses']=C.DOSES=={'fresh':400000,'finish':60000} and C.LR=={'fresh':.001,'finish':.0001}
 checks['limits']=B.LIMITS['card_gpu_s']==10800 and B.LIMITS['stage_gpu_s']=={'fresh':8700,'finish':1800}
 checks['accounting_stage']=B.allocation(13,'fresh')=={'fresh':13,'finish':0.} and B.allocation(13,'finish')=={'fresh':0.,'finish':13}
 checks['shared_counted_only_card']=B.allocation(13)=={'fresh':0.,'finish':0.}
 import torch,copy
 from card085_resume import validate_resume_payload
 ck=torch.load(C.R.parent/'card075-train/uniform/ckpts/ckpt-step60000.pt',map_location='cpu',weights_only=False)
 ck['card012_identity']=dict(ck['card012_identity'],arm='finish')
 validate_resume_payload(ck,60000);checks['actual_full_state_schema']=True
 for key,bad,msg in [('scaler',{},'missing AMP scaler continuation state'),('loader_gen',None,'device loader RNG schema'),('loader_pos_idx',-1,'PERM cursor'),('loader_rank_of_node',torch.zeros(1),'uniform policy unexpectedly retains ranking')]:
  altered=dict(ck);altered[key]=bad
  checks['resume_corrupt_'+key]=rejected(lambda:validate_resume_payload(altered,60000),msg)
 from unittest.mock import patch
 with tempfile.TemporaryDirectory() as empty, patch.object(C,'O',Path(empty)):
  checks['no_release']=rejected(C.require_release,'NO ROOT GPU RELEASE')
 checks['approved_reallocation']=C.read(C.O/'card085-budget-reallocation-addendum.json')['new_stage_caps_s']==B.LIMITS['stage_gpu_s']
 assert all(checks.values())
 C.write(C.O/'card085-cpu-contracts.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'device_tests':'NOT_RUN_NO_ROOT_RELEASE'})
 print('PASS',len(checks))
if __name__=='__main__':main()
