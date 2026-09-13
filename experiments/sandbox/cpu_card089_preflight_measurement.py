"""Execute real preflight main without GPU; active ledger is deliberately unreadable."""
from unittest.mock import patch
from contextlib import contextmanager
import sys
import card089_common as C
import gpu_card089_preflight as P

def main():
 clock=[100.];out={};doses=[]
 @contextmanager
 def probe(record,weight):
  record.update(epoch_construction_s=[.1],cdf={'positive_zero_width_probability':0.})
  yield
 def fit(arm,dose,dest,**kwargs):
  doses.append(dose);clock[0]+=dose*.02+1
  return object(),{}, {'train_stats':{'positive_lr_optimizer_steps':dose},'global_vram_GiB':1.,'support_fractions':{},'exposure':{}}
 def save(*args,**kwargs):clock[0]+=.1
 with patch.multiple(C,require_release=lambda:{'cdf_lost_probability_cap':1e-10},input_check=lambda:None,weight_sha=lambda p:'weights',source_check=lambda:'runtime',sha=lambda p:'data',read=lambda p:(_ for _ in ()).throw(AssertionError('active ledger must not be read')),write=lambda p,r:out.update(r)),patch.multiple(P,fit=fit,probe=probe),patch.object(P.time,'monotonic',lambda:clock[0]),patch.object(P.torch,'save',save),patch.object(P.torch.cuda,'empty_cache',lambda:None),patch.object(sys,'argv',['preflight',C.ARMS[0]]):P.main()
 assert doses==[500,3500] and out['PASS'] and out['measurement_only'] and 'stage_cap' not in out['checks']
 C.write(C.O/'card089-final-readiness/preflight-measurement-contracts.json',{'PASS':True,'n_checks':4,'actual_main_called':True,'doses':doses,'measurement_only':True,'no_active_ledger_read':True,'finite_positive_full_dose_estimate':out['estimate'],'scope':'Actual preflight main with mocked fits/timers/serialization; no CUDA, feature access or live ledger mutation.'});print('PREFLIGHT MEASUREMENT PASS4')
if __name__=='__main__':main()
