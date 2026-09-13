"""Actual chain main with mocked process stages, clock and in-memory ledgers."""
from unittest.mock import patch
import copy,math,tempfile
from pathlib import Path
import card089_common as C
import card089_budget as B
import run_card089_chain as M

def run(failure=None,estimate=1451.33):
 clock=[1000.];ledger={'batch_spent_s':17.,'arm_spent_s':dict.fromkeys(C.ARMS,0.)};window={'spent_s':91.};events=[];outputs={};stages=[]
 def transact(tag,seconds,arm=None,kind='reservation',check=False):
  if failure=='reservation' and check:clock[0]+=2;raise RuntimeError('reservation')
  ledger['batch_spent_s']+=seconds;window['spent_s']+=seconds
  if arm:ledger['arm_spent_s'][arm]+=seconds
  events.append((tag,kind,seconds,arm))
 def read(path):
  if path==B.L:return copy.deepcopy(ledger)
  if path==B.W:return copy.deepcopy(window)
  return {'READY':True}
 def stage(tag,script,cap,args=(),arm=None,receipt_path=None):
  stages.append(tag);clock[0]+=5
  elapsed=139.5 if tag.startswith('preflight-') else (1300. if arm else 10.)
  transact(tag,cap,arm)
  if tag.startswith('preflight-') and estimate==1451.33:
   assert ledger['arm_spent_s'][arm]+estimate>1600,'regression does not reproduce active reservation STOP'
  clock[0]+=elapsed;M.STAGE_TIME+=elapsed;transact(tag,elapsed-cap,arm,'settlement')
  if failure==tag:raise RuntimeError(tag)
  if tag.startswith('preflight-'):
   batch={'negative_source_sha':'s','negative_target_sha':'t','sampler_rng_sha':'r','positive_slots':1638,'negative_slots':14746}
   M.STAGE_RECEIPTS[tag]={'receipt':{'PASS':True,'runtime_sha':'runtime','data_manifest_sha':'data','measurement_only':True,'estimate':{'complete_arm_s':estimate},'fits':{n:{'sampler':{'actual_first_batch':batch}} for n in ['500','3500']}}}
  return elapsed
 def available(arm=None):return min(3600-ledger['batch_spent_s'],1600-ledger['arm_spent_s'][arm] if arm else 3600)
 def verify():
  clock[0]+=7
  if failure=='verify':raise RuntimeError('verify')
 def leases():
  clock[0]+=3
  if failure=='leases':raise RuntimeError('leases')
 def inputs():
  clock[0]+=4
  if failure=='inputs':raise RuntimeError('inputs')
 caught=None
 with patch.multiple(C,read=read,write=lambda p,r:outputs.update({str(p):copy.deepcopy(r)}),source_check=lambda:'runtime',input_check=inputs,sha=lambda p:'data'),patch.multiple(B,transact=transact,available=available,END=1e20),patch.multiple(M,verify=verify,verify_external_leases=leases,stage=stage),patch.object(M.time,'monotonic',lambda:clock[0]):
  try:M.main()
  except (RuntimeError,AssertionError) as e:caught=str(e)
 if failure:assert caught==failure,(failure,caught)
 elif math.isfinite(estimate):assert caught is None,caught
 else:assert caught=='invalid full-dose estimate'
 assert abs((ledger['batch_spent_s']-17.)-(clock[0]-1000))<1e-8,'occupancy not exactly charged'
 assert abs((window['spent_s']-91.)-(clock[0]-1000))<1e-8,'rolling ledger occupancy mismatch'
 if stages:assert stages[0]=='cdf_constructor_diagnostic','diagnostic must precede other stages'
 if failure is None and math.isfinite(estimate):
  admission=outputs[str(C.O/'card089-preflight.json')]
  assert admission['ledger_at_admission']['arm_spent_s']==dict.fromkeys(C.ARMS,139.5),'preflight not settled'
  if estimate==1451.33:
   assert admission['PASS'];assert outputs[str(C.O/'card089-execution.json')]['status']=='TRAINED_VALIDATED'
  else:assert not admission['PASS'] and not any(x in C.ARMS for x in stages),'over-cap admission launched production'
 return {'charged_s':ledger['batch_spent_s']-17.,'elapsed_s':clock[0]-1000,'events':events,'stages':stages,'caught':caught}

def stage_hash_failure():
 clock=[0.];events=[];writes={};M.STAGE_TIME=0.
 def fail_hash():clock[0]+=7;raise RuntimeError('source hash')
 def transact(tag,seconds,arm=None,**kw):events.append(seconds)
 with patch.multiple(M,verify=lambda:'release'),patch.multiple(B,available=lambda a:1000,transact=transact),patch.multiple(C,source_check=fail_hash,write=lambda p,r:writes.update({str(p):r})),patch.object(M.time,'monotonic',lambda:clock[0]),patch.object(M.subprocess,'Popen',side_effect=AssertionError('must not launch')):
  try:M.stage('test','never.py',100,receipt_path=Path('/tmp/none'))
  except RuntimeError as e:assert str(e)=='source hash'
  else:raise AssertionError('hash failure hidden')
 assert events==[100,-93] and M.STAGE_TIME==7
 assert writes[str(C.O/'card089-stage-test.json')]['PASS'] is False
 return {'PASS':True,'charged_s':sum(events),'no_process_launched':True}

def main():
 results={'success_active_reservation_regression':run(),'over_cap_stop':run(estimate=1600.),'invalid_estimate_stop':run(estimate=float('nan'))}
 for fault in ['verify','leases','reservation','inputs','cdf_constructor_diagnostic','preflight-rank_count_control','reciprocal','rank_count_control']:results[fault]=run(fault)
 results['stage_prelaunch_hash_failure']=stage_hash_failure()
 C.write(C.O/'card089-final-readiness/chain-accounting-contracts.json',{'PASS':True,'n_checks':len(results),'results':results,'scope':'Actual main plus actual stage prelaunch failure; mocked clocks/process stages and in-memory ledgers. Initial verify/lease/reservation/input errors charged, all stage/controller actuals settled, old spend retained, no GPU/live ledger writes.'});print('CHAIN ACCOUNTING PASS',len(results))
if __name__=='__main__':main()
