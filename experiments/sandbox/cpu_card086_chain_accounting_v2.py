"""Execute real chain main with in-memory ledger and mocked stages; never launch."""
from unittest.mock import patch
import copy
import card086_common as C
import card086_budget as B
import run_card086_chain as M

def run(fail=False):
 clock=[1000.];ledger={'batch_spent_s':0.,'arm_spent_s':dict.fromkeys(C.ARMS,0.)};window={'spent_s':0.};events=[];outputs={}
 def transact(tag,seconds,arm=None,kind='reservation',check=False):
  ledger['batch_spent_s']+=seconds;window['spent_s']+=seconds
  if arm:ledger['arm_spent_s'][arm]+=seconds
  events.append((tag,kind,seconds,arm))
 def read(path):
  if path==B.L:return copy.deepcopy(ledger)
  if path==B.W:return copy.deepcopy(window)
  return {'READY':True}
 def stage(tag,script,cap,args=(),arm=None,receipt_path=None):
  clock[0]+=5 # actual controller work outside the stage
  elapsed=139.5 if tag.startswith('preflight-') else (1300. if arm else 10.)
  transact(tag,cap,arm);clock[0]+=elapsed;M.STAGE_TIME+=elapsed;transact(tag,elapsed-cap,arm,'settlement')
  if fail and tag=='preflight-membership':raise RuntimeError('stage fault')
  if tag.startswith('preflight-'):
   batch={'negative_source_sha':'s','negative_target_sha':'t','sampler_rng_sha':'r','positive_slots':1638,'negative_slots':14746}
   M.STAGE_RECEIPTS[tag]={'receipt':{'PASS':True,'runtime_sha':'runtime','data_manifest_sha':'data','estimate':{'complete_arm_s':1451.33},'fits':{n:{'sampler':{'actual_first_batch':batch}} for n in ['500','3500']}}}
  return elapsed
 def available(arm=None):return min(3600-ledger['batch_spent_s'],1600-ledger['arm_spent_s'][arm] if arm else 3600)
 M.STAGE_TIME=0.;M.STAGE_RECEIPTS={}
 with patch.multiple(C,read=read,write=lambda p,r:outputs.update({str(p):copy.deepcopy(r)}),source_check=lambda:'runtime',input_check=lambda:None,sha=lambda p:'data'),patch.multiple(B,transact=transact,available=available,END=1e20),patch.multiple(M,verify=lambda:None,verify_external_leases=lambda:None,stage=stage),patch.object(M.time,'monotonic',lambda:clock[0]):
  if fail:
   try:M.main()
   except RuntimeError as e:assert str(e)=='stage fault'
   else:raise AssertionError('stage failure hidden')
  else:M.main()
 assert abs(ledger['batch_spent_s']-(clock[0]-1000))<1e-8,'controller or stage accounting differs from actual elapsed'
 if not fail:
  admission=outputs[str(C.O/'card086-preflight.json')];assert admission['PASS'];assert admission['ledger_at_admission']['arm_spent_s']==dict.fromkeys(C.ARMS,139.5)
  assert admission['ledger_at_admission']['batch_spent_s']==319.,admission['ledger_at_admission']
  assert outputs[str(C.O/'card086-execution.json')]['status']=='TRAINED_VALIDATED'
 return {'actual_total_s':ledger['batch_spent_s'],'events':events}
r={'PASS':True,'n_checks':2,'success':run(),'stage_failure':run(True),'scope':'Real chain main, simulated clock/stages, in-memory ledger; verifies settled admission139.5 not240 and exact controller+stage settlement including exception. No GPU, files or live ledger mutation inside simulation.'}
C.write(C.O/'card086-cdf-repair-v2/chain-accounting-contracts.json',r);print('CHAIN ACCOUNTING PASS2')
