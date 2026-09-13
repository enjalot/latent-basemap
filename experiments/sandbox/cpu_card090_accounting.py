"""Actual main and stage fault controls; no live ledger, subprocess or GPU access."""
from unittest.mock import patch
import copy,math
from pathlib import Path
import card090_common as C
import card090_budget as B
import run_card090_chain as M

def run(failure=None,estimate=1300.,remaining=30000):
 clock=[1000.];ledger={'batch_spent_s':17.,'shared_spent_s':17.,'arm_spent_s':dict.fromkeys(C.ARMS,0.)};window={'spent_s':91.};outputs={};events=[];stages=[]
 def transact(tag,seconds,arm=None,kind='reservation',check=False):
  if failure=='reservation' and check:clock[0]+=2;raise RuntimeError('reservation')
  ledger['batch_spent_s']+=seconds;window['spent_s']+=seconds
  if arm:ledger['arm_spent_s'][arm]+=seconds
  else:ledger['shared_spent_s']+=seconds
  events.append((tag,seconds,arm))
 def read(p):
  if p==B.L:return copy.deepcopy(ledger)
  if p==B.W:return copy.deepcopy(window)
  return {'READY':True}
 def stage(tag,script,cap,args=(),arm=None,receipt_path=None):
  stages.append(tag);clock[0]+=5;elapsed=110. if tag.startswith('preflight') else 1000. if arm else 10.
  transact(tag,cap,arm)
  if tag.startswith('preflight'):assert arm is None,'preflight must be shared, not production dose'
  clock[0]+=elapsed;M.STAGE_TIME+=elapsed;transact(tag,elapsed-cap,arm,'settlement')
  if failure==tag:raise RuntimeError(tag)
  if tag.startswith('preflight'):
   M.STAGE_RECEIPTS[tag]={'receipt':{'PASS':True,'runtime_sha':'runtime','data_manifest_sha':'data','measurement_only':True,'estimate':{'complete_arm_s':estimate}}}
  return elapsed
 def verify():
  clock[0]+=7
  if failure=='verify':raise RuntimeError('verify')
 def leases():
  clock[0]+=3
  if failure=='leases':raise RuntimeError('leases')
 def inputs():
  clock[0]+=4
  if failure=='inputs':raise RuntimeError('inputs')
 def available(arm=None):return min(6500-ledger['batch_spent_s'],1400-ledger['arm_spent_s'][arm] if arm else 900-ledger['shared_spent_s'])
 caught=None
 with patch.multiple(C,read=read,write=lambda p,r:outputs.update({str(p):copy.deepcopy(r)}),source_check=lambda:'runtime',input_check=inputs,sha=lambda p:'data'),patch.multiple(B,transact=transact,available=available,END=1000+remaining),patch.multiple(M,verify=verify,verify_external_leases=leases,stage=stage),patch.object(M.time,'monotonic',lambda:clock[0]),patch.object(M.time,'time',lambda:clock[0]):
  try:M.main()
  except (RuntimeError,AssertionError) as e:caught=str(e)
 if failure:assert caught==failure,(failure,caught)
 elif not math.isfinite(estimate):assert caught=='invalid full-dose estimate'
 else:assert caught is None,caught
 assert abs(ledger['batch_spent_s']-17-(clock[0]-1000))<1e-8,'unaccounted occupancy'
 assert abs(window['spent_s']-91-(clock[0]-1000))<1e-8,'rolling ledger drift'
 assert abs(ledger['batch_spent_s']-ledger['shared_spent_s']-sum(ledger['arm_spent_s'].values()))<1e-8
 if failure is None and math.isfinite(estimate):
  adm=outputs[str(C.O/'card090-preflight.json')];assert not any(adm['ledger_at_admission']['arm_spent_s'].values()),'shared preflight charged to production'
  expected=estimate<=1400 and remaining>10000
  assert adm['PASS']==expected
  if expected:assert stages[-4:]==C.ARMS
  else:assert not any(a in stages for a in C.ARMS)
 return {'charged':ledger['batch_spent_s']-17.,'elapsed':clock[0]-1000,'stages':stages,'caught':caught}

def hash_failure():
 clock=[0.];events=[];M.STAGE_TIME=0.
 def source():clock[0]+=7;raise RuntimeError('hash')
 with patch.object(M,'verify',lambda:'release'),patch.multiple(B,available=lambda a:1000,transact=lambda tag,s,*a,**kw:events.append(s)),patch.multiple(C,source_check=source,write=lambda *a:None),patch.object(M.time,'monotonic',lambda:clock[0]),patch.object(M.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
  try:M.stage('fault','none.py',100,receipt_path=Path('/tmp/none'))
  except RuntimeError as e:assert str(e)=='hash'
  else:raise AssertionError('fault passed')
 assert events==[100,-93] and M.STAGE_TIME==7;return True

def main():
 results={'success':run(),'arm_limit':run(estimate=1401),'deadline_CPU_reserve':run(remaining=6500),'invalid_estimate':run(estimate=float('nan'))}
 for f in ['verify','leases','reservation','inputs','history_replay','prepare','graph_canary','preflight-ranked43','preflight-uniform43',*C.ARMS]:results[f]=run(f)
 results['stage_hash_failure']=hash_failure()
 # Reproduce reservation versus settled-actual shared guard, using actual admission function.
 l={'batch_spent_s':740.,'shared_spent_s':740.,'arm_spent_s':dict.fromkeys(C.ARMS,0.)};w={'spent_s':100.};e=dict.fromkeys(C.ARMS,1300.)
 assert M.settled_admission_checks(l,w,e,10000)['shared_cap']
 l['shared_spent_s']+=100;l['batch_spent_s']+=100
 assert not M.settled_admission_checks(l,w,e,10000)['shared_cap'];results['active_reservation_vs_actual_shared']=True
 C.write(C.O/'card090-readiness/accounting-contracts.json',{'PASS':True,'n_checks':len(results),'results':results,'scope':'Actual main and stage hash-failure path, mocked clock/ledgers/processes; no GPU or live accounting mutation.'});print('ACCOUNTING PASS',len(results))
if __name__=='__main__':main()
