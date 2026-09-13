"""No device/ledger writes: actual admission helper and scoped cache fault controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import tempfile,json,copy,hashlib
from pathlib import Path
import card086_common as C
import card086_canary_admission as A
from run_card086_chain import settled_admission_checks
checks={}
def reject(f,msg):
 try:f()
 except AssertionError as e:assert str(e)==msg,(str(e),msg)
 else:raise AssertionError('fault accepted')
ledger={'batch_spent_s':139.5,'arm_spent_s':{'all_one':139.5,'membership':0.}}
est={'all_one':1451.33,'membership':1400.};window={'spent_s':0.}
assert all(settled_admission_checks(ledger,window,est,10000).values());checks['settled1590_83_pass']=True
reserved=copy.deepcopy(ledger);reserved['arm_spent_s']['all_one']=240
assert not settled_admission_checks(reserved,window,est,10000)['all_one'];checks['active1691_33_would_fail']=True
for name,l,w,t,key in [('arm',{'batch_spent_s':139.5,'arm_spent_s':{'all_one':148.68,'membership':0}},window,10000,'all_one'),('card',dict(ledger,batch_spent_s=800),window,10000,'card_cap'),('window',ledger,{'spent_s':165000},10000,'window_cap'),('deadline',ledger,window,100,'deadline')]:
 assert not settled_admission_checks(l,w,est,t)[key];checks[name+'_cap_not_relaxed']=True
reject(lambda:settled_admission_checks(ledger,window,dict(est,all_one=float('nan')),10000),'invalid full-dose estimate');checks['nonfinite_rejected']=True
original=(C.require_release,C.source_check,C.input_check,C.O);real_sha=C.sha
with tempfile.TemporaryDirectory() as td:
 C.O=Path(td);release=C.O/'card086-release.json';source=C.O/'source';data=C.O/'input';mutable=C.O/'checkpoint'
 for p in [release,source,data,mutable]:p.write_text('original')
 expected={p:real_sha(p) for p in [release,source,data]};calls={'release':0,'source':0,'input':0}
 def req():
  calls['release']+=1;assert C.sha(release)==expected[release],'release full hash mismatch';return {'runtime':'r'}
 def src():
  calls['source']+=1;assert C.sha(source)==expected[source],'source full hash mismatch';return 'r'
 def inp():
  calls['input']+=1;assert C.sha(data)==expected[data],'input full hash mismatch'
 C.require_release=req;C.source_check=src;C.input_check=inp
 try:
  record={}
  with A.admitted_canary(record):
   for i in range(40):assert C.require_release()=={'runtime':'r'} and C.source_check()=='r'
   h=C.sha(mutable);mutable.write_text('changed');assert C.sha(mutable)!=h
  assert calls=={'release':2,'source':2,'input':2} and record['post_full_fingerprint_PASS'];checks['two_full_passes_80_guarded_calls']=True;checks['mutable_checkpoint_hash_not_cached']=True
  assert C.require_release is req and C.source_check is src;checks['offpath_restored']=True
  for label,p,msg in [('source',source,'source full hash mismatch'),('input',data,'input full hash mismatch'),('release',release,'release full hash mismatch')]:
   def fault():
    with A.admitted_canary({}):
     p.write_text('changed')
     try:C.require_release()
     except AssertionError as e:assert str(e)=='canary admission metadata changed';checks[label+'_percall_reject']=True
     else:raise AssertionError('metadata fault accepted')
   reject(fault,msg);checks[label+'_posthash_reject']=True;p.write_text('original')
  from unittest.mock import patch
  for label,target,value,msg in [('pid','os.getpid',-1,'canary admission PID changed'),('expiry','time.monotonic',1e20,'canary admission expired')]:
   with A.admitted_canary({}):
    with patch(target,return_value=value):reject(C.require_release,msg)
   checks[label+'_rejected']=True
  before=os.environ.get('CARD086_RELEASE_SHA')
  with A.admitted_canary({}):
   os.environ['CARD086_RELEASE_SHA']='changed'
   reject(C.require_release,'canary release environment changed')
   if before is None:os.environ.pop('CARD086_RELEASE_SHA')
   else:os.environ['CARD086_RELEASE_SHA']=before
  checks['release_environment_rejected']=True
  try:
   with A.admitted_canary({}):raise ValueError('device control failed')
  except ValueError as e:assert str(e)=='device control failed'
  assert C.require_release is req and C.source_check is src;checks['exception_restoration']=True
 finally:C.require_release,C.source_check,C.input_check,C.O=original
assert C.sha is real_sha
C.write(C.O/'card086-reciprocal-readiness/admission-contracts.json',{'PASS':True,'n_checks':len(checks),'checks':checks,'scope':'Pure settled budget checks, real filesystem full pre/post and metadata guards, mutable checkpoint hash offpath; no GPU or live ledger mutation.'})
print('V2 ADMISSION PASS',len(checks))
