"""Bounded supervised-readiness then matched graph training. Both leases held externally."""
import sys,time,subprocess,fcntl,datetime as dt
from pathlib import Path
import card065_common as C
from card065_fit import validate_arm
PY='/home/enjalot/code/latent-basemap/.venv/bin/python';L=C.O/'card065-ledger.json';W=C.O/'cards-24h-window-ledger.json';END=dt.datetime.fromisoformat('2026-09-13T23:50:55+00:00').timestamp();START=None;CHARGED=0.
def charge(tag,seconds,rc):
 global CHARGED
 CHARGED+=seconds
 with (C.O/'window-ledger-write.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  for p,key in [(L,'batch_spent_s'),(W,'spent_s')]:
   r=C.read(p);r[key]+=seconds;r.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'065','tag':tag,'event':'exclusive_GPU_stage','wall_s':seconds,'rc':rc});C.write(p,r)
def remaining(cap,arm=None):
 r=C.read(L);w=C.read(W);pending=max(0.,time.monotonic()-START-CHARGED);v=[cap,5400-r['batch_spent_s']-pending,165491-w['spent_s']-pending,END-time.time()]
 if arm:v.append(1500-sum(e['wall_s'] for e in r['entries'] if e['tag']==arm))
 return min(v)
def stage(tag,script,cap,args=(),arm=None):
 C.source_check();timeout=remaining(cap,arm);assert timeout>5;start=time.monotonic();rc=999;print('START',tag,timeout,flush=True)
 try:rc=subprocess.run([PY,str(C.R/'experiments/sandbox'/script),*args],cwd=C.R,timeout=timeout).returncode
 except subprocess.TimeoutExpired:rc=124
 finally:charge(tag,time.monotonic()-start,rc)
 C.source_check();print('END',tag,rc,flush=True);return rc
def receipt(name,since):
 p=C.O/name;r=C.read(p);assert p.stat().st_mtime>=since and r['PASS'] and r['runtime_sha']==C.source_check();return r
def main():
 global START
 START=time.monotonic()
 if not L.exists():C.write(L,{'batch_cap_s':5400,'batch_spent_s':0.,'entries':[]})
 rel=C.read(C.O/'card065-release.json');assert rel['PASS'] and all(C.sha(p)==h for p,h in rel['files'].items());C.input_check();t=time.time()
 assert stage('prep_canary','gpu_card065_prepare_canary.py',300)==0;receipt('card065-prep-canary.json',t)
 for arm in ['pca','shuffled','random']:
  rc=stage('prepare_'+arm,'run_card065_prepare.py',600,args=(arm,))
  if rc==3:C.write(C.O/'card065-execution.json',{'status':'INITIALIZATION_READINESS_FAIL','arm':arm,'reason':'fixed2000-step PCA initialization did not reach frozen R2 threshold; no graph arms launched'});return
  assert rc==0,'preparation failed; preserve resumable states'
  p=C.read(C.TD/arm/'preparation.json');assert p['READY'] and p['prepared_sha']==C.sha(C.warm(arm))
 assert stage('graph_canary','gpu_card065_graph_canary.py',600)==0;receipt('card065-graph-canary.json',t)
 rc=stage('preflight','gpu_card065_preflight.py',600)
 if rc==3:C.write(C.O/'card065-execution.json',{'status':'ADMISSION_STOP','reason':'all unchanged60K graph doses exceed measured caps; no truncation'});return
 assert rc==0;pf=receipt('card065-preflight.json',t);receipts={}
 for arm in C.ARMS:
  if (C.TD/arm/'manifest.json').exists():receipts[arm]=validate_arm(arm);continue
  assert remaining(1500,arm)>=pf['estimates'][arm]['complete_arm_s'],'full unchanged dose cannot be admitted'
  assert stage(arm,'run_card065_arm.py',1500,args=(arm,),arm=arm)==0,'graph arm failed; preserve checkpoints';receipts[arm]=validate_arm(arm)
 assert len({r['state_sha'] for r in receipts.values()})==3
 C.write(C.O/'card065-execution.json',{'status':'TRAINED_VALIDATED','at':dt.datetime.now(dt.timezone.utc).isoformat(),'arms':receipts,'quality':'NOT_SCORED','runtime_sha':C.source_check()})
 subprocess.run([PY,str(C.O/'notify.py'),'post','codex-overseer',str(C.O/'card065-execution.json'),'Card065 informative-initialization three matched60K graph heads trained and validated. Separate frozen CPU quality score/audit next; no promotion yet.'],timeout=30)
if __name__=='__main__':
 try:main()
 except Exception as e:C.write(C.O/'card065-execution.json',{'status':'EXECUTION_FAILED','error':repr(e),'at':dt.datetime.now(dt.timezone.utc).isoformat()});raise
 finally:
  if START is not None:
   charge('controller_reconciliation',max(0.,time.monotonic()-START-CHARGED),0)
   if (C.O/'card065-execution.json').exists() and C.read(C.O/'card065-execution.json')['status']!='TRAINED_VALIDATED':
    subprocess.run([PY,str(C.O/'notify.py'),'post','codex-overseer',str(C.O/'card065-execution.json'),'Card065 stopped before a complete matched pilot; inspect readiness/execution receipt, preserve attempts and unchanged gates.'],timeout=30)
