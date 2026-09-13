"""Root-admitted landmark-correlation chain. External launcher holds both GPU leases; every exit charged."""
import sys,time,subprocess,fcntl,datetime as dt
from pathlib import Path
import card060_common as C
from card060_fit import validate_arm
PY='/home/enjalot/code/latent-basemap/.venv/bin/python';L=C.O/'card060-ledger.json';W=C.O/'cards-24h-window-ledger.json';END=dt.datetime.fromisoformat('2026-09-13T23:50:55+00:00').timestamp();START=None;CHARGED=0.
def charge(tag,seconds,rc):
 global CHARGED
 CHARGED+=seconds
 with (C.O/'window-ledger-write.lock').open('a') as lk:
  fcntl.flock(lk,fcntl.LOCK_EX)
  for p,key in [(L,'batch_spent_s'),(W,'spent_s')]:
   r=C.read(p);r[key]+=seconds;r.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'060','tag':tag,'event':'exclusive_GPU_stage','wall_s':seconds,'rc':rc});C.write(p,r)
def remaining(cap,arm=None):
 card=C.read(L);win=C.read(W);pending=max(0.,time.monotonic()-START-CHARGED);v=[cap,2400-card['batch_spent_s']-pending,165491-win['spent_s']-pending,END-time.time()]
 if arm:v.append(1000-sum(e['wall_s'] for e in card['entries'] if e['tag']==arm))
 return min(v)
def stage(tag,script,cap,args=(),arm=None):
 C.source_check();timeout=remaining(cap,arm);assert timeout>5,'no time remaining';start=time.monotonic();rc=999;print('START',tag,'timeout',timeout,flush=True)
 try:rc=subprocess.run([PY,str(C.R/'experiments/sandbox'/script),*args],cwd=C.R,timeout=timeout).returncode
 except subprocess.TimeoutExpired:rc=124
 finally:charge(tag,time.monotonic()-start,rc)
 C.source_check();print('END',tag,rc,flush=True);return rc
def fresh(name,start):
 p=C.O/name;r=C.read(p);assert p.stat().st_mtime>=start and r['PASS'] and r['runtime_sha']==C.source_check(),'stale/invalid receipt';return r
def main():
 global START
 START=time.monotonic()
 if not L.exists():C.write(L,{'schema':'card060-ledger','batch_cap_s':2400,'batch_spent_s':0.,'entries':[]})
 t=time.time();rel=C.read(C.O/'card060-release.json');assert rel['PASS'] and all(C.sha(Path(p))==h for p,h in rel['files'].items());C.input_check();C.graph_check()
 assert stage('device_canary','gpu_card060_canary.py',600)==0,'device canary failed';fresh('card060-device-canary.json',t)
 rc=stage('preflight','gpu_card060_preflight.py',500)
 if rc==3:C.write(C.O/'card060-execution.json',{'status':'ADMISSION_STOP','at':dt.datetime.now(dt.timezone.utc).isoformat(),'reason':'unchanged doses do not fit measured time caps; no truncation'});return
 assert rc==0,'preflight failed';pf=fresh('card060-preflight.json',t);receipts={}
 for arm in C.TREATMENTS:
  if (C.TD/arm/'manifest.json').exists():receipts[arm]=validate_arm(arm);continue
  assert remaining(1000,arm)>=pf['estimates'][arm]['complete_arm_s'],'unchanged full dose cannot be admitted'
  assert stage(arm,'run_card060_arm.py',1000,args=(arm,),arm=arm)==0,'arm stopped; preserve checkpoints';receipts[arm]=validate_arm(arm)
 assert len({r['state_sha'] for r in receipts.values()})==2,'no exposed intervention'
 C.write(C.O/'card060-execution.json',{'status':'TRAINED_VALIDATED','at':dt.datetime.now(dt.timezone.utc).isoformat(),'arms':receipts,'quality':'NOT_YET_SCORED','runtime_sha':C.source_check()})
 subprocess.run([PY,str(C.O/'notify.py'),'post','codex-overseer',str(C.O/'card060-execution.json'),'Card060 landmark-correlation two stronger-weight matched3D arms trained and validated; original054 controls reused. Separate CPU scoring next; no promotion yet.'],timeout=30)
if __name__=='__main__':
 try:main()
 except Exception as e:C.write(C.O/'card060-execution.json',{'status':'EXECUTION_FAILED','at':dt.datetime.now(dt.timezone.utc).isoformat(),'error':repr(e)});raise
 finally:
  if START is not None:charge('controller_reconciliation',max(0.,time.monotonic()-START-CHARGED),0)
