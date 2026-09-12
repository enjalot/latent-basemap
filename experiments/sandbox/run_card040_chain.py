"""Exclusive GPU chain with cumulative attempt charging and unchanged 20K doses."""
from pathlib import Path
import sys,time,json,subprocess,fcntl,datetime as dt
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import card040_common as V
C=V.O/'card040-ledger.json';W=V.O/'cards-24h-window-ledger.json';PY='/home/enjalot/code/latent-basemap/.venv/bin/python';END=dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp();START=None;CHARGED=0.
def charge(tag,seconds,rc):
 global CHARGED
 CHARGED+=seconds
 with (V.O/'window-ledger-write.lock').open('a') as lk:
  fcntl.flock(lk,fcntl.LOCK_EX)
  for p,key in [(C,'batch_spent_s'),(W,'spent_s')]:
   r=json.loads(p.read_text());r[key]+=seconds;r.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'040','tag':tag,'event':'exclusive_GPU_stage','wall_s':seconds,'rc':rc});V.atomic(p,r)
def room(cap,tag):
 c=json.loads(C.read_text());w=json.loads(W.read_text());pending=max(0,time.monotonic()-START-CHARGED);prior=sum(x['wall_s'] for x in c['entries'] if x['tag']==tag)
 return min(cap-prior,1800-c['batch_spent_s']-pending,86400-w['spent_s']-pending,END-time.time())
def stage(tag,name,cap,args=()):
 V.runtime_check();timeout=room(cap,tag);assert timeout>5,'time admission failed';start=time.monotonic();rc=999;print('START',tag,timeout,flush=True)
 try:rc=subprocess.run([PY,str(V.R/'experiments/sandbox'/name),*args],cwd=V.R,timeout=timeout).returncode
 except subprocess.TimeoutExpired:rc=124
 finally:charge(tag,time.monotonic()-start,rc)
 V.runtime_check();print('END',tag,rc,flush=True);return rc

def main():
 global START
 START=time.monotonic()
 if not C.exists():V.atomic(C,{'batch_cap_s':1800,'batch_spent_s':0.,'entries':[]})
 attempt=time.time()
 for tag,name,cap in [('bank','build_card040_bank_gpu.py',300),('canary','gpu_card040_canary.py',180),('preflight','gpu_card040_preflight.py',300)]:
  rc=stage(tag,name,cap)
  if rc==3:V.atomic(V.O/'card040-execution.json',{'status':'ADMISSION_STOP','stage':tag,'quality':'NOT_TESTED','at':dt.datetime.now(dt.timezone.utc).isoformat()});return
  assert rc==0,(tag,rc)
  if tag in ['canary','preflight']:
   p=V.O/('card040-'+tag+'.json');r=json.loads(p.read_text());assert p.stat().st_mtime>=attempt and r['PASS'] and r['runtime_manifest_sha']==V.runtime_check()
 from run_card040_arm import validate_arm,latest
 pf=json.loads((V.O/'card040-preflight.json').read_text());receipts={}
 for a in V.ARMS:
  if not (V.TRAIN/a/'manifest.json').exists():
   step,_=latest(a);need=pf['estimates'][a]['per_step_s']*(V.DOSE-step)*1.2+25;assert room(600,a)>=need,'remaining exact dose exceeds cap';assert stage(a,'run_card040_arm.py',600,(a,))==0,'arm failed; preserve checkpoints'
  receipts[a]=validate_arm(a)
 stats={a:json.loads((V.TRAIN/a/'manifest.json').read_text())['stats'] for a in V.ARMS};assert stats['post']['row_probes']==stats['pre']['row_probes'],'matched row/target exposure mismatch'
 assert stage('evaluate','gpu_card040_evaluate.py',180)==0,'evaluation engineering failure';ev=V.O/'card040-evaluation.json';r=json.loads(ev.read_text());assert r['PASS'] and ev.stat().st_mtime>=attempt and r['runtime_manifest_sha']==V.runtime_check()
 V.atomic(V.O/'card040-execution.json',{'status':'TRAINED_VALIDATED_BENCHMARKED','at':dt.datetime.now(dt.timezone.utc).isoformat(),'arms':receipts,'evaluation_sha':V.sha(ev),'quality':'NOT_YET_SCORED'})
if __name__=='__main__':
 try:main()
 except Exception as ex:V.atomic(V.O/'card040-execution.json',{'status':'EXECUTION_FAILED','at':dt.datetime.now(dt.timezone.utc).isoformat(),'error':repr(ex)});raise
 finally:
  if START is not None:charge('controller_reconciliation',max(0,time.monotonic()-START-CHARGED),0)
