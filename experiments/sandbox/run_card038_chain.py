"""Root-admitted sequential compact-width experiment, both leases held by launcher.
Actual canary/preflight, two unchanged doses and measured projection benchmark; every exit charged.
"""
from pathlib import Path
import sys,json,subprocess,time,datetime as dt,fcntl,math
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import card038_validate as V
R=Path(__file__).resolve().parents[2];O=V.OC;PY='/home/enjalot/code/latent-basemap/.venv/bin/python';C=O/'card038-ledger.json';W=O/'cards-24h-window-ledger.json';END=dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp();CAPS={'wide2048':1800,'compact1024':3000};START=None;CHARGED=0.
def write(p,r):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def charge(tag,seconds,rc):
 global CHARGED
 CHARGED+=seconds
 with (O/'window-ledger-write.lock').open('a') as lk:
  fcntl.flock(lk,fcntl.LOCK_EX)
  for p,key in [(C,'batch_spent_s'),(W,'spent_s')]:
   r=json.loads(p.read_text());r[key]+=seconds;r.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'038','tag':tag,'event':'exclusive_GPU_stage','wall_s':seconds,'rc':rc});write(p,r)
def remaining(cap,arm=None):
 card=json.loads(C.read_text());win=json.loads(W.read_text());pending=max(0,time.monotonic()-START-CHARGED);values=[cap,5400-card['batch_spent_s']-pending,86400-win['spent_s']-pending,END-time.time()]
 if arm:values.append(CAPS[arm]-sum(e['wall_s'] for e in card['entries'] if e['tag']==arm))
 return min(values)
def stage(tag,name,cap,args=(),arm=None):
 assert V.runtime_manifest_check(R)[0];timeout=remaining(cap,arm);assert timeout>5,'no remaining admission time';start=time.monotonic();rc=999;print(dt.datetime.now(dt.timezone.utc).isoformat(),'START',tag,'timeout',timeout,flush=True)
 try:rc=subprocess.run([PY,str(R/'experiments/sandbox'/name),*args],cwd=R,timeout=timeout).returncode
 except subprocess.TimeoutExpired:rc=124
 finally:charge(tag,time.monotonic()-start,rc)
 print('DONE',tag,'rc',rc,flush=True);assert V.runtime_manifest_check(R)[0];return rc

def main():
 global START
 START=time.monotonic()
 if not C.exists():write(C,{'schema':'card038-ledger','batch_cap_s':5400,'batch_spent_s':0.,'entries':[]})
 attempt=time.time();r=stage('device_canary','gpu_card038_canary.py',600);assert r==0,'device canary failed'
 for n in ['card038-canary.json']:
  p=O/n;x=json.loads(p.read_text());assert p.stat().st_mtime>=attempt and x['PASS'] and x['runtime_manifest_sha']==V.full_sha(V.runtime_manifest_path(R)), 'stale/wrong device receipt'
 r=stage('preflight','gpu_card038_preflight.py',700)
 if r==3:
  write(O/'card038-execution.json',{'status':'ADMISSION_STOP','at':dt.datetime.now(dt.timezone.utc).isoformat(),'reason':'unchanged fits do not meet measured time caps; no dose truncation'});return
 assert r==0,'preflight failed';pf=json.loads((O/'card038-preflight.json').read_text());assert pf['PASS'] and (O/'card038-preflight.json').stat().st_mtime>=attempt and pf['runtime_manifest_sha']==V.full_sha(V.runtime_manifest_path(R))
 receipts={}
 for a in V.ARMS:
  m=V.TD_DEFAULT/f'manifest-{a}.json'
  if m.exists():receipts[a]=V.strict_validate_arm(a,R);continue
  from run_card038_arm import _latest_ckpt
  ck,gs=_latest_ckpt(V.TD_DEFAULT/a/'ckpts');need=pf['estimates'][a]['per_step_s']*(V.DOSE[a]-gs)+pf['estimates'][a]['reserve_s'];assert remaining(CAPS[a],a)>=need, 'unchanged remaining dose does not fit'
  rc=stage(a,'run_card038_arm.py',CAPS[a],args=(a,str(V.DOSE[a])),arm=a);assert rc==0, f'{a} rc={rc}; preserve checkpoint and charge attempt'
  receipts[a]=V.strict_validate_arm(a,R)
 # Positive probes match attempted stream; ranked negative IDs may differ by model, explicitly disclosed.
 stats={a:json.loads((V.TD_DEFAULT/f'manifest-{a}.json').read_text())['train_stats'] for a in V.ARMS};probes={a:{p['attempted_step']:p for p in st['card038_attempted_probes']} for a,st in stats.items()};comparison={s:probes['wide2048'][s]['positive_pair_sha256']==probes['compact1024'][s]['positive_pair_sha256'] for s in [1,30000,60000]};assert all(comparison.values()),'attempted positive-edge stream mismatch'
 rc=stage('projection_benchmark','gpu_card038_benchmark.py',180);assert rc==0,'projection benchmark engineering failure'
 write(O/'card038-execution.json',{'status':'TRAINED_VALIDATED_BENCHMARKED','at':dt.datetime.now(dt.timezone.utc).isoformat(),'arms':receipts,'attempted_positive_probe_match':comparison,'quality':'NOT_YET_SCORED','benchmark_sha':V.full_sha(O/'card038-benchmark.json')})
 subprocess.run([PY,str(O/'notify.py'),'post','codex-overseer',str(O/'card038-execution.json'),'Card038 wide60K and compact180K heads trained, strict-validated and projection-benchmarked. Compact60K diagnostic retained. CPU quality scoring next; no promotion yet.'],timeout=30)
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(O/'card038-execution.json',{'status':'EXECUTION_FAILED','at':dt.datetime.now(dt.timezone.utc).isoformat(),'error':repr(e)});raise
 finally:
  if START is not None:charge('controller_reconciliation',max(0,time.monotonic()-START-CHARGED),0)
