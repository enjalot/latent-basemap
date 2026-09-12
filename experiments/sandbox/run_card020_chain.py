"""Run only under root's two-lock systemd wrapper; strict card/window occupancy accounting."""
from run_card020 import *
import subprocess,datetime as dt,fcntl
CAP=300;LEDGER=OC/'card020-ledger.json';WINDOW=OC/'cards-24h-window-ledger.json';END=dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp();PY='/home/enjalot/code/latent-basemap/.venv/bin/python'
def charge(tag,seconds,rc):
 with (OC/'window-ledger-write.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  for p,key in [(LEDGER,'batch_spent_s'),(WINDOW,'spent_s')]:
   d=json.loads(p.read_text());d[key]=float(d.get(key,0))+seconds;d.setdefault('entries',[]).append({'card':'020','event':'exclusive_gpu_stage','tag':tag,'wall_s':seconds,'rc':rc});write(p,d)
def remaining():return min(CAP-json.loads(LEDGER.read_text())['batch_spent_s'],86400-json.loads(WINDOW.read_text())['spent_s'],END-time.time())
def main():
 if not LEDGER.exists():write(LEDGER,{'batch_cap_s':CAP,'batch_spent_s':0,'entries':[]})
 prep=time.monotonic()
 try:
  cpu=json.loads((OC/'card020-cpu-canary.json').read_text());assert cpu['PASS'] and cpu['identity']==identity(True);validate_inputs()
 finally:charge('chain-admission',time.monotonic()-prep,0)
 for tag,script,args in [('device-canary','card020_canary.py',['cuda']),('continuation','run_card020.py',[])]:
  if tag=='continuation':
   c=json.loads((OC/'card020-cuda-canary.json').read_text());assert c['PASS'] and c['identity']==identity(True)
   assert remaining()>=c['admission_estimate_s'],'measured continuation estimate exceeds remaining occupancy'
  t=time.monotonic();rc=999
  try:rc=subprocess.run([PY,str(ROOT/'experiments/sandbox'/script),*args],cwd=ROOT,timeout=max(1,remaining())).returncode
  except subprocess.TimeoutExpired:rc=124
  finally:charge(tag,time.monotonic()-t,rc)
  assert rc==0,f'{tag} failure {rc}'
 close=time.monotonic()
 try:
  v=validate_endpoint();write(OC/'card020-execution.json',{'status':'TRAINED_VALIDATED','validation':v,'quality':'NOT_YET_SCORED'})
 finally:charge('chain-closeout',time.monotonic()-close,0)
if __name__=='__main__':
 try:main()
 except Exception as e:write(OC/'card020-execution.json',{'status':'EXECUTION_FAILED','error':repr(e)});raise
