"""Sequential source-support pilot with bounded stage occupancy and frozen admissions."""
from pathlib import Path
import datetime as dt,fcntl,json,subprocess,time
import numpy as np
import torch
from run_card017_arm import ROOT,OC,OUT,SUBD,ARMS,SNAPS,sha,write,state_sha
PY='/home/enjalot/code/latent-basemap/.venv/bin/python';CAP=5400
CARD=OC/'card017-ledger.json';WIN=OC/'cards-24h-window-ledger.json'
END=dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp()


def charge(tag,seconds,rc):
 with (OC/'window-ledger-write.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for p,key in [(CARD,'batch_spent_s'),(WIN,'spent_s')]:
   d=json.loads(p.read_text());d[key]=float(d.get(key,0))+seconds
   d.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'017','event':'exclusive_gpu_stage','tag':tag,'wall_s':seconds,'rc':rc})
   write(p,d)


def validate(arm):
 folder=OUT/arm;a=json.loads((folder/'admission.json').read_text());m=json.loads((folder/'complete.json').read_text())
 assert a==m['identity'] and m['successful_steps']==m['train_stats']['positive_lr_optimizer_steps']==a['steps']==60000
 assert a['lr']==m['train_stats']['lr_used_min']==m['train_stats']['lr_used_max']==.001
 assert a['shared_init_parameter_order_sha']==m['warm_parameter_sha']=='589895f037d406ae'
 assert a['n_components']==2 and a['batch_size']==16384 and a['rankneg_window']==75000 and not a['canary']
 assert a['graph_manifest_sha']==sha(SUBD/'graph-manifest.json') and a['draw_manifest_sha']==sha(SUBD/'draw-manifest.json')
 assert all(sha(ROOT/n)==h for n,h in a['runtime'].items())
 snapshots=[]
 for step in SNAPS:
  p=folder/f'model-step{step}.pt';o=torch.load(p,map_location='cpu',weights_only=False)
  assert o['n_components']==2 and o['learning_rate']==.001 and o['lr_schedule']=='constant'
  assert all(torch.isfinite(v).all() for v in o['model_state_dict'].values()) and p.stat().st_mtime>=(folder/'admission.json').stat().st_mtime
  snapshots.append(state_sha(o['model_state_dict']))
 assert len(set(snapshots))==4 and snapshots[-1]==m['endpoint_named_sha']
 endpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
 assert state_sha(endpoint['model_state_dict'])==snapshots[-1] and sha(folder/'model.pt')==m['endpoint_file_sha']
 xy=np.load(folder/'train-xy.npy',mmap_mode='r');assert xy.shape==(300000,2) and np.isfinite(xy).all()
 return {'arm':arm,'PASS':True,'endpoint_named_sha':m['endpoint_named_sha'],'warm_parameter_sha':m['warm_parameter_sha']}


def notify(msg):
 subprocess.run([PY,str(OC/'notify.py'),'post','codex-overseer',str(OC/'card017-execution.json'),msg],timeout=30)


def main():
 if not CARD.exists():write(CARD,{'schema':'card017-ledger','batch_cap_s':CAP,'batch_spent_s':0,'entries':[]})
 runtime=json.loads((ROOT/'card017-runtime-sha.json').read_text());assert all(sha(ROOT/n)==h for n,h in runtime.items())
 assert json.loads((SUBD/'graph-manifest.json').read_text())['complete']
 assert json.loads((OC/'card017-independent-cpu-audit.json').read_text())['PASS']
 stages=[('gpu_canary','gpu_card017_canary.py',[],90,300)]+[(a,'run_card017_arm.py',[a],1050,1500) for a in ARMS]
 done=[]
 for tag,script,args,expected,cap in stages:
  available=min(cap,CAP-json.loads(CARD.read_text())['batch_spent_s'],86400-json.loads(WIN.read_text())['spent_s'],END-time.time())
  assert available>=expected,f'cannot admit {tag} with {available:.1f}s remaining'
  print(dt.datetime.now(dt.timezone.utc).isoformat(),'START',tag,'timeout',available,flush=True)
  t=time.monotonic();rc=999
  try:rc=subprocess.run([PY,str(ROOT/'experiments/sandbox'/script),*args],cwd=ROOT,timeout=available).returncode
  except subprocess.TimeoutExpired:rc=124
  finally:charge(tag,time.monotonic()-t,rc)
  assert rc==0,f'{tag} failed rc={rc}'
  if tag=='gpu_canary':assert json.loads((OC/'card017-gpu-canary.json').read_text())['PASS']
  else:
   done.append(validate(tag));write(OC/'card017-completion-validation.json',{'arms':done,'all_three_valid':len(done)==3})
 assert len({a['endpoint_named_sha'] for a in done})==3
 write(OC/'card017-execution.json',{'status':'TRAINED_VALIDATED','arms':done,'quality':'NOT_YET_SCORED'})
 notify('Card017 uniform/geometric/hybrid 300K-support heads trained and validated at matched60K updates. CPU quality scoring next; no promotion yet. Codex owns next allocation.')


if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(OC/'card017-execution.json',{'status':'EXECUTION_FAILED','error':repr(exc),'at':dt.datetime.now(dt.timezone.utc).isoformat()})
  notify(f'Card017 stopped fail-closed: {exc}. Preserve attempts and charge occupancy; no quality verdict.')
  raise
