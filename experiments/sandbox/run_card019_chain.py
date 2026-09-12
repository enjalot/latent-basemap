from pathlib import Path
import datetime as dt,fcntl,json,subprocess,time
import torch
from run_card019_arm import ROOT,OC,OUT,DATA,HEAD,ARMS,ACT,SNAPS,sha,write,state_sha
PY='/home/enjalot/code/latent-basemap/.venv/bin/python';CAP=3600
CARD=OC/'card019-ledger.json';WIN=OC/'cards-24h-window-ledger.json';END=dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp()
def charge(tag,seconds,rc):
 with (OC/'window-ledger-write.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for p,key in [(CARD,'batch_spent_s'),(WIN,'spent_s')]:
   d=json.loads(p.read_text());d[key]=float(d.get(key,0))+seconds;d.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'019','event':'exclusive_gpu_stage','tag':tag,'wall_s':seconds,'rc':rc});write(p,d)
def validate(arm):
 folder=OUT/arm;a=json.loads((folder/'admission.json').read_text());m=json.loads((folder/'complete.json').read_text())
 assert a==m['identity'] and m['successful_steps']==m['train_stats']['positive_lr_optimizer_steps']==m['train_stats']['executed_iters']==a['steps']==60000
 assert a['arm']==arm and a['activation']==ACT[arm] and a['negative_slope']==(.01 if arm=='leaky' else 0.)
 assert a['lr']==m['train_stats']['lr_used_min']==m['train_stats']['lr_used_max']==.0001 and a['lr_schedule']=='constant'
 assert a['head_sha']==sha(HEAD) and a['data_manifest_sha']==sha(DATA/'manifest.json') and m['pipeline']['x_residency']=='device_fp16'
 assert all(sha(ROOT/n)==h for n,h in a['runtime'].items())
 states=[]
 for step in SNAPS:
  p=folder/f'model-step{step}.pt';o=torch.load(p,map_location='cpu',weights_only=False)
  assert o['final_activation']==ACT[arm] and o['learning_rate']==.0001 and o['lr_schedule']=='constant'
  assert all(torch.isfinite(v).all() for v in o['model_state_dict'].values()) and p.stat().st_mtime>=(folder/'admission.json').stat().st_mtime
  states.append(state_sha(o['model_state_dict']))
  c=torch.load(folder/f'ckpt/ckpt-step{step}.pt',map_location='cpu',weights_only=False)
  assert c['global_step']==step and c['step_checkpoint'] and c['card012_identity']==a and c['config']['final_activation']==ACT[arm]
 assert len(set(states))==3 and states[-1]==m['endpoint_named_sha']
 endpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
 assert endpoint['final_activation']==ACT[arm] and state_sha(endpoint['model_state_dict'])==states[-1] and sha(folder/'model.pt')==m['endpoint_file_sha']
 return {'PASS':True,'arm':arm,'activation':ACT[arm],'endpoint_named_sha':m['endpoint_named_sha'],'warm_named_sha':a['warm_named_sha'],'warm_parameter_sha':m['warm_parameter_sha']}
def notify(msg):subprocess.run([PY,str(OC/'notify.py'),'post','codex-overseer',str(OC/'card019-execution.json'),msg],timeout=30)
def main():
 if not CARD.exists():write(CARD,{'schema':'card019-ledger','batch_cap_s':CAP,'batch_spent_s':0,'entries':[]})
 rt=json.loads((ROOT/'card019-runtime-sha.json').read_text());assert all(sha(ROOT/n)==h for n,h in rt.items())
 cpu=json.loads((OC/'card019-cpu-canary.json').read_text());assert cpu['PASS'] and cpu['core_sha']==sha(ROOT/'basemap/pumap/parametric_umap/core.py') and cpu['module_sha']==sha(ROOT/'basemap/pumap/parametric_umap/models/mlp.py')
 stages=[('gpu_canary','gpu_card019_canary.py',[],120,300)]+[(a,'run_card019_arm.py',[a],1050,1500) for a in ARMS]
 done=[]
 for tag,script,args,expected,limit in stages:
  avail=min(limit,CAP-json.loads(CARD.read_text())['batch_spent_s'],86400-json.loads(WIN.read_text())['spent_s'],END-time.time());assert avail>=expected
  print(dt.datetime.now(dt.timezone.utc).isoformat(),'START',tag,'timeout',avail,flush=True);t=time.monotonic();rc=999
  try:rc=subprocess.run([PY,str(ROOT/'experiments/sandbox'/script),*args],cwd=ROOT,timeout=avail).returncode
  except subprocess.TimeoutExpired:rc=124
  finally:charge(tag,time.monotonic()-t,rc)
  assert rc==0,f'{tag} failed rc={rc}'
  if tag=='gpu_canary':assert json.loads((OC/'card019-gpu-canary.json').read_text())['PASS']
  else:done.append(validate(tag));write(OC/'card019-completion-validation.json',{'arms':done,'both_valid':len(done)==2})
 assert len({a['warm_named_sha'] for a in done})==1 and len({a['endpoint_named_sha'] for a in done})==2
 write(OC/'card019-execution.json',{'status':'TRAINED_VALIDATED','arms':done,'quality':'NOT_YET_SCORED'});notify('Card019 matched final-activation continuation heads trained and validated. Tie-aware pile and common quality scoring next; no promotion yet.')
if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(OC/'card019-execution.json',{'status':'EXECUTION_FAILED','error':repr(exc),'at':dt.datetime.now(dt.timezone.utc).isoformat()});notify(f'Card019 stopped fail-closed: {exc}. No quality verdict.');raise
