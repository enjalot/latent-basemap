"""One exact-dose readout fit with genuine resumable state; no frozen-backbone optimization."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import torch,numpy as np
import card040_common as V
from card040_engine import Bank,Engine,validate_payload

def latest(arm):
 d=V.TRAIN/arm;out=[]
 for p in d.glob('ckpt-step*.pt'):
  ck=torch.load(p,map_location='cpu',weights_only=False);s=int(p.stem.removeprefix('ckpt-step'));assert ck['step']==s and s in V.STEPS,'checkpoint filename/dose mismatch';validate_payload(ck,V.identity(arm));out.append((s,p))
 return max(out,key=lambda row:row[0]) if out else (0,None)
def validate_arm(arm):
 p=V.TRAIN/arm;ad=json.loads((p/'admission.json').read_text());man=json.loads((p/'manifest.json').read_text());expected=V.identity(arm);assert ad['identity']==man['identity']==expected;assert man['positive_lr_updates']==V.DOSE and man['init_sha']==expected['init_sha'];snapshots={}
 for step in V.STEPS:
  path=p/f'ckpt-step{step}.pt';ck=torch.load(path,map_location='cpu',weights_only=False);validate_payload(ck,expected);assert ck['step']==step and path.stat().st_mtime>=(p/'admission.json').stat().st_mtime;snapshots[step]=V.state_sha(ck['model'])
 assert len(set(snapshots.values()))==len(V.STEPS),'no checkpoint progress';ep=torch.load(p/'adapter.pt',map_location='cpu',weights_only=False);assert ep['schema']=='card040-inference-readout-v1' and ep['identity']==expected and V.state_sha(ep['model'])==snapshots[V.DOSE];assert ep['teacher_sha']==V.BODY_SHA and ep['PCA_sha']==V.PCA_SHA and ep['base_negative_slope']==V.SLOPE
 norm=np.load(V.DATA/'normalization.npz');assert torch.equal(ep['mean'],torch.from_numpy(norm[arm+'_mean'])) and torch.equal(ep['std'],torch.from_numpy(norm[arm+'_std'])) and ep['target_scale']==expected['target_scale'];assert man['stats']==ck['stats'] and man['model_sha']==V.sha(p/'adapter.pt');assert man['resources']['global_gpu_gib']<30 and man['resources']['max_rss_mib']<12288;return {'PASS':True,'arm':arm,'model_sha':man['model_sha'],'state_sha':snapshots[V.DOSE],'init_sha':expected['init_sha'],'steps':V.DOSE,'identity':expected}
def main():
 arm=sys.argv[1];assert arm in V.ARMS;V.device_setup();V.runtime_check();d=V.TRAIN/arm;d.mkdir(exist_ok=True,parents=True)
 if (d/'manifest.json').exists():print(validate_arm(arm));return
 bank=Bank();e=Engine(arm,bank);ad=d/'admission.json';s,path=latest(arm)
 if path is not None:assert json.loads(ad.read_text())['identity']==e.identity;e.restore(path)
 else:
  assert not ad.exists(),'admission without checkpoint: root must preserve/charge interrupted short attempt before retry';V.atomic(ad,{'identity':e.identity,'written_before_steps':True})
 start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 while e.step<V.DOSE:
  e.advance()
  if e.step in V.STEPS:e.save(d/f'ckpt-step{e.step}.pt');print(arm,e.step,e.stats['loss_last'],flush=True)
  if e.step%1000==0:V.resources()
 norm=np.load(V.DATA/'normalization.npz');artifact={'schema':'card040-inference-readout-v1','identity':e.identity,'model':e.model.state_dict(),'mean':torch.from_numpy(norm[arm+'_mean']),'std':torch.from_numpy(norm[arm+'_std']),'target_scale':bank.manifest['target_scale'],'teacher_sha':V.BODY_SHA,'PCA_sha':V.PCA_SHA,'base_negative_slope':V.SLOPE};torch.save(artifact,d/'adapter.pt');man={'identity':e.identity,'init_sha':e.init_sha,'positive_lr_updates':e.step,'stats':e.stats,'model_sha':V.sha(d/'adapter.pt'),'fit_wall_s':time.monotonic()-start,'resumed_from_step':s,'resources':V.resources(),'loaded_modules':V.loaded_check()};V.atomic(d/'manifest.json',man);result=validate_arm(arm);print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
