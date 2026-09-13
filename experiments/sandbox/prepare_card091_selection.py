"""Root-invoked only after all090 endpoints; no forwards or raw-bank hashing."""
import os,sys,subprocess,time
import card091_common as C
PY='/home/enjalot/code/latent-basemap/.venv/bin/python'
def build():
 start=time.monotonic();assert not (C.D/'selection.json').exists(),'existing selection requires root archive/review'
 ex=C.read(C.O/'card090-execution.json');arms={}
 for a in C.ARMS:
  p=C.TD/a;arms[a]=tuple(C.read(p/n) for n in ['validation.json','manifest.json','admission.json','preparation.json'])
 C.validate_completed(ex,arms)
 # Actual canonical validator, in its own original runtime. No import substitution.
 code="import sys,json;sys.path[:0]=sys.argv[1:3];import card090_fit as F;print(json.dumps({a:F.validate_arm(a) for a in ['ranked43','uniform43','ranked44','uniform44']}))"
 result=subprocess.run([PY,'-c',code,str(C.R090/'experiments/sandbox'),str(C.R090)],capture_output=True,text=True,check=True,timeout=600,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
 import json
 deep=json.loads(result.stdout.strip().splitlines()[-1])
 assert all(deep[a]==arms[a][0] for a in C.ARMS),'canonical090 validation differs from producer'
 bindings={};retained={}
 for a in C.ARMS:
  p=C.TD/a;paths=[p/n for n in ['model.pt','manifest.json','validation.json','admission.json','preparation.json','prepared.pt']]
  paths += [p/f'model-step{step}.pt' for step in [20000,40000,60000]]+[p/'ckpts'/f'ckpt-step{step}.pt' for step in [20000,40000,60000]]
  kept=sorted((p/'ckpts').glob('*.pt'));assert any(x.name.startswith('ckpt-epoch') for x in kept),'missing retained epoch'
  paths+=kept;retained[a]=[str(x) for x in kept]
  for x in paths:bindings[str(x)]=C.sha(x)
  assert bindings[str(p/'model.pt')]==arms[a][0]['model_sha'],'model changed after validation'
 for p in [C.O/'card090-execution.json',C.O/'card090-release.json',C.R090/'card090-runtime-sha.json']:
  bindings[str(p)]=C.sha(p)
 assert bindings[str(C.R090/'card090-runtime-sha.json')]==C.RUNTIME090,'090 source manifest differs'
 s=C.read(C.O/'card078-full/selection.json')
 for k in list(s):
  if k.startswith('reused_'):del s[k]
 s.update(status='FOUR_CANONICAL090_ENDPOINTS_BOUND_NOT_PROJECTED',models={a:{'path':str(C.TD/a/'model.pt'),'sha':arms[a][0]['model_sha']} for a in C.ARMS},runtime=str(C.R),runtime_manifest_sha=C.source_check(),baseline_runtime_sha=C.RUNTIME090,protocol_sha=C.sha(C.O/'card091-full-reference-seeds.md'),quality_sha=C.sha(C.O/'card091-quality-prereg.md'),endpoint_bindings=bindings,retained_checkpoint_files=retained,canonical090_validation=deep,limits='Four fixed-parent continuation heads; quality PASS is not eligibility; historical42 score context only')
 s.pop('at',None);C.validate_selection(s);C.verify_sources(s)
 C.write(C.D/'selection.json',s)
 C.write(C.D/'selection-build.json',{'PASS':True,'selection_sha':C.sha(C.D/'selection.json'),'runtime_sha':C.source_check(),'cpu_wall_s':time.monotonic()-start,'no_forward':True,'quality_not_consulted':True,'cpu_budget_note':'Root must include this preparation CPU cost in the single091 CPU allocation; no new allocation is created here.'})
 return s
if __name__=='__main__':build()
