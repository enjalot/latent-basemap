"""CPU initialization staging. Finish is prepared only after validated full U400."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,torch
import card085_common as C

def prepare(arm):
 assert C.read(C.O/'card085-history-compatibility.json')['CPU_PASS'], 'historical CPU compatibility STOP'
 C.source_check()
 if arm=='fresh':
  source=C.R.parent/'card015-init/init-card015-3d.pt';sd=torch.load(source,map_location='cpu',weights_only=False)['model_state']
  assert C.sha(source)=='3995f1dd65a6c427b32a2903794c27de365469e40d5381abdfbd1f78f952540b'
 else:
  from card085_fit import validate_arm
  assert validate_arm('fresh')['PASS'];source=C.TD/'fresh/model.pt';sd=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict']
 dest=C.TD/arm;dest.mkdir(parents=True,exist_ok=True)
 rec=dest/'preparation.json'
 if rec.exists():
  r=C.read(rec);assert r['parent_sha']==C.sha(source) and r['prepared_sha']==C.sha(dest/'prepared.pt') and r['runtime_sha']==C.source_check();return
 assert not (dest/'prepared.pt').exists(), 'orphan prepared artifact STOP'
 torch.save({'READY':True,'arm':arm,'model_state':sd,'prepared_state_sha':C.state_sha(sd)},dest/'prepared.pt')
 C.write(rec,{'READY':True,'arm':arm,'parent_path':str(source),'parent_sha':C.sha(source),'prepared_sha':C.sha(dest/'prepared.pt'),'prepared_state_sha':C.state_sha(sd),'runtime_sha':C.source_check(),'optimizer_reset':True,'optimizer_state_copied':False})
if __name__=='__main__':prepare(sys.argv[1] if len(sys.argv)>1 else 'fresh')
