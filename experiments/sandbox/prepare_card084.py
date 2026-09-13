import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import time,torch
import card084_common as C
start=time.monotonic();assert not (C.O/'card084-release.json').exists(), 'root release already frozen';C.source_check();C.input_check();base=torch.load(C.CHAMP,map_location='cpu',weights_only=False)['model_state_dict'];receipts={}
for arm in C.ARMS:
 dest=C.TD/arm;dest.mkdir(parents=True,exist_ok=True);sd={k:v.clone() for k,v in base.items()};assert all(torch.equal(base[k],sd[k]) for k in base)
 if (dest/'prepared.pt').exists():
  prior=torch.load(dest/'prepared.pt',map_location='cpu',weights_only=False)
  assert prior['READY'] and prior['prepared_state_sha']==C.state_sha(sd)
  assert set(prior['model_state'])==set(sd) and all(torch.equal(prior['model_state'][k],sd[k]) for k in sd)
 else:torch.save({'READY':True,'arm':arm,'model_state':sd,'prepared_state_sha':C.state_sha(sd)},dest/'prepared.pt')
 r={'READY':True,'arm':arm,'parent_sha':C.sha(C.CHAMP),'parent_path':str(C.CHAMP),'full_parent_tensors_bit_identical':True,'prepared_state_sha':C.state_sha(sd),'prepared_sha':C.sha(dest/'prepared.pt'),'runtime_sha':C.source_check()};C.write(dest/'preparation.json',r);receipts[arm]=r
assert len({r['prepared_state_sha'] for r in receipts.values()})==1;C.write(C.O/'card084-initialization.json',{'PASS':True,'arms':receipts,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check()});print('MATCHED INITIALIZATION PASS')
