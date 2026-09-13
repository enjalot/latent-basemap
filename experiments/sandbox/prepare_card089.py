import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import time,torch
import card089_common as C
C.require_release()
start=time.monotonic();C.source_check();C.input_check();base=torch.load(C.CHAMP,map_location='cpu',weights_only=False)['model_state_dict'];receipts={}
for arm in C.ARMS:
 dest=C.TD/arm;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'preparation.json').exists():
  r=C.read(dest/'preparation.json');assert r['READY'] and r['runtime_sha']==C.source_check() and r['parent_sha']==C.sha(C.CHAMP) and r['prepared_sha']==C.sha(dest/'prepared.pt');receipts[arm]=r;continue
 sd={k:v.clone() for k,v in base.items()};assert all(torch.equal(base[k],sd[k]) for k in base)
 torch.save({'READY':True,'arm':arm,'model_state':sd,'prepared_state_sha':C.state_sha(sd)},dest/'prepared.pt');r={'READY':True,'arm':arm,'parent_sha':C.sha(C.CHAMP),'parent_path':str(C.CHAMP),'full_parent_tensors_bit_identical':True,'prepared_state_sha':C.state_sha(sd),'prepared_sha':C.sha(dest/'prepared.pt'),'runtime_sha':C.source_check()};C.write(dest/'preparation.json',r);receipts[arm]=r
assert len({r['prepared_state_sha'] for r in receipts.values()})==1;C.write(C.O/'card089-initialization.json',{'PASS':True,'arms':receipts,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check()});print('MATCHED INITIALIZATION PASS')
