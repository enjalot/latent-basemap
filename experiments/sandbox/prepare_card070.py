import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import time,torch
import card070_common as C
start=time.monotonic();C.source_check();C.input_check();dest=C.TD/'high';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'preparation.json').exists()
parent=C.R.parent/'card065-train/random/model.pt';source=C.R.parent/'card069-train/continued/prepared.pt';base=torch.load(parent,map_location='cpu',weights_only=False)['model_state_dict'];sd=torch.load(source,map_location='cpu',weights_only=False)['model_state'];assert C.state_sha(base)==C.state_sha(sd) and all(torch.equal(base[k],sd[k]) for k in base)
torch.save({'READY':True,'arm':'high','model_state':sd,'prepared_state_sha':C.state_sha(sd)},dest/'prepared.pt')
r={'READY':True,'arm':'high','parent_sha':C.sha(parent),'parent_path':str(parent),'shared_gentle_initialization_sha':C.sha(source),'full_parent_tensors_bit_identical':True,'prepared_state_sha':C.state_sha(sd),'prepared_sha':C.sha(dest/'prepared.pt'),'runtime_sha':C.source_check()};C.write(dest/'preparation.json',r);C.write(C.O/'card070-initialization.json',{'PASS':True,'arms':{'high':r},'wall_s':time.monotonic()-start,'runtime_sha':C.source_check()});print('INITIALIZATION PASS')
