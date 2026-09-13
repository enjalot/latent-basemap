"""Freeze two trained warm starts; PCA was fit on training outputs before this card."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,time
import numpy as np,torch
import card069_common as C
sys.path.insert(0,str(C.R))
from basemap.pumap.parametric_umap.core import ParametricUMAP
start=time.monotonic();torch.set_num_threads(2);C.source_check();C.input_check()
readout=C.O/'card067-scoring/training-readout.npz';rd=np.load(readout);U=rd['components'];center=rd['center'];assert U.shape==(8,3) and np.allclose(U.T@U,np.eye(3),atol=1e-12)
assert np.array_equal(rd['train_rows'],np.load(C.D/'supervised-selection.npz')['train_rows'])
paths={'continued':C.R.parent/'card065-train/random/model.pt','compressed':C.R.parent/'card067-train/eight/model.pt'}
checks={};receipts={};rows=rd['train_rows'][::29];X=np.array(np.load(C.D/'train.f16.npy',mmap_mode='r')[rows],dtype='f4');X/=np.linalg.norm(X,axis=1,keepdims=True).clip(1e-12);x=torch.from_numpy(X)
for arm,path in paths.items():
 dest=C.TD/arm;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'preparation.json').exists()
 parent=ParametricUMAP.load(str(path),device='cpu').model.eval();base=parent.state_dict();sd={k:v.clone() for k,v in base.items()}
 if arm=='compressed':
  sd['proj_out.weight']=(torch.from_numpy(U.T)@base['proj_out.weight'].double()).float();sd['proj_out.bias']=((base['proj_out.bias'].double()-torch.from_numpy(center))@torch.from_numpy(U)).float()
 p=ParametricUMAP.load(str(C.CHAMP),device='cpu');p.n_components=3;p._init_model(1536);p.model.load_state_dict(sd);m=p.model.eval();aa=[];bb=[]
 with torch.inference_mode():
  for lo in range(0,len(x),512):
   y=parent(x[lo:lo+512]).numpy().astype('f8');aa.append((y-center)@U if arm=='compressed' else y);bb.append(m(x[lo:lo+512]).numpy().astype('f8'))
 a=np.concatenate(aa);b=np.concatenate(bb);checks[arm+'_finite']=bool(np.isfinite(b).all());checks[arm+'_hidden_identical']=all(torch.equal(sd[k],base[k]) for k in base if not k.startswith('proj_out.'));checks[arm+'_direct_forward']=bool(np.all(np.abs(a-b)<=2e-5+2e-6*np.abs(a)))
 if arm=='continued':checks['control_full_state_identical']=C.state_sha(sd)==C.state_sha(base)
 assert all(checks.values()),checks
 state=C.state_sha(sd);torch.save({'READY':True,'model_state':sd,'prepared_state_sha':state,'arm':arm},dest/'prepared.pt')
 r={'READY':True,'arm':arm,'output_dim':3,'parent_sha':C.sha(path),'parent_path':str(path),'readout_sha':C.sha(readout),'selection_sha':C.sha(C.D/'supervised-selection.npz'),'hidden_tensors_identical_to_parent':True,'no_extra_rescaling':True,'max_forward_abs_error':float(np.max(np.abs(a-b))),'forward_training_rows':len(rows),'prepared_state_sha':state,'prepared_sha':C.sha(dest/'prepared.pt'),'runtime_sha':C.source_check()};C.write(dest/'preparation.json',r);receipts[arm]=r
C.write(C.O/'card069-initialization.json',{'PASS':all(checks.values()),'checks':checks,'arms':receipts,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check()});print('CPU INITIALIZATION PASS',checks,flush=True)
