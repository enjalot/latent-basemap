"""CPU-only output-dimension intervention; no supervision or evaluation rows."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,time,math
import numpy as np,torch
import card067_common as C
sys.path.insert(0,str(C.R))
from basemap.pumap.parametric_umap.core import ParametricUMAP
start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(67067);C.source_check();C.input_check();dest=C.TD/'eight';dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'preparation.json').exists(),'immutable preparation already exists'
original=C.R.parent/'card065-train/random/prepared.pt';base=torch.load(original,map_location='cpu',weights_only=False)['model_state'];sd={k:v.clone() for k,v in base.items()}
extra=torch.nn.Linear(2048,5);sd['proj_out.weight']=torch.cat([sd['proj_out.weight'],extra.weight.detach()],0);sd['proj_out.bias']=torch.cat([sd['proj_out.bias'],extra.bias.detach()],0)
p=ParametricUMAP.load(str(C.CHAMP),device='cpu');p.n_components=8;p._init_model(1536);p.model.load_state_dict(sd);m=p.model.eval();X=np.load(C.D/'train.f16.npy',mmap_mode='r');rows=np.load(C.D/'supervised-selection.npz')['train_rows'];out=[]
with torch.inference_mode():
 for lo in range(0,len(rows),8192):out.append(m(torch.from_numpy(np.array(X[rows[lo:lo+8192]],dtype='f4'))).numpy())
y=np.concatenate(out).astype('f8');center=y.mean(0);rms=float(np.sqrt(((y-center)**2).sum(1).mean()));target=C.read(C.D/'scale-manifest.json')['shared_RMS'];factor=target/rms;assert np.isfinite(rms) and rms>1e-8 and 0<factor<1000
with torch.no_grad():m.proj_out.weight.mul_(factor);m.proj_out.bias.copy_((m.proj_out.bias.double()-torch.from_numpy(center)).mul(factor).float())
# Independent second forward validates actual transformed coordinates, not affine arithmetic alone.
out=[]
with torch.inference_mode():
 for lo in range(0,len(rows),8192):out.append(m(torch.from_numpy(np.array(X[rows[lo:lo+8192]],dtype='f4'))).numpy())
y2=np.concatenate(out).astype('f8');actual=float(np.sqrt(((y2-y2.mean(0))**2).sum(1).mean()));err=float(np.linalg.norm(y2.mean(0))/target);assert abs(actual/target-1)<2e-4 and err<2e-4
new=m.state_dict();assert all(torch.equal(new[k],base[k]) for k in base if not k.startswith('proj_out.'))
assert all(torch.isfinite(v).all() for v in new.values());assert new['proj_out.weight'].shape==(8,2048)
state=C.state_sha(new);torch.save({'READY':True,'model_state':new,'prepared_state_sha':state,'arm':'eight','shared_RMS':target},dest/'prepared.pt')
r={'READY':True,'runtime_sha':C.source_check(),'arm':'eight','output_dim':8,'original_3d_prepared_sha':C.sha(original),'original_3d_endpoint_sha':C.sha(C.R.parent/'card065-train/random/model.pt'),'added_output_seed':67067,'training_rows_n':len(rows),'selection_sha':C.sha(C.D/'supervised-selection.npz'),'input_sha':C.sha(C.D/'inputs-manifest.json'),'shared_RMS':target,'actual_train_RMS':actual,'center_error_R':err,'pre_affine_RMS':rms,'affine_factor':factor,'affine_center':center.tolist(),'hidden_tensors_bit_identical':True,'prepared_sha':C.sha(dest/'prepared.pt'),'prepared_state_sha':state,'cpu_s':time.monotonic()-start,'scope':'Five independent random output rows appended to065random start; common hidden weights, first3 rows before affine. All8 centered/scaled on same240Ktraining rows; graph outputdimension is treatment. No new loss or supervision.'};C.write(dest/'preparation.json',r);C.write(C.O/'card067-initialization.json',r);print('READY',actual,err,flush=True)
