"""CPU-only inputs/init/shape contract and small exact graph for real device resume checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import numpy as np,torch,faiss
import card038_validate as V
from card038_fit import configure_pumap
from basemap.pumap.parametric_umap.core import ParametricUMAP
R=Path(__file__).resolve().parents[2];O=V.OC;D=O/'card038-canary-data';D.mkdir(exist_ok=True);checks={};torch.set_num_threads(2);faiss.omp_set_num_threads(2)
checks['substrate_hash']=V.full_sha(V.SUB)==V.SUB_SHA256;checks['graph_hash']=V.full_sha(V.GRAPH)==V.GRAPH_SHA256;checks['radii_hash']=V.full_sha(V.RADII)==V.R_HALF_SHA256
r=np.load(V.RADII);parent=np.load(V.SB/'card013-radii/r_actual.npy');checks['radii_exact_sqrt_parent']=np.array_equal(r,np.sqrt(parent));checks['radii_finite_positive']=r.shape==(V.N,) and np.isfinite(r).all() and (r>0).all()
for arm in V.ARMS:
 sd=V.expected_init(arm);p=ParametricUMAP.load(str(V.CHAMPION),device='cpu');configure_pumap(p,arm,V.DOSE[arm],r,{'test':'CPU-config'},V.SNAP[arm]);p._init_model(1536);p.model.load_state_dict(sd)
 checks[arm+'_actual_parameter_count']=sum(t.numel() for t in p.model.parameters())==V.NPARAM[arm];checks[arm+'_actual_width']=p.model.proj_in.out_features==V.WIDTH[arm] and p.model.proj_out.out_features==3;checks[arm+'_other_interventions_off']=not p.midnear_enabled and p.density_weight==p.correlation_weight==p.anchor_hold_weight==p.replay_weight==p.deriv_weight==0
 with torch.inference_mode():y=p.model(torch.ones(7,1536));checks[arm+'_finite_forward']=y.shape==(7,3) and bool(torch.isfinite(y).all())
x=np.array(np.load(V.SUB,mmap_mode='r')[:512],'f4',copy=True);idx=faiss.IndexFlatL2(1536);idx.add(x);dist,nn=idx.search(x,16);near=np.stack([row[row!=i][:15] for i,row in enumerate(nn)]);checks['small_graph_no_self']=not (near==np.arange(512)[:,None]).any();checks['small_graph_exact15']=near.shape==(512,15)
np.save(D/'X.npy',x);np.save(D/'radii.npy',r[:512]);np.savez(D/'edges.npz',sources=np.repeat(np.arange(512,dtype='i8'),15),targets=near.astype('i8').reshape(-1),weights=np.ones(512*15,'f4'))
result={'PASS':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'canary_data':{n:V.full_sha(D/n) for n in ['X.npy','radii.npy','edges.npz']},'scope':'CPU exact identities, real architecture tensors/config, existing full training data; fixed512-row graph only for device resume canary. NoGPU or new scientific outcome.'};(O/'card038-foundation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['PASS']
