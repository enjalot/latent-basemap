import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,tempfile,importlib.util,copy,json
from pathlib import Path
import numpy as np
import torch
from run_card019_arm import ROOT,OC,DATA,HEAD,sha,write,ACT,ParametricUMAP
from basemap.pumap.parametric_umap.models.mlp import ResidualBottleneckMLP
BR=OC/'band-investigation-20260911/pile-audit-v1'
def main():
 torch.set_num_threads(2);checks={}
 spec=importlib.util.spec_from_file_location('old_frozen_mlp',BR/'frozen_mlp.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 torch.manual_seed(19);a=old.ResidualBottleneckMLP(768,2048,2,3,.75);rng_a=torch.random.get_rng_state()
 torch.manual_seed(19);b=ResidualBottleneckMLP(768,2048,2,3,.75);rng_b=torch.random.get_rng_state()
 checks['default_init_and_rng_unchanged']=torch.equal(rng_a,rng_b) and all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items())
 x=torch.randn(8,768);checks['default_forward_bitwise']=torch.equal(a(x),b(x))
 p=ParametricUMAP.load(str(HEAD),device='cpu');features=np.load(BR/'features.npz');outputs=np.load(BR/'outputs.npz');x=torch.from_numpy(features['model_input_fp32'])
 with torch.inference_mode(): y=p.model(x)
 checks['old_checkpoint_defaults_relu']=p.final_activation=='relu' and isinstance(p.model.up[1],torch.nn.ReLU)
 checks['old_pile_matches_frozen']=torch.equal(y,torch.from_numpy(outputs['cpu_fp32']))
 leaky=copy.deepcopy(p);leaky.final_activation=ACT['leaky'];leaky.model.up[1]=torch.nn.LeakyReLU(.01)
 with torch.inference_mode(): z=leaky.model(x)
 checks['intervention_exposed']=not torch.equal(z,y)
 panel=np.load(BR/'frozen-panel.npz');pile_x=x[torch.from_numpy(panel['role']=='pile')]
 leaky.model.zero_grad(set_to_none=True);leaky.model(pile_x).sum().backward()
 checks['pile_gradient_reaches_earlier_weights']=bool(leaky.model.proj_in.weight.grad.abs().sum()>0)
 p.model.zero_grad(set_to_none=True);p.model(pile_x).sum().backward()
 checks['relu_pile_gradient_blocked']=bool(p.model.proj_in.weight.grad.abs().sum()==0)
 with tempfile.TemporaryDirectory() as td:
  for label,m in [('relu',p),('leaky',leaky)]:
   path=Path(td)/(label+'.pt');m.save(str(path));loaded=ParametricUMAP.load(str(path),device='cpu')
   with torch.inference_mode(): yy=loaded.model(x);xx=m.model(x)
   checks[label+'_save_load_exact']=torch.equal(yy,xx) and loaded.final_activation==m.final_activation
 ids=np.load(DATA/'draw_ids.npy');checks['fresh_reserve_excluded']=not np.isin(ids,np.load(OC.parent/'card012-pool/confirm_reserved.npy')).any()
 # Independently recompute a deterministic row panel from original source data.
 m=json.loads((DATA/'manifest.json').read_text());raw=np.load(m['source']['path'],mmap_mode='r');pca=np.load(m['pca']);ix=np.random.default_rng(19019).choice(len(ids),128,False)
 xx=torch.from_numpy(np.asarray(raw[ids[ix]],dtype='f4'));modelin=torch.nn.functional.normalize((xx-torch.from_numpy(pca['mean']))@torch.from_numpy(pca['components']),dim=1)
 stored=np.asarray(np.load(DATA/'substrate.f16.npy',mmap_mode='r')[ix]);err=float(np.max(np.abs(modelin.numpy()-stored.astype('f4'))))
 checks['stored_input_fidelity']=err<.0005
 checks['builder_source_unchanged']=sha(ROOT/'experiments/sandbox/build_card019_data.py')==m['builder_sha']
 for n,h in m['files'].items():checks['data_'+n]=sha(DATA/n)==h
 r={'PASS':all(checks.values()),'checks':checks,'input_fp16_max_error':err,'module_sha':sha(ROOT/'basemap/pumap/parametric_umap/models/mlp.py'),'core_sha':sha(ROOT/'basemap/pumap/parametric_umap/core.py')}
 write(OC/'card019-cpu-canary.json',r);print(json.dumps(r,indent=2));assert r['PASS']
if __name__=='__main__':main()
