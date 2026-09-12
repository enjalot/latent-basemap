import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,copy,tempfile,importlib.util
from _paths import ensure_paths
ensure_paths()
import torch,numpy as np
from run_card030_arm import ROOT,OC,DATA,HEAD,sha,write,ParametricUMAP,state_sha,ACT
from card030_resume import validate_resume_payload

def main():
 torch.set_num_threads(2);checks={};raw=torch.load(DATA/'init-card010.pt',map_location='cpu',weights_only=False);init=raw['model_state'];checks['init_file']=sha(DATA/'init-card010.pt')=='7d313c265cb659c954951d79ec395fa4d267f0f34a0bfb6aafe0dd23d5985825'
 gen=torch.Generator().manual_seed(30030);x=torch.randn(64,1536,generator=gen);x=torch.nn.functional.normalize(x,dim=1);models={}
 for a in ['relu','leaky']:
  p=ParametricUMAP.load(str(HEAD),device='cpu');p.final_activation=ACT[a];p._init_model(1536);p.model.load_state_dict(init);p.learning_rate=.001;p.lr_schedule='constant';models[a]=p
 checks['same_actual_parameters']=state_sha(models['relu'].model.state_dict())==state_sha(models['leaky'].model.state_dict())
 # Compare actual old runtime MLP and explicit-ReLU new MLP on identical weights/inputs.
 f=ROOT.parent/'card023-code/basemap/pumap/parametric_umap/models/mlp.py';sp=importlib.util.spec_from_file_location('card030_old_mlp',f);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);old=mod.ResidualBottleneckMLP(1536,2048,2,num_layers=3,neck_fraction=.75);old.load_state_dict(init)
 with torch.inference_mode():
  y={a:p.model(x) for a,p in models.items()};checks['default_relu_bitexact']=torch.equal(y['relu'],old(x));checks['leaky_changes_function']=not torch.equal(y['relu'],y['leaky'])
  with tempfile.TemporaryDirectory() as td:
   for a,p in models.items():
    path=Path(td)/f'{a}.pt';p.save(str(path));r=ParametricUMAP.load(str(path),device='cpu');checks[a+'_save_load']=r.final_activation==ACT[a] and torch.equal(r.model(x),y[a])
 for a,p in models.items():
  q=torch.tensor([-2.,-1.,0.,1.,2.],requires_grad=True);p.model.up[1](q).sum().backward();checks[a+'_negative_branch_gradient']=torch.equal(q.grad[:2],torch.full((2,),0. if a=='relu' else .01))
 # The previously completed real019 checkpoint tests schema logic with LR adapted only in metadata.
 ck=torch.load(ROOT.parent/'card019-train/leaky/ckpt/ckpt-step20000.pt',map_location='cpu',weights_only=False);ck['config']['learning_rate']=.001
 for g in ck['optimizer']['param_groups']:g['lr']=.001
 for key in ['base_lrs','_last_lr']:ck['scheduler'][key]=[.001 for _ in ck['scheduler'][key]]
 validate_resume_payload(ck,20000);checks['actual_checkpoint_schema']=True
 for key in ['optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank']:
  damaged=dict(ck);damaged.pop(key,None)
  try:validate_resume_payload(damaged,20000);checks['missing_'+key+'_rejects']=False
  except (AssertionError,KeyError,TypeError):checks['missing_'+key+'_rejects']=True
 d=copy.deepcopy(ck);next(iter(d['optimizer']['state'].values())).pop('exp_avg')
 try:validate_resume_payload(d,20000);checks['missing_Adam_moment_rejects']=False
 except AssertionError:checks['missing_Adam_moment_rejects']=True
 d=copy.deepcopy(ck);d['config']['final_activation']='unknown'
 try:validate_resume_payload(d,20000);checks['wrong_activation_rejects']=False
 except AssertionError:checks['wrong_activation_rejects']=True
 r={'PASS':all(checks.values()),'checks':checks,'n_checks':len(checks),'core_sha':sha(ROOT/'basemap/pumap/parametric_umap/core.py'),'module_sha':sha(ROOT/'basemap/pumap/parametric_umap/models/mlp.py'),'source_sha':sha(__file__),'scope':'CPU activation/default/save-load checks; actual historical checkpoint schema validation. Genuine new GPU resume still required.'};write(OC/'card030-cpu-canary.json',r);print(r['PASS'],r['n_checks']);assert r['PASS']
if __name__=='__main__':main()
