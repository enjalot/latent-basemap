"""Read-only historical compatibility gate; never imports CUDA or edits controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib
from pathlib import Path
import torch,numpy as np
R=Path(__file__).resolve().parents[2];SB=R.parent;O=SB/'overseer-codex'
sys.path.insert(0,str(R))
from basemap.pumap.parametric_umap.core import ParametricUMAP

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(sd):
 h=hashlib.sha256()
 for k in sorted(sd):h.update(k.encode());h.update(sd[k].cpu().numpy().tobytes())
 return h.hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
 checks={};evidence={};torch.set_num_threads(2)
 for card,sub,step,lr in [('023','actual3d',400000,.001),('075','uniform',60000,.0001)]:
  base=SB/f'card{card}-train';m=json.loads((base/('manifest-actual3d.json' if card=='023' else 'uniform/manifest.json')).read_text());ck=load(base/sub/'ckpts'/f'ckpt-step{step}.pt');ts=ck['train_stats'];g=ck['optimizer']['param_groups'][0]
  checks[card+'_actual_lr']=ck['config']['lr_schedule']=='constant' and ts['lr_used_min']==ts['lr_used_max']==g['lr']==g['initial_lr']==lr and ts['lr_used_count']==step and ts['scheduler_steps']==0
  checks[card+'_dose']=ts['executed_iters']==ts['positive_lr_optimizer_steps']==ts['optimizer_steps_succeeded']==step and all(float(v['step'])==step for v in ck['optimizer']['state'].values())
  checks[card+'_adamw']=g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==.01 and g['decoupled_weight_decay'] and not g['amsgrad']
  checks[card+'_precision']=ts['use_amp'] and ts['amp_dtype']=='float16' and ts['pipeline_x_residency']=='device_fp16'
  mods=m['loaded_modules'].get('basemap_modules',m['loaded_modules']);checks[card+'_loaded_hashes']=all(sha(v['path']).startswith(v.get('sha16',v.get('sha',''))) for v in mods.values() if isinstance(v,dict) and 'path' in v)
  model=load(base/('model-actual3d.pt' if card=='023' else 'uniform/model.pt'));checks[card+'_endpoint']=all(torch.equal(ck['model'][k],model['model_state_dict'][k]) for k in ck['model']) and sha(base/('model-actual3d.pt' if card=='023' else 'uniform/model.pt'))==m.get('model_file_sha256',m.get('model_sha'))
  checks[card+'_recipe']=model['fneg_weight']==1. and model['neg_tanh_gamma']==4. and model['pos_ratio']==.1 and model['rankneg_window']==(500000 if card=='023' else 0)
  evidence[card]={'lr':lr,'optimizer_group':g,'amp_skips':ts['amp_overflow_skips'],'attempts':ts['attempted_batches'],'model_sha':m.get('model_file_sha256',m.get('model_sha')),'model_config':{k:v for k,v in model.items() if k not in ('model_state_dict','model_state') and isinstance(v,(str,int,float,bool,type(None)))}}
 a=SB/'card023-code/basemap/pumap/parametric_umap';b=SB/'card075-code/basemap/pumap/parametric_umap'
 hook="                # Card054 isolated default-off stateless landmark objective.\n                _lmc_hook = getattr(self, '_card054_lmc', None)\n                if _lmc_hook is not None:\n                    loss = loss + _lmc_hook(self.model, global_step)\n\n"
 checks['core_only_default_off_hook']=(b/'core.py').read_text().replace(hook,'')==(a/'core.py').read_text()
 for f in ('datasets/edge_list_dataset.py','models/mlp.py','utils/losses.py','utils/data_prefetcher.py'):
  checks['unchanged_'+f]=sha(a/f)==sha(b/f)==sha(R/'basemap/pumap/parametric_umap'/f)
 checks['proposed_core_is075']=sha(b/'core.py')==sha(R/'basemap/pumap/parametric_umap/core.py')
 champion=SB/'dino-arrival-t0/champion-bs16k/model.pt';i2=load('/data/latent-basemap/substrates/card010-adaptive/init-card010.pt')['model_state'];i3path=SB/'card015-init/init-card015-3d.pt';i3=load(i3path)['model_state']
 p=ParametricUMAP.load(str(champion),device='cpu');p.model=None;torch.manual_seed(42);np.random.seed(42);p._init_model(1536);checks['reconstructed_untrained2d']=all(torch.equal(v,p.model.state_dict()[k]) for k,v in i2.items())
 p.model=None;p.n_components=3;torch.manual_seed(42);np.random.seed(42);p._init_model(1536);fresh=p.model.state_dict();rebuilt={k:torch.cat([i2[k],v[2:3]],0) if k in ('proj_out.weight','proj_out.bias') else i2[k].clone() for k,v in fresh.items()}
 checks['reconstructed_untrained3d']=all(torch.equal(rebuilt[k],v) for k,v in i3.items());checks['init_file_and_state']=sha(i3path)=='3995f1dd65a6c427b32a2903794c27de365469e40d5381abdfbd1f78f952540b' and state(i3).startswith('5544a31160054bcc')
 parent=load(SB/'card023-train/model-actual3d.pt')['model_state_dict'];prepared=load(SB/'card075-train/uniform/prepared.pt')['model_state'];checks['075_exact023_parent']=all(torch.equal(v,prepared[k]) for k,v in parent.items())
 adm23=json.loads((SB/'card023-train/admission-actual3d.json').read_text())['card012_identity'];adm75=json.loads((SB/'card075-train/uniform/admission.json').read_text());checks['graph_radius_match']=adm23['edges_sha256']==adm75['graph_sha'] and adm23['radii_sha256']==adm75['radii_sha']
 out={'CPU_PASS':all(checks.values()),'READY_FOR_GPU':False,'device_replay':'PENDING_ROOT_RELEASE: original/proposed same-device replay and epoch boundary required','checks':checks,'evidence':evidence,'init_sha':sha(i3path),'init_state_sha':state(i3),'validation_rows':10660}
 (O/'card085-history-compatibility.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'CPU_PASS':out['CPU_PASS'],'checks':checks},indent=2));return 0 if out['CPU_PASS'] else 3
if __name__=='__main__':raise SystemExit(main())
