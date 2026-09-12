"""Small FP32 readout fit; full frozen feature bank shared across real canary/preflight/training."""
from pathlib import Path
import hashlib,copy
import numpy as np,torch
import card040_common as V
class Bank:
 def __init__(self):
  V.device_setup();V.validate_inputs();self.manifest=V.validate_bank();self.sha=V.sha(V.DATA/'bank.json');self.pre=torch.from_numpy(np.array(np.load(V.DATA/'pre.f32.npy',mmap_mode='r'))).cuda();self.target=torch.from_numpy(np.array(np.load(V.DATA/'delta.f32.npy'))).cuda()/self.manifest['target_scale'];self.ids=np.load(V.DATA/'draw_ids.npy');self.inactive=torch.from_numpy(np.load(V.DATA/'inactive-rows.npy'));self.active=torch.from_numpy(np.load(V.DATA/'active-rows.npy'));z=np.load(V.DATA/'normalization.npz');self.norm={a:(torch.from_numpy(z[a+'_mean']).cuda(),torch.from_numpy(z[a+'_std']).cuda()) for a in V.ARMS};src=np.load(V.DATA/'draw_source.npy');self.sources,self.source_code=np.unique(src,return_inverse=True)
 def batch(self,basis,idx):
  j=idx.cuda();x=self.pre.index_select(0,j);x=x.relu() if basis=='post' else x;mean,std=self.norm[basis];return (x-mean)/std,self.target.index_select(0,j)
class Engine:
 def __init__(self,basis,bank,dose=V.DOSE):
  assert basis in V.ARMS;self.basis=basis;self.bank=bank;self.dose=dose;torch.manual_seed(V.SEED);torch.cuda.manual_seed_all(V.SEED);self.model=V.Readout().cuda();init=torch.load(V.DATA/'init.pt',map_location='cpu',weights_only=False)['model'];self.model.load_state_dict(init);self.init_sha=V.state_sha(self.model.state_dict());self.opt=torch.optim.AdamW(self.model.parameters(),lr=V.LR,weight_decay=V.WD);self.gen=torch.Generator(device='cpu').manual_seed(V.SEED);self.step=0;self.stats={'updates':0,'rows':0,'inactive_rows':0,'active_rows':0,'source_exposure':[0]*len(bank.sources),'row_probes':[],'lr_used_min':None,'lr_used_max':None,'loss_first':None,'loss_last':None}
  self.identity=V.identity(basis,dose);assert self.init_sha==self.identity['init_sha']
 def next_indices(self):
  i=self.bank.inactive[torch.randint(0,len(self.bank.inactive),(V.BATCH//2,),generator=self.gen)];a=self.bank.active[torch.randint(0,len(self.bank.active),(V.BATCH//2,),generator=self.gen)];return torch.cat([i,a])
 def advance(self):
  assert self.step<self.dose;idx=self.next_indices();x,y=self.bank.batch(self.basis,idx);self.opt.zero_grad(set_to_none=True);pred=self.model(x);loss=(pred-y).square().mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),V.CLIP,error_if_nonfinite=True);assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters());assert self.opt.param_groups[0]['lr']==V.LR;self.opt.step();self.step+=1;s=self.stats;s['updates']=self.step;s['rows']+=len(idx);s['inactive_rows']+=V.BATCH//2;s['active_rows']+=V.BATCH//2;s['source_exposure']=(np.array(s['source_exposure'])+np.bincount(self.bank.source_code[idx.numpy()],minlength=len(self.bank.sources))).tolist();s['lr_used_min']=s['lr_used_max']=V.LR
  if s['loss_first'] is None:s['loss_first']=float(loss.detach())
  s['loss_last']=float(loss.detach())
  if self.step in [1,5000,10000,15000,20000]:
   ids=self.bank.ids[idx.numpy()];s['row_probes'].append({'step':self.step,'ordered_global_ids_sha':hashlib.sha256(ids.tobytes()).hexdigest(),'ordered_target_sha':hashlib.sha256(y.detach().cpu().numpy().tobytes()).hexdigest()})
  return float(loss.detach())
 def payload(self):
  return {'schema':'card040-resumable-v1','identity':self.identity,'step':self.step,'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),'row_rng':self.gen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'stats':self.stats}
 def save(self,path):
  p=Path(path);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix('.tmp');torch.save(self.payload(),t);t.replace(p)
 def restore(self,path):
  p=torch.load(path,map_location='cpu',weights_only=False);validate_payload(p,self.identity);self.model.load_state_dict(p['model']);self.opt.load_state_dict(p['optimizer']);self.gen.set_state(p['row_rng']);torch.set_rng_state(p['torch_rng']);torch.cuda.set_rng_state_all(p['cuda_rng']);self.step=p['step'];self.stats=copy.deepcopy(p['stats'])
def validate_payload(p,identity):
 assert p['schema']=='card040-resumable-v1' and p['identity']==identity,'admission-identity mismatch';step=p['step'];assert isinstance(step,int) and 0<step<=identity['dose'];s=p['stats'];assert s['updates']==step and s['rows']==step*V.BATCH and s['inactive_rows']==s['active_rows']==step*V.BATCH//2 and sum(s['source_exposure'])==s['rows'];assert s['lr_used_min']==s['lr_used_max']==V.LR;assert np.isfinite(s['loss_first']) and np.isfinite(s['loss_last']);expected=torch.load(V.DATA/'init.pt',map_location='cpu',weights_only=False)['model'];assert set(p['model'])==set(expected) and all(p['model'][k].shape==expected[k].shape and torch.isfinite(p['model'][k]).all() for k in expected)
 groups=p['optimizer']['param_groups'];states=p['optimizer']['state'];ids=[i for g in groups for i in g['params']];assert len(ids)==len(expected) and len(set(ids))==len(ids) and set(ids)==set(states)
 for g in groups:assert g['lr']==V.LR and g['weight_decay']==V.WD
 for i,t in zip(ids,p['model'].values()):
  st=states[i];assert float(st['step'])==step
  for k in ['exp_avg','exp_avg_sq']:assert st[k].shape==t.shape and torch.isfinite(st[k]).all()
 torch.Generator(device='cpu').set_state(p['row_rng']);torch.Generator(device='cpu').set_state(p['torch_rng']);assert isinstance(p['cuda_rng'],list) and p['cuda_rng'] and all(v.dtype==torch.uint8 and v.numel()==16 for v in p['cuda_rng']);assert [x['step'] for x in s['row_probes']]==[x for x in [1,5000,10000,15000,20000] if x<=step]
 return True
