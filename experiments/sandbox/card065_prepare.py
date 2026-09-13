"""Fixed-dose supervised starts, independent resumable optimizer, then affine scale matching."""
import sys,time,hashlib
from pathlib import Path
import numpy as np,torch
import card065_common as C
sys.path.insert(0,str(C.R));from basemap.pumap.parametric_umap.core import ParametricUMAP
def setup():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(C.PREP_SEED);torch.cuda.manual_seed_all(C.PREP_SEED);np.random.seed(C.PREP_SEED)
def prep_identity(arm,dose):
 return {'card':'065','phase':'supervised','arm':arm,'dose':dose,'lr':.001,'weight_decay':.01,'batch':C.PREP_BATCH,'seed':C.PREP_SEED,'precision':'FP32_no_AMP','loss':'mean coordinate MSE divided by shared_RMS squared','input_sha':C.sha(C.D/'inputs-manifest.json'),'scale_sha':C.sha(C.D/'scale-manifest.json'),'target_sha':C.sha(C.D/'pca-targets.f32.npy'),'selection_sha':C.sha(C.D/'supervised-selection.npz'),'fresh_init_sha':C.sha(C.D/'fresh-init.pt'),'template_sha':C.sha(C.CHAMP),'runtime_sha':C.source_check()}
def load_data():
 x=torch.tensor(np.array(np.load(C.D/'train.f16.npy',mmap_mode='r')),device='cuda');sel=np.load(C.D/'supervised-selection.npz');train=torch.tensor(sel['train_rows'],device='cuda');target=torch.tensor(np.load(C.D/'pca-targets.f32.npy'),device='cuda');perm=torch.tensor(sel['target_permutation'],device='cuda');return x,train,target,perm,sel['check_rows']
def fresh():
 m=ParametricUMAP.load(str(C.CHAMP),device='cuda').model.float();original=torch.load(C.D/'fresh-init.pt',map_location='cpu',weights_only=False)['model_state'];m.load_state_dict(original,strict=True);assert C.state_sha(m.state_dict()).startswith('5544a31160054bcc');return m
def load_resume(path,ident,m,opt,gen):
 ck=torch.load(path,map_location='cpu',weights_only=False)
 if ck['identity']!=ident:raise ValueError('card065 supervised admission-identity mismatch')
 assert 0<ck['step']<=ident['dose'];m.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);gen.set_state(ck['sampler_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);return ck['step']
def run(arm,dose,dest,data,*,resume=None,stop_at=None,save_steps=()):
 assert arm in ('pca','shuffled');setup();dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);ident=prep_identity(arm,dose);m=fresh().train();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,foreach=False);gen=torch.Generator(device='cuda').manual_seed(C.PREP_SEED);x,train,y,perm,_=data;r=C.read(C.D/'scale-manifest.json')['shared_RMS'];step=load_resume(resume,ident,m,opt,gen) if resume else 0;losses=[];draw_hash=hashlib.sha256()
 admission=dest/'prep-admission.json'
 if admission.exists():assert C.read(admission)==ident,'saved supervised admission mismatch'
 else:C.write(admission,ident)
 for step0 in range(step,stop_at or dose):
  ix=train[torch.randint(len(train),(C.PREP_BATCH,),device='cuda',generator=gen)];yy=y[ix if arm=='pca' else perm[ix]];pred=m(x[ix].float());loss=torch.mean(((pred-yy)/r)**2);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();step=step0+1;losses.append(float(loss.detach()));draw_hash.update(ix.cpu().numpy().tobytes())
  if step in save_steps or step==(stop_at or dose):
   payload={'identity':ident,'step':step,'model':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'optimizer':opt.state_dict(),'sampler_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()};p=dest/f'prep-step{step}.pt';tmp=p.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(p)
 free,total=torch.cuda.mem_get_info();assert (total-free)/2**30<30
 return m,{'identity':ident,'step':step,'state_sha':C.state_sha(m.state_dict()),'first_loss':losses[0] if losses else None,'last_loss':losses[-1] if losses else None,'draw_digest':draw_hash.hexdigest(),'optimizer':opt.state_dict()},gen
@torch.inference_mode()
def predict(m,x,ids):
 return np.concatenate([m(x[torch.as_tensor(ids[lo:lo+4096],device='cuda')].float()).cpu().numpy() for lo in range(0,len(ids),4096)]).astype('f8')
@torch.inference_mode()
def predict_amp(m,x,ids):
 out=[]
 with torch.autocast(device_type='cuda',dtype=torch.float16):
  for lo in range(0,len(ids),4096):out.append(m(x[torch.as_tensor(ids[lo:lo+4096],device='cuda')]).float().cpu().numpy())
 return np.concatenate(out).astype('f8')
def finish(arm,m,report,data,dest):
 x,train,y,perm,check=data;rows=train.cpu().numpy();r=C.read(C.D/'scale-manifest.json')['shared_RMS'];m.eval();before=predict(m,x,rows);center=before.mean(0);rms=float(np.sqrt(np.mean(np.sum((before-center)**2,axis=1))));factor=r/rms;assert rms>1e-8 and np.isfinite(factor) and factor<=1000,'degenerate start cannot be safely scale-matched'
 with torch.no_grad():m.proj_out.weight.mul_(factor);m.proj_out.bias.copy_((m.proj_out.bias.double()-torch.tensor(center,device='cuda'))*factor)
 after=predict(m,x,rows);actual=float(np.sqrt(np.mean(np.sum((after-after.mean(0))**2,axis=1))));assert abs(actual/r-1)<2e-4 and np.linalg.norm(after.mean(0))/r<2e-4
 yy=y[torch.as_tensor(check,device='cuda')].cpu().numpy().astype('f8');pred=predict(m,x,check);r2=float(1-np.sum((pred-yy)**2)/np.sum((yy-yy.mean(0))**2));amp=predict_amp(m,x,check);amp_r2=float(1-np.sum((amp-yy)**2)/np.sum((yy-yy.mean(0))**2));amp_train=predict_amp(m,x,rows);amp_rms=float(np.sqrt(np.mean(np.sum((amp_train-amp_train.mean(0))**2,axis=1))));ready=bool((arm!='pca' or min(r2,amp_r2)>=.90) and abs(amp_rms/r-1)<=.02)
 saved={'model_state':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'arm':arm,'prepared_state_sha':C.state_sha(m.state_dict()),'shared_RMS':r,'actual_train_RMS':actual,'check_PCA_R2':r2,'READY':ready};torch.save(saved,Path(dest)/'prepared.pt')
 report.pop('optimizer',None);report.update(arm=arm,READY=ready,check_PCA_R2=r2,check_PCA_AMP_R2=amp_r2,AMP_train_RMS=amp_rms,pre_affine_train_RMS=rms,affine_center=center.tolist(),affine_factor=factor,shared_RMS=r,actual_train_RMS=actual,center_error_R=float(np.linalg.norm(after.mean(0))/r),prepared_sha=C.sha(Path(dest)/'prepared.pt'),prepared_state_sha=saved['prepared_state_sha'],loaded_modules=C.loaded_modules());C.write(Path(dest)/'preparation.json',report);return report
