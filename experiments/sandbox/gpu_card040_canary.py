"""Actual bank/teacher pairing, small-readout gradients and full-state resume twins."""
from pathlib import Path
import sys,json,tempfile,copy,shutil
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import numpy as np,torch
import card040_common as V
from card040_engine import Bank,Engine,validate_payload

def same(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 V.device_setup();bank=Bank();checks={};t=V.load_body('teacher','cuda');l=V.load_body('leaky','cuda');tsha=V.state_sha(t.state_dict());lsha=V.state_sha(l.state_dict());idx=torch.cat([bank.inactive[torch.arange(128)%len(bank.inactive)],bank.active[:128]]);x=torch.from_numpy(np.array(np.load(V.DATA/'train-input.f32.npy',mmap_mode='r')[idx.numpy()])).cuda()
 with torch.inference_mode():
  ty,f,z=V.frozen_views(t,x);c=(ty.double()+V.ALPHA*(l(x).double()-ty.double())).float();delta=(c.double()-f.double()).float();stored=bank.pre[idx.cuda()];checks['independent_frozen_feature_row_join']=torch.allclose(z,stored,rtol=2e-6,atol=2e-5);raw_delta=torch.from_numpy(np.load(V.DATA/'delta.f32.npy')[idx.numpy()]).cuda();checks['independent_frozen_target_row_join']=torch.allclose(delta,raw_delta,rtol=2e-5,atol=2e-5);checks['wrong_target_pairing_fails']=not torch.allclose(delta,raw_delta.roll(1,0),rtol=2e-5,atol=2e-5)
  xpre,_=bank.batch('pre',idx);xpost,_=bank.batch('post',idx);checks['inactive_post_identical']=torch.equal(xpost[:128],xpost[:1].expand(128,-1));checks['inactive_pre_distinct']=len(torch.unique(xpre[:128],dim=0))>1
  for a in V.ARMS:
   e=Engine(a,bank,64);pred=e.model(xpre if a=='pre' else xpost);mapped=(f.double()+bank.manifest['target_scale']*pred.double()).float();checks[a+'_zero_readout_equals_F']=torch.equal(mapped,f)
 # Actual CPU64 fixed-input forward contract, including inactive rows; input precision is fixed.
 tc=copy.deepcopy(t).cpu().double();lc=copy.deepcopy(l).cpu().double()
 with torch.inference_mode():
  tx=tc(x.cpu().double());lx=lc(x.cpu().double());bound=2e-4+2e-5*tx.abs();checks['teacher_GPU_CPU64']=bool(((ty.cpu().double()-tx).abs()<=bound).all());checks['leaky_GPU_CPU64']=bool(((l(x).cpu().double()-lx).abs()<=2e-4+2e-5*lx.abs()).all())
 del tc,lc
 # Independently recompute normalization using a centered second pass (builder uses raw moments).
 norms=np.load(V.DATA/'normalization.npz');features=np.load(V.DATA/'pre.f32.npy',mmap_mode='r')
 for a in V.ARMS:
  total=np.zeros(2048,'f8')
  for j in range(0,len(features),8192):
   b=np.array(features[j:j+8192],'f8');b=np.maximum(b,0) if a=='post' else b;total+=b.sum(0)
  mean=total/len(features);ss=np.zeros(2048,'f8')
  for j in range(0,len(features),8192):
   b=np.array(features[j:j+8192],'f8');b=np.maximum(b,0) if a=='post' else b;ss+=np.square(b-mean).sum(0)
  std=np.maximum(np.sqrt(ss/len(features)),.001);checks[a+'_full_bank_normalization']=np.allclose(mean,norms[a+'_mean'],rtol=1e-6,atol=1e-6) and np.allclose(std,norms[a+'_std'],rtol=1e-6,atol=1e-6)
 inits={};probes={}
 with tempfile.TemporaryDirectory(dir=str(V.SB),prefix='card040-canary-') as temp:
  td=Path(temp)
  for a in V.ARMS:
   e=Engine(a,bank,64);inits[a]=e.init_sha
   while e.step<7:e.advance()
   ck=td/(a+'-mid.pt');e.save(ck)
   while e.step<64:e.advance()
   full=copy.deepcopy(e.payload());probes[a]=full['stats']['row_probes'];r=Engine(a,bank,64);r.restore(ck)
   while r.step<64:r.advance()
   checks[a+'_resume_all_state_bitwise']=same(full,r.payload());checks[a+'_student_gradient_nonzero']=any(p.grad is not None and float(p.grad.abs().sum())>0 for p in r.model.parameters())
   bad=torch.load(ck,map_location='cpu',weights_only=False)
   for name,value in [('basis','other'),('bank_sha','wrong'),('normalization_sha','wrong'),('dose',65)]:
    corrupt=copy.deepcopy(bad);corrupt['identity'][name]=value;path=td/(a+'-bad.pt');torch.save(corrupt,path);fresh=Engine(a,bank,64);before=V.state_sha(fresh.model.state_dict())
    try:fresh.restore(path)
    except AssertionError as ex:checks[a+'_reject_'+name]='admission-identity mismatch' in str(ex) and V.state_sha(fresh.model.state_dict())==before
    else:checks[a+'_reject_'+name]=False
  # Exercise the actual bundle validator on an isolated copy; never mutate the production bank.
  target=td/'bank-copy';target.mkdir()
  for p in V.DATA.iterdir():
   if not p.is_file():continue
   if p.name in ['delta.f32.npy','normalization.npz']:shutil.copy2(p,target/p.name)
   else:(target/p.name).hardlink_to(p)
  original=V.DATA;V.DATA=target
  try:
   V.validate_bank();checks['bundle_copy_valid']=True
   for name in ['delta.f32.npy','normalization.npz']:
    p=target/name;data=p.read_bytes();p.write_bytes(data[:-1]+bytes([data[-1]^1]))
    try:V.validate_bank()
    except AssertionError as ex:checks[name+'_corrupt_reject']='bank hash' in str(ex)
    else:checks[name+'_corrupt_reject']=False
    p.write_bytes(data)
  finally:V.DATA=original
 checks['same_readout_init']=len(set(inits.values()))==1;checks['same_actual_row_target_probes']=probes['pre']==probes['post'];checks['teacher_parameters_unchanged']=tsha==V.state_sha(t.state_dict()) and lsha==V.state_sha(l.state_dict());checks['teacher_no_gradients']=all(p.grad is None and not p.requires_grad for m in [t,l] for p in m.parameters());res={'PASS':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'runtime_manifest_sha':V.runtime_check(),'bank_sha':bank.sha,'resources':V.resources(),'scope':'Actual device bank/teacher target join, pre/post features, zero correction, student-only gradients, fullmodel/Adam/RNG/exposure resume, wrongadmission before restore and bundle-corruption controls. Noproductionfit orquality result.'};V.atomic(V.O/'card040-canary.json',res);print(json.dumps(res));assert res['PASS']
if __name__=='__main__':main()
