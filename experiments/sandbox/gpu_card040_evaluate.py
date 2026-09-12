"""Consistent strict-FP32 coordinate bundles and alternating end-to-end readout benchmark."""
from pathlib import Path
import sys,time,json,copy
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import torch,numpy as np
import card040_common as V
from run_card040_arm import validate_arm
D=V.O/'card040-scoring'
class Views:
 def __init__(self):
  self.t=V.load_body('teacher','cuda');self.l=V.load_body('leaky','cuda');self.a={};self.norm={};self.scale={}
  for a in V.ARMS:
   validate_arm(a);p=torch.load(V.TRAIN/a/'adapter.pt',map_location='cpu',weights_only=False);m=V.Readout().cuda().eval().requires_grad_(False);m.load_state_dict(p['model']);self.a[a]=m;self.norm[a]=(p['mean'].cuda(),p['std'].cuda());self.scale[a]=p['target_scale']
 @torch.inference_mode()
 def one(self,x,name):
  if name=='teacher':return self.t(x)
  if name=='two_head':
   t=self.t(x);return (t.double()+V.ALPHA*(self.l(x).double()-t.double())).float()
  t,f,z=V.frozen_views(self.t,x)
  if name=='fixed':return f
  mean,std=self.norm[name];h=z.relu() if name=='post' else z;g=self.a[name]((h-mean)/std);return (f.double()+self.scale[name]*g.double()).float()
 @torch.inference_mode()
 def all(self,x):
  t,f,z=V.frozen_views(self.t,x);out={'teacher':t,'fixed':f,'two_head':(t.double()+V.ALPHA*(self.l(x).double()-t.double())).float()}
  for a in V.ARMS:
   mean,std=self.norm[a];h=z.relu() if a=='post' else z;out[a]=(f.double()+self.scale[a]*self.a[a]((h-mean)/std).double()).float()
  return out

def main():
 V.device_setup();V.validate_inputs();bank=V.validate_bank();D.mkdir(exist_ok=True);views=Views();names=['teacher','two_head','fixed','post','pre'];hashes={a:V.sha(V.TRAIN/a/'adapter.pt') for a in V.ARMS};tsha=V.state_sha(views.t.state_dict());lsha=V.state_sha(views.l.state_dict());start=time.monotonic()
 for cohort in ['reference','query']:
  x=np.load(V.DATA/(cohort+'-input.f32.npy'),mmap_mode='r');out={a:np.lib.format.open_memmap(D/(a+'-'+cohort+'.npy'),mode='w+',dtype='f4',shape=(len(x),2)) for a in names}
  for j in range(0,len(x),V.FORWARD_BATCH):
   xx=torch.from_numpy(np.array(x[j:j+V.FORWARD_BATCH])).cuda();y=views.all(xx)
   for a in names:out[a][j:j+len(xx)]=y[a].cpu().numpy()
  for y in out.values():assert np.isfinite(y).all();y.flush()
 # Benchmark exactly the predeclared 100K training inputs, one warm and five alternating passes.
 x=torch.from_numpy(np.array(np.load(V.DATA/'train-input.f32.npy',mmap_mode='r')[:100000])).cuda();timings={a:[] for a in ['teacher','two_head','post','pre']}
 for rep in range(6):
  order=list(timings) if rep%2==0 else list(reversed(timings))
  for a in order:
   torch.cuda.synchronize();ts=time.monotonic()
   for j in range(0,len(x),V.FORWARD_BATCH):views.one(x[j:j+V.FORWARD_BATCH],a).cpu().numpy()
   torch.cuda.synchronize()
   if rep:timings[a].append(time.monotonic()-ts)
 # Saved inference recipes must agree with separately executed paths, including model-specific outputs.
 checks={};qx=torch.from_numpy(np.array(np.load(V.DATA/'query-input.f32.npy',mmap_mode='r')[:256])).cuda()
 for a in names:
  y=views.one(qx,a).cpu().numpy();expected=np.load(D/(a+'-query.npy'),mmap_mode='r')[:256];checks[a+'_separate_path_matches_saved']=np.array_equal(y,expected)
 checks['frozen_body_unchanged']=tsha==V.state_sha(views.t.state_dict()) and lsha==V.state_sha(views.l.state_dict());checks['adapters_unchanged']=all(V.sha(V.TRAIN/a/'adapter.pt')==h for a,h in hashes.items());assert all(checks.values()),checks
 r={'PASS':True,'checks':checks,'precision':'float32','TF32':False,'batch_size':V.FORWARD_BATCH,'benchmark_input_n':100000,'benchmark_input_sha':V.sha(V.DATA/'train-input.f32.npy'),'warm_passes':1,'measured_passes':5,'cpu_output_copy':True,'timings_s':timings,'median_s':{a:float(np.median(v)) for a,v in timings.items()},'runtime_manifest_sha':V.runtime_check(),'input_manifest_sha':V.sha(V.DATA/'inputs.json'),'bank_sha':V.sha(V.DATA/'bank.json'),'models':hashes,'teacher_sha':V.BODY_SHA,'leaky_sha':V.LEAKY_SHA,'files':{p.name:V.sha(p) for p in D.glob('*.npy')},'wall_s':time.monotonic()-start,'resources':V.resources(),'loaded_modules':V.loaded_check()};V.atomic(V.O/'card040-evaluation.json',r);print(json.dumps(r))
if __name__=='__main__':main()
