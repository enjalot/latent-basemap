"""Prove conservative CPU bias prefilter, including active exact-bias outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import numpy as np,torch
import card062_common as C
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.up=torch.nn.Sequential(torch.nn.Linear(1536,64),torch.nn.ReLU());self.proj_out=torch.nn.Linear(64,3)
  with torch.no_grad():
   self.up[0].weight.zero_();self.up[0].bias.fill_(-1);self.up[0].bias[:2]=0;self.up[0].weight[0,0]=1;self.up[0].weight[1,1]=1;self.proj_out.weight.zero_();self.proj_out.weight[0,0]=1;self.proj_out.weight[0,1]=-.5;self.proj_out.bias.copy_(torch.tensor([.2,-.3,.4]))
  self._card062_bias=self.proj_out.bias.detach().numpy().copy()
 def forward(self,x):return self.proj_out(self.up(x))
m=Toy().eval().requires_grad_(False);orig=C.kernel;calls=[]
def count(*args,**kwargs):calls.append(len(args[0]));return orig(*args,**kwargs)
C.kernel=count;checks=[]
for label,xy,ncalls in [('nonbias',(1,0),0),('active_cancellation',(1,2),2),('inactive',(-1,-1),2)]:
 x=np.zeros((257,1536),'f4');x[:,:2]=xy;calls.clear();a=C.project(m,x,.001)['xy'];actual=list(calls);b=C.project(m,x,.001,collect=True);assert np.array_equal(a,b['xy']) and len(actual)==ncalls;checks.append(label+' prefilter conservatism')
 if label=='active_cancellation':assert not b['inactive'].any() and np.array_equal(a,b['teacher']);checks.append('active exact-bias collision is retained')
 if label=='inactive':assert b['inactive'].all() and not np.array_equal(a,b['teacher']);checks.append('inactive nonzero-direction rows never skipped')
C.write(C.O/'card062-prefilter-cpu-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks)});print('PASS',len(checks))
