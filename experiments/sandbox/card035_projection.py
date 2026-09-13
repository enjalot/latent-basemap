"""Read-only FP32 model forwards plus final-ReLU support counts; no training/global side effects."""
import numpy as np
import torch

class Projector:
 def __init__(self,models,head_batch=256):
  assert isinstance(head_batch,int) and head_batch>0
  self.models=models;self.head_batch=head_batch;self.counts={};self.hooks=[]
  for name,m in models.items():
   assert type(m).__name__=='ResidualBottleneckMLP' and isinstance(m.up[0],torch.nn.Linear) and type(m.up[1]) is torch.nn.ReLU and isinstance(m.proj_out,torch.nn.Linear)
   assert m.proj_out.out_features==3 and m.up[0].out_features<=65535 and not m.training
   def capture(module,args,out,key=name):self.counts[key]=(out>0).sum(1)
   self.hooks.append(m.up[0].register_forward_hook(capture))
 @torch.inference_mode()
 def __call__(self,x):
  assert x.ndim==2 and x.dtype==torch.float32 and len(x)>0 and bool(torch.isfinite(x).all())
  result={}
  for name,m in self.models.items():
   ys=[];cs=[]
   for lo in range(0,len(x),self.head_batch):
    self.counts[name]=None;y=m(x[lo:lo+self.head_batch]);count=self.counts[name];assert count is not None and count.shape==(len(y),)
    assert y.shape==(len(count),3) and bool(torch.isfinite(y).all()) and bool((count<=m.up[0].out_features).all())
    zero=count==0
    if bool(zero.any()):assert torch.equal(y[zero],m.proj_out.bias[None].expand(int(zero.sum()),3)), 'all-inactive output must equal final bias'
    ys.append(y.cpu().numpy());cs.append(count.cpu().numpy().astype('u2'))
   result[name]=np.concatenate(ys).astype('f4',copy=False);result[name+'-positive-final']=np.concatenate(cs)
  return result
 def close(self):
  for hook in self.hooks:hook.remove()
  self.hooks=[];self.counts={}
