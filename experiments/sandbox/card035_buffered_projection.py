"""Same FP32 256-row forwards/counts, bulk output transfers after each head."""
import numpy as np
import torch
from card035_projection import Projector
class BufferedProjector(Projector):
 @torch.inference_mode()
 def __call__(self,x):
  assert x.ndim==2 and x.dtype==torch.float32 and len(x)>0 and bool(torch.isfinite(x).all())
  result={}
  for name,m in self.models.items():
   ys=[];cs=[]
   for lo in range(0,len(x),self.head_batch):
    self.counts[name]=None;y=m(x[lo:lo+self.head_batch]);count=self.counts[name];assert count is not None and count.shape==(len(y),)
    assert y.shape==(len(count),3);ys.append(y);cs.append(count)
   y=torch.cat(ys);count=torch.cat(cs);assert bool(torch.isfinite(y).all()) and bool((count<=m.up[0].out_features).all())
   zero=count==0
   if bool(zero.any()):assert torch.equal(y[zero],m.proj_out.bias[None].expand(int(zero.sum()),3))
   result[name]=y.cpu().numpy().astype('f4',copy=False);result[name+'-positive-final']=count.cpu().numpy().astype('u2')
  return result
