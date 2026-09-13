"""CPU-only mathematical contracts; no training, GPU or evaluation data."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import importlib.util
from pathlib import Path
import json
import torch
spec=importlib.util.spec_from_file_location('attraction',Path(__file__).resolve().parents[2]/'basemap/pumap/parametric_umap/bounded_attraction.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
f=m.add_attraction
torch.set_num_threads(2)
checks=[]
for dtype in (torch.float32, torch.float64):
 for family in ('quadratic','pseudo_huber'):
  for distance in (0.,1e-10,1.,1e6):
   x=torch.tensor([[distance,0.,0.],[2.,3.,4.]],dtype=dtype,requires_grad=True)
   y=torch.zeros_like(x,requires_grad=True)
   scale=torch.tensor([4.,9.],dtype=dtype,requires_grad=True)
   mask=torch.tensor([True,False]); delta=.9
   val=f(torch.zeros((),dtype=dtype),x,y,mask,scale,coefficient=1.,family=family,delta=delta)
   gx,gy,gs=torch.autograd.grad(val,(x,y,scale),allow_unused=True)
   expected=x.detach()[0]/4
   if family=='pseudo_huber': expected=expected/torch.sqrt(1+(distance/2/delta)**2*torch.ones((),dtype=dtype))
   torch.testing.assert_close(gx[0],expected,rtol=2e-5,atol=1e-12)
   assert torch.isfinite(val) and torch.isfinite(gx).all() and gs is None
   assert torch.equal(gx,-gy) and torch.equal(gx[1],torch.zeros(3,dtype=dtype))
   checks.append(f'{dtype}/{family}/{distance}: analytic gradient, finite, teacher detached, negatives excluded')
for family in ('quadratic','pseudo_huber'):
 x=torch.tensor([[.2,.3,.5],[0.,0.,0.]],dtype=torch.float64,requires_grad=True)
 y=torch.zeros_like(x,requires_grad=True);scale=torch.tensor([2.,3.],dtype=torch.float64)
 mask=torch.ones(2,dtype=torch.bool)
 fun=lambda a,b:f(a.new_zeros(()),a,b,mask,scale,coefficient=.3,family=family,delta=.9)
 assert torch.autograd.gradcheck(fun,(x,y))
 q=torch.tensor([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
 torch.testing.assert_close(fun(x,y),fun(x@q+7,y@q+7))
 g=torch.autograd.grad(fun(x,y),x)[0]
 xx=(x.detach()@q+7).requires_grad_();yy=y.detach()@q+7
 torch.testing.assert_close(torch.autograd.grad(fun(xx,yy),xx)[0],g@q)
 checks.append(f'{family}: gradcheck including zero, translation/rotation value and gradient')
base=torch.tensor(2.,requires_grad=True);state=torch.random.get_rng_state().clone()
assert f(base,None,None,None,None,coefficient=0.,family='invalid') is base
assert torch.equal(state,torch.random.get_rng_state())
checks.append('coefficient zero: same tensor object, no data access, no RNG change')
# Resume-neutral helper: same result after unrelated use; no mutable module state.
x=torch.ones(2,3);mask=torch.ones(2,dtype=torch.bool);scale=torch.ones(2)
a=f(base,x,x*0,mask,scale,coefficient=.2,delta=.9)
b=f(base,x,x*0,mask,scale,coefficient=.2,delta=.9)
assert torch.equal(a,b)
checks.append('stateless repeat identity (not a production resume canary)')
print(json.dumps({'PASS':True,'checks':checks,'device':'cpu','production_resume_tested':False},indent=2))
