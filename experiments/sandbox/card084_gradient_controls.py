"""Unscaled extra-gradient fault controls, shared by CPU and released device canary."""
import torch
from basemap.pumap.parametric_umap.bounded_attraction import add_attraction,checked_unscaled_gradients

class BrokenBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return x.clone()
    @staticmethod
    def backward(ctx,gradient):return torch.full_like(gradient,float('nan'))

def check_extra_gradient_faults(device):
    checks=[]
    for family in ('quadratic','pseudo_huber'):
        x=torch.ones((2,3),device=device,requires_grad=True)
        y=torch.zeros_like(x);mask=torch.ones(2,dtype=torch.bool,device=device);scale=torch.ones(2,device=device)
        def objective(a):return add_attraction(a.new_zeros(()),a,y,mask,scale,coefficient=1.,family=family,delta=.9139534189451107)
        extra=objective(x);grads=checked_unscaled_gradients(extra,[x])
        assert all(torch.isfinite(g).all() for g in grads)
        defect=objective(BrokenBackward.apply(x));assert torch.isfinite(defect)
        try:checked_unscaled_gradients(defect,[x])
        except FloatingPointError as error:
            assert str(error)=='Card084 nonfinite extra attraction gradient; STOP'
        else:raise AssertionError('genuine new-term backward defect accepted')
        # Scaled overflow is left to the matched training GradScaler/skip path.
        scaled=objective(x)*x.new_tensor(float('inf'))
        raw=torch.autograd.grad(scaled,[x])
        assert not torch.isfinite(raw[0]).all()
        checks.append(f'{device}/{family}: finite unscaled gradient, exact genuine backward-fault rejection, scaled overflow left to matched policy')
    return checks
