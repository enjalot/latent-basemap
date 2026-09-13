"""Opt-in ordered q(i) q(j)/(1-q(i)) negatives, independent of embedding geometry."""
import numpy as np
import torch

class ConditionalCDF:
    def __init__(self, q, device):
        q=np.asarray(q,dtype=np.float64)
        if q.ndim!=1 or len(q)<2 or not np.isfinite(q).all() or not (q>0).all() or not (q<1).all() or abs(q.sum()-1)>2e-12:
            raise ValueError('invalid degree proposal')
        c=np.cumsum(q);c[-1]=1.
        if not (np.diff(np.r_[0.,c])>0).all():raise ValueError('non-increasing proposal CDF')
        self.cdf=torch.tensor(c,device=device,dtype=torch.float64)
        self.prob=torch.diff(torch.cat((self.cdf.new_zeros(1),self.cdf)))
        self.left=self.cdf-self.prob
        self.count=0
    def from_uniforms(self,u,v):
        # right=True maps interval boundaries to their following half-open cell.
        i=torch.searchsorted(self.cdf,u.contiguous(),right=True)
        pi=self.prob.index_select(0,i);left=self.left.index_select(0,i)
        removed=v*(1-pi)
        restored=removed+torch.where(removed>=left,pi,torch.zeros_like(pi))
        restored=torch.minimum(restored,torch.nextafter(restored.new_tensor(1.),restored.new_tensor(0.)))
        j=torch.searchsorted(self.cdf,restored.contiguous(),right=True)
        if torch.any(i==j) or torch.any(j>=len(self.cdf)):raise RuntimeError('conditional CDF produced invalid/self pair')
        return i,j
    def draw(self,n,generator):
        u=torch.rand(n,dtype=torch.float64,device=self.cdf.device,generator=generator)
        v=torch.rand(n,dtype=torch.float64,device=self.cdf.device,generator=generator)
        i,j=self.from_uniforms(u,v);self.count+=n
        return i,j
