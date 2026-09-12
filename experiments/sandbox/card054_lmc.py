"""Stateless sampled point-to-landmark correlation; no global RNG or optimizer state."""
import hashlib
import numpy as np
import torch

def negative_correlation(dx,dy):
 x=dx.float().reshape(-1);y=dy.float().reshape(-1);xc=x-x.mean();yc=y-y.mean();vx=xc.square().mean();vy=yc.square().mean();valid=(vx>1e-12)&(vy>1e-12);corr=(xc*yc).mean()/(vx.clamp_min(1e-12).sqrt()*vy.clamp_min(1e-12).sqrt());loss=torch.where(valid,-corr,y.sum()*0.);return loss,valid
class LandmarkLoss:
 def __init__(self,bank,weight,device='cuda',rows=256,seed=54056):
  b=np.load(bank);self.x=torch.from_numpy(b['pool_X'].astype('f4')).to(device);self.l=torch.from_numpy(b['landmark_X'].astype('f4')).to(device);self.d=torch.from_numpy(b['distances'].astype('f4')).to(device);self.weight=weight;self.rows=rows;self.seed=seed;self.calls=0;self.raw_sum=torch.zeros((),device=device);self.invalid=torch.zeros((),device=device,dtype=torch.int64);self.probes={};assert self.d.shape==(len(self.x),len(self.l))
 def indices(self,step):
  gen=torch.Generator(device=self.x.device);gen.manual_seed(self.seed+int(step));return torch.randint(len(self.x),(self.rows,),device=self.x.device,generator=gen)
 def __call__(self,model,step):
  assert not torch.backends.cuda.matmul.allow_tf32,'LMC requires strictFP32'
  ix=self.indices(step)
  with torch.autocast(device_type=self.x.device.type,enabled=False):
   y=model(self.x.index_select(0,ix));l=model(self.l);distance=(y[:,None,:]-l[None,:,:]).square().sum(2).add(1e-12).sqrt();raw,valid=negative_correlation(self.d.index_select(0,ix),distance);loss=self.weight*raw
  self.calls+=1;self.raw_sum+=raw.detach();self.invalid+=(~valid).detach().long()
  if step in [0,1,4,7,9,17,499,3499,4999,14999,29999]:self.probes[str(step)]={'indices_sha':hashlib.sha256(ix.cpu().numpy().tobytes()).hexdigest(),'raw_loss':float(raw.detach()),'weighted_loss':float(loss.detach())}
  return loss
 def stats(self):return {'attempted_calls':self.calls,'mean_raw_loss':float(self.raw_sum)/max(1,self.calls),'invalid_variance_calls':int(self.invalid),'weight':self.weight,'rows_per_attempt':self.rows,'landmarks':len(self.l),'seed':self.seed,'probes':self.probes,'scope':'Stats cover this process attempt; stateless samples keyed by successful-update index, repeated after AMP skips.'}
