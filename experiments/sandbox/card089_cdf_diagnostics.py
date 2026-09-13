"""Persist CDF measurements BEFORE separate unchanged numerical rejections."""
import math
import torch
import card089_common as C

def finite_number(x):
 v=float(x);return v if math.isfinite(v) else None

def persist_and_check(cdf,w,path,metadata):
 interval=torch.diff(cdf,prepend=torch.zeros(1,dtype=cdf.dtype,device=cdf.device));mass=w.sum();negative=interval<0;lost=(w>0)&(interval==0);terminal=finite_number(cdf[-1]);residual=None if terminal is None else terminal-1
 checks={'dtype_float64':cdf.dtype==torch.float64,'finite':bool(torch.isfinite(cdf).all()),'nondecreasing':bool((interval>=0).all()),'terminal_1e12':residual is not None and abs(residual)<1e-12}
 out={**metadata,'status':'CDF_DIAGNOSTIC_BEFORE_ADMISSION','checks':checks,'cdf_terminal':terminal,'terminal_residual':residual,'min_interval':finite_number(interval.min()),'decreasing_interval_count':int(negative.sum()),'decreasing_interval_input_mass':finite_number(w[negative].sum()),'nonfinite_cdf_count':int((~torch.isfinite(cdf)).sum()),'positive_zero_width_count':int(lost.sum()),'positive_zero_width_mass':finite_number(w[lost].sum()),'positive_zero_width_probability':finite_number(w[lost].sum()/mass),'zero_weights':int((w==0).sum()),'tiny_below_2pow24_probability':finite_number(w[(w>0)&(w<2**-24)].sum()/mass),'weight_mass':finite_number(mass)}
 C.write(path,out)
 for key in checks:assert checks[key],'card089 CDF '+key+' failed'
 return out
