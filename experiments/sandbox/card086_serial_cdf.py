"""Explicit serial FP64 accumulation, no fast math, no parallel scan."""
import numpy as np
from numba import njit
@njit(fastmath=False,parallel=False)
def serial_prefix(weights):
 out=np.empty(len(weights),np.float64);total=np.float64(0.)
 for i in range(len(weights)):
  total=total+np.float64(weights[i]);out[i]=total
 return out

def build(weights):
 w=np.asarray(weights);assert w.ndim==1 and np.isfinite(w).all() and (w>=0).all(),'invalid CDF weights'
 prefix=serial_prefix(w);assert len(prefix) and prefix[-1]>0 and np.isfinite(prefix[-1]),'invalid CDF mass'
 terminal=float(prefix[-1]);reciprocal=np.float64(np.float64(1.)/np.float64(terminal));prefix*=reciprocal
 assert np.isfinite(prefix).all() and (np.diff(prefix)>=0).all() and prefix[-1]==1.,'serial CDF invariant'
 return prefix,terminal
