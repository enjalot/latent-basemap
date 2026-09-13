"""CPU-only weights: same ordered original15, no feature or query access."""
from pathlib import Path
import numpy as np
ARMS=['reciprocal','rank_count_control']
def weights(mask,arm):
 assert arm in ARMS,'unknown weight arm'
 assert mask.ndim==2 and mask.shape[1]==15 and mask.dtype==bool,'invalid mutual mask'
 m=mask.sum(1);favored=mask if arm=='reciprocal' else np.arange(15)[None,:]<m[:,None]
 raw=1+3*favored.astype('f8');w=(raw*(15/(15+3*m))[:,None]).astype('f4')
 assert np.isfinite(w).all() and (w>0).all(),'invalid positive weights'
 return w,favored
