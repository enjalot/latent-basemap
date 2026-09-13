import numpy as np
import torch
KEYS=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen','mn_gen','mn_rng','dens_gen','dens_rng','hold_gen','hold_rng','deriv_gen','global_step','epoch','train_stats','config','step_checkpoint']
def same(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b
def compare(a,b):
 assert a.keys()==b.keys(),'checkpoint field set differs'
 assert set(KEYS)<=set(a),'required checkpoint state missing'
 for k in a:assert same(a[k],b[k]),'fullstate mismatch: '+k
 return sorted(a)
