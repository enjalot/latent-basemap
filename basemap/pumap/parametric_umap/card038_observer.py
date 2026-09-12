"""Default-off read-only exposure counters; no random draws or training mutations."""
import hashlib
import numpy as np

def digest(src,dst,mask):
    h=hashlib.sha256()
    for x in [src[mask],dst[mask]]:h.update(np.ascontiguousarray(x.detach().cpu().numpy()).tobytes())
    return h.hexdigest()

def attempt(stats,targets,loader):
    pos=targets>0.5;npos=int(pos.sum());nneg=int(targets.numel())-npos
    for name,value in [('attempted_positive',npos),('attempted_negative',nneg)]:stats['card038_'+name]=stats.get('card038_'+name,0)+value
    step=stats['attempted_batches']
    if step in [1,30000,60000,120000,180000]:
        src,dst=loader._last_all_src,loader._last_all_dst
        assert src.shape==dst.shape==targets.shape
        stats.setdefault('card038_attempted_probes',[]).append({'attempted_step':step,'npos':npos,'nneg':nneg,'positive_pair_sha256':digest(src,dst,pos),'negative_pair_sha256':digest(src,dst,~pos)})
    return npos,nneg

def succeeded(stats,counts):
    for name,value in zip(['successful_positive','successful_negative'],counts):stats['card038_'+name]=stats.get('card038_'+name,0)+value
