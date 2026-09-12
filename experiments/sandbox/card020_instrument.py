import numpy as np
import torch
import faiss
from scipy.spatial.distance import cdist
from card016_model import interpolate_density
BUDGETS=[50,100,250,500,1000,2000]
torch.set_num_threads(2);faiss.omp_set_num_threads(2)
def project(model,data):
    chunks=[]
    with torch.inference_mode():
        for s in range(0,len(data),8192):
            x=np.array(data[s:s+8192],dtype='f4',copy=True)
            x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)
            chunks.append(model(torch.from_numpy(x)).numpy())
    xy=np.concatenate(chunks)
    assert xy.shape==(len(data),2) and np.isfinite(xy).all()
    return xy

def recall(ref,val,truth):
    index=faiss.IndexFlatL2(2); index.add(np.ascontiguousarray(ref,dtype='f4'))
    values={b:[] for b in BUDGETS}
    for s in range(0,len(val),256):
        _,ix=index.search(np.ascontiguousarray(val[s:s+256],dtype='f4'),2000)
        hit=(ix[:,:,None]==truth[s:s+256,None,:]).any(2)
        for b in BUDGETS: values[b].append(hit[:,:b].sum(1)/truth.shape[1])
    return {b:np.concatenate(x) for b,x in values.items()}

def continuity(xy,hi):
    n=len(xy); k=hi.shape[1]
    d=cdist(xy.astype('f8'),xy.astype('f8'),'sqeuclidean'); np.fill_diagonal(d,np.inf)
    order=np.argsort(d,axis=1,kind='stable'); rank=np.empty((n,n),dtype='i4')
    rank[np.arange(n)[:,None],order]=np.arange(1,n+1)
    return 1-2/(k*(2*n-3*k-1))*np.maximum(rank[np.arange(n)[:,None],hi]-k,0).sum(1)

def logdensity(native,manifest,grid):
    xy=(native-np.asarray(manifest['center'],dtype='f4'))/manifest['span']+.5
    outside=((xy<0)|(xy>1)).any(1)
    # Grid domain has a .1 margin around all fit targets, ten KDE bandwidths.
    # Outside it report the fixed density floor and an explicit domain flag.
    clipped=np.clip(xy,0,1)
    with torch.no_grad(): lp=interpolate_density(grid,torch.from_numpy(clipped.astype('f4'))).clamp_min(1e-12).log().numpy()
    lp[outside]=np.log(1e-12)
    return lp,outside,xy

def equal_strata_mean(values, strata):
    return float(np.mean([values[i].mean() for i in strata]))

def quality_guards(a,b,recs,cont,by,dec):
    gates={'continuity_loss_le_005':float(cont[b].mean()-cont[a].mean())<=.005}; detail={}
    for budget in [250,2000]:
        loss=recs[b][budget]-recs[a][budget]
        equal=float(np.mean([loss[ix].mean() for ix in by.values()]))
        source={g:float(loss[ix].mean()) for g,ix in by.items()}
        sparse=float(loss[dec>=7].mean()); sd={str(j+1):float(loss[dec==j].mean()) for j in [7,8,9]}
        detail[str(budget)]={'equal9_loss':equal,'source_loss':source,'sparse_mean_loss':sparse,'sparse_decile_loss':sd}
        gates.update({f'equal9_B{budget}':equal<=.005,f'every_source_B{budget}':max(source.values())<=.01,
                      f'sparse_mean_B{budget}':sparse<=.005,f'every_sparse_decile_B{budget}':max(sd.values())<=.01})
    return {k:bool(v) for k,v in gates.items()},detail
