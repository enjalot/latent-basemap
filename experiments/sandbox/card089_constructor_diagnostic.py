"""Actual sampler constructor only; no feature access, draws, iteration or fit."""
from pathlib import Path
import hashlib,time,gc,sys,inspect
import numpy as np,torch
import card089_common as C
sys.path.insert(0,str(C.R))
from card089_cdf_diagnostics import persist_and_check
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceArrayDataset,DeviceEdgeSampler

def values_sha(x):
 return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def inspect_arm(arm,device,output):
 sampler_source=Path(inspect.getfile(DeviceEdgeSampler)).resolve();assert sampler_source.is_relative_to(C.R),'diagnostic sampler import escaped'
 start=time.monotonic();graph=C.GD/f'{arm}-edges.npz';graph_sha=C.sha(graph)
 with np.load(graph) as z:
  sources=z['sources'];targets=z['targets'];weights=z['weights'];nodes=int(z['n_nodes'])
 assert nodes==C.N and len(weights)==nodes*15
 # A zero one-column stand-in is used only to satisfy constructor dataset shape.
 # No iteration/gather is allowed, so this cannot expose encoder features.
 dataset=DeviceArrayDataset(np.zeros((nodes,1),dtype='f4'),device=device)
 results=[];previous=None;errors=[]
 for repeat in range(2):
  sampler=DeviceEdgeSampler(dataset,sources,targets,weights,nodes,pos_ratio=.1,batch_size=16384,random_state=42,positive_target_mode='binary',weighted_edge_sampling=True,uniform_with_replacement=False,device=device)
  if device=='cuda':torch.cuda.synchronize()
  cdf=sampler.sample_cdf;w=torch.as_tensor(weights,dtype=torch.float64,device=device)
  metadata={'arm':arm,'device':device,'repeat':repeat,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.GD/'manifest.json'),'graph_sha':graph_sha,'weight_values_sha':values_sha(weights),'sampler_source':str(sampler_source),'sampler_source_sha':C.sha(sampler_source),'constructor_only':True,'draws':0,'optimizer_steps':0}
  proof=Path(output)/f'{arm}-{repeat}.json'
  try:report=persist_and_check(cdf,w,proof,metadata)
  except AssertionError as error:report=C.read(proof);errors.append(str(error))
  host=cdf.detach().cpu().numpy().copy()
  report.update(cdf_values_sha=values_sha(host),repeat_bit_identical=None if previous is None else bool(np.array_equal(previous,host)),max_repeat_difference=None if previous is None else float(np.max(np.abs(previous-host))))
  C.write(proof,report);results.append({'path':str(proof),'sha':C.sha(proof),'checks':report['checks'],'cdf_values_sha':report['cdf_values_sha'],'repeat_bit_identical':report['repeat_bit_identical'],'max_repeat_difference':report['max_repeat_difference']});previous=host
  assert sampler.perm is None and sampler.pos_idx==0,'constructor diagnostic advanced sampler'
  del sampler,cdf,w;gc.collect()
  if device=='cuda':torch.cuda.empty_cache()
 positive=weights[weights>0];stats={'count':len(weights),'positive_count':len(positive),'zero_count':int((weights==0).sum()),'min_positive':float(positive.min()) if len(positive) else None,'max':float(weights.max()),'dynamic_range':float(positive.max())/float(positive.min()) if len(positive) else None,'sum_FP64':float(weights.sum(dtype='f8'))}
 return {'PASS':not errors,'arm':arm,'device':device,'weights':stats,'repeats':results,'errors':errors,'wall_s':time.monotonic()-start,'graph_sha':graph_sha}
