"""Root-gated Card088 same-support extra45 search. Never run in CPU preparation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import time,math,mmap,resource,hashlib
import numpy as np,torch
import card088_graph as G
import card088_common as C
import card088_budget as Budget
Q=1024;BLOCK=65536

def independent_extra_search(db,panel,original,check):
 # Independent reference: NumPy lexicographic merge over complete blocks.
 # Device only evaluatesFP64 normalized dot products; no production topk helper.
 query=torch.nn.functional.normalize(db[torch.as_tensor(panel,device=db.device)].double(),dim=1)
 scores=np.empty((len(panel),0),'f8');ids=np.empty((len(panel),0),'i8')
 for lo in range(0,len(db),8192):
  hi=min(lo+8192,len(db));ref=torch.nn.functional.normalize(db[lo:hi].double(),dim=1);sim=(query@ref.T).cpu().numpy();block_ids=np.arange(lo,hi)
  next_scores=[];next_ids=[]
  for j,row in enumerate(panel):
   eligible=~np.isin(block_ids,np.r_[row,original[j]])
   sc=np.r_[scores[j],sim[j,eligible]];ii=np.r_[ids[j],block_ids[eligible]]
   take=np.lexsort((ii,-sc))[:46];next_scores.append(sc[take]);next_ids.append(ii[take])
  scores=np.stack(next_scores);ids=np.stack(next_ids);check()
 return torch.as_tensor(ids,device=db.device),torch.as_tensor(scores,device=db.device)

def main():
 release=C.require_release();start=time.monotonic();C.input_check(require_graph=False);stage_cap=float(os.environ['CARD088_STAGE_LIMIT_S'])
 assert torch.cuda.is_available(),'device search requires CUDA'
 assert release.get('graph_stage_cap_s')==1500,'root graph cap binding required'
 reserve=float(release['post_graph_full_dose_reserve_s']);assert reserve>=3000,'root full-dose reserve too small'
 D=G.D;D.mkdir(parents=True,exist_ok=True);runtime=C.source_check()
 inputs={str(G.INPUT/'manifest.json'):G.sha(G.INPUT/'manifest.json')}
 im=G.read(G.INPUT/'manifest.json');assert im['PASS']
 for p,h in im['inputs'].items():assert G.sha(p)==h;inputs[p]=h
 oldpath=G.OLD/'knn_indices.npy';feature=G.INPUT/'substrate.f16.npy'
 assert G.sha(feature)==im['files']['substrate.f16.npy'];inputs[str(feature)]=G.sha(feature)
 if (D/'manifest.json').exists():
  G.validate_bundle();G.write(C.O/'card088-graph-ready.json',{'PASS':True,'revalidated_existing':True,'runtime_sha':runtime,'data_manifest_sha':G.sha(D/'manifest.json')});return
 old=np.load(oldpath,mmap_mode='r');assert old.shape==(G.N,15) and old.dtype==np.int32
 mm=np.load(feature,mmap_mode='r');assert mm.shape==(G.N,G.DIM) and mm.dtype==np.float16
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 db=torch.empty(mm.shape,device='cuda',dtype=torch.float16)
 def limits():
  free,total=torch.cuda.mem_get_info();used=(total-free)/2**30;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
  assert used<30 and rss<49152,'graph memory STOP'
  assert time.monotonic()-start<stage_cap-5,'graph wall STOP'
  assert time.time()<Budget.END-1,'graph deadline STOP'
  return used,rss
 for lo in range(0,G.N,65536):
  hi=min(lo+65536,G.N);buf=np.array(mm[lo:hi]);assert np.isfinite(buf).all() and (np.linalg.norm(buf.astype('f4'),axis=1)>0).all(),'invalid source feature';db[lo:hi]=torch.from_numpy(buf).to('cuda');mm._mmap.madvise(mmap.MADV_DONTNEED);limits()
 identity={'runtime_sha':runtime,'builder_sha':G.sha(__file__),'inputs':inputs,'q':Q,'block':BLOCK,'search_precision':'normalize_FP32_stored_FP16_TF32_off','exclude':'self_and_exact_original15','extra':45,'tie_order':'descending_score_ascending_training_local_ID'}
 progress=D/'graph-progress.json';hist=[];cursor=0
 if progress.exists():
  state=G.read(progress);assert state['identity']==identity,'graph resume identity mismatch';hist=state['history'];cursor=state['cursor']
 timings=[]
 with torch.inference_mode():
  tinyrows=np.arange(4);tinyold=np.array([[(i+j)%4096 for j in range(1,16)] for i in tinyrows]);tiny,tinys,_=G.search(db[:4096],tinyrows,tinyold,block=1024,k=46)
  small=db[:4096].double().cpu().numpy();small/=np.linalg.norm(small,axis=1,keepdims=True)
  for j,row in enumerate(tinyrows):
   eligible=np.setdiff1d(np.arange(4096),np.r_[row,tinyold[j]]);ss=small[eligible]@small[row];cut=np.sort(ss)[-46];got=tiny[j].cpu().numpy();actual=small[got]@small[row];assert (actual>=cut-G.TOL).all() and np.max(np.abs(actual-tinys[j].cpu().numpy()))<=G.TOL,'device tiny independent precision canary'
  for lo in [0,G.N//2,G.N-Q]:
   rows=np.arange(lo,lo+Q);t=time.monotonic();G.search(db,rows,np.array(old[rows]),block=BLOCK,k=46,check=limits);torch.cuda.synchronize();timings.append(time.monotonic()-t)
  estimate=max(timings)*math.ceil((G.N-cursor)/Q)*1.25+180
  # Ancestor chain reserved this entire stage conservatively; use its before-stage
  # availability, then subtract actual elapsed, not its unspent reservation twice.
  stage_available=float(os.environ['CARD088_STAGE_AVAILABLE_S'])
  remain=min(stage_cap-(time.monotonic()-start),stage_available-(time.monotonic()-start)-reserve,Budget.END-time.time()-reserve)
  pf={'PASS':bool(estimate<=remain),'runtime_sha':runtime,'search_identity':identity,'block_seconds':timings,'full_remaining_estimate_s':estimate,'available_graph_s':remain,'post_graph_reserved_s':reserve,'global_vram_gib':limits()[0]}
  G.write(C.O/'card088-graph-preflight.json',pf)
  assert pf['PASS'],'graph full-dose joint reserve admission STOP'
  mode='r+' if progress.exists() else 'w+'
  ii=np.lib.format.open_memmap(D/'extra45.i32.npy',mode=mode,dtype='i4',shape=(G.N,45));sc=np.lib.format.open_memmap(D/'extra46-scores.f32.npy',mode=mode,dtype='f4',shape=(G.N,46))
  assert len(hist)==math.ceil(cursor/Q) and (cursor==G.N or cursor%Q==0),'graph resume cursor'
  for j,h in enumerate(hist):
   lo,hi=h['lo'],h['hi'];assert lo==j*Q and hi==min(lo+Q,G.N),'graph history order'
   assert hashlib.sha256(ii[lo:hi].tobytes()).hexdigest()==h['ids_sha'] and hashlib.sha256(sc[lo:hi].tobytes()).hexdigest()==h['score_sha'],'graph completed chunk mutation'
  for lo in range(cursor,G.N,Q):
   hi=min(lo+Q,G.N);rows=np.arange(lo,hi);orig=np.array(old[lo:hi]);assert not (orig==rows[:,None]).any() and (np.diff(np.sort(orig,axis=1),axis=1)>0).all(),'original15 invalid'
   idx,scores,diag=G.search(db,rows,orig,block=BLOCK,k=46,check=limits);a=idx[:,:45].cpu().numpy().astype('i4');b=scores.cpu().numpy();ii[lo:hi]=a;sc[lo:hi]=b;ii.flush();sc.flush();hist.append({'lo':lo,'hi':hi,'ids_sha':hashlib.sha256(a.tobytes()).hexdigest(),'score_sha':hashlib.sha256(b.tobytes()).hexdigest(),**diag});G.write(progress,{'identity':identity,'cursor':hi,'history':hist})
   if len(hist)%64==0:print('088 graph',hi,'/',G.N,flush=True)
  # Fixed panel chosen by local IDs, before reading quality. Independent FP64
  # normalization/dot arithmetic over every support row; original15 still excluded.
  panel=np.random.default_rng(88088).choice(G.N,64,replace=False);o=np.array(old[panel]);idx64,score64=independent_extra_search(db,panel,o,limits)
  q=torch.nn.functional.normalize(db[torch.as_tensor(panel,device='cuda')].double(),dim=1)
  picked=torch.as_tensor(np.array(ii[panel]),device='cuda',dtype=torch.long);neigh=torch.nn.functional.normalize(db[picked].double(),dim=2);direct=(q[:,None]*neigh).sum(2)
  bad=direct<score64[:,44,None]-G.TOL;gap=(direct-torch.as_tensor(np.array(sc[panel,:45]),device='cuda').double()).abs()
  assert not bool(bad.any()) and float(gap.max())<=G.TOL,'graph precision material miss'
  recall=np.array([len(set(a)&set(b))/45 for a,b in zip(ii[panel],idx64[:,:45].cpu().numpy())])
  # Separately characterize original15 against the NEW arithmetic, not replace it.
  fullbest=torch.full((len(panel),16),-torch.inf,device='cuda',dtype=torch.float64);fullids=torch.full((len(panel),16),-1,device='cuda',dtype=torch.long)
  for lo in range(0,G.N,8192):
   hi=min(lo+8192,G.N);ref=torch.nn.functional.normalize(db[lo:hi].double(),dim=1);sim=q@ref.T;mask=(panel>=lo)&(panel<hi);sim[np.flatnonzero(mask),panel[mask]-lo]=-torch.inf;s,i,_=G.block_top(sim,lo,16);fullbest,fullids=G.deterministic_top(torch.cat([fullbest,s],1),torch.cat([fullids,i],1),16);limits()
  old_recall=np.array([len(set(a)&set(b))/15 for a,b in zip(o,fullids[:,:15].cpu().numpy())]);oldunit=torch.nn.functional.normalize(db[torch.as_tensor(o,device='cuda')].double(),dim=2);oldscores=(q[:,None]*oldunit).sum(2);old_bad=oldscores<fullbest[:,14,None]-G.TOL
  np.savez(D/'search-audit.npz',panel_local=panel,original15=o,extra45=np.array(ii[panel]),FP64_extra46=idx64.cpu().numpy(),FP64_scores=score64.cpu().numpy(),extra_recall45=recall,extra_material_miss=bad.cpu().numpy(),stored_score_gap=gap.cpu().numpy(),original15_recall=old_recall,original15_material_disagreement=old_bad.cpu().numpy(),unrestricted_FP64_top16=fullids.cpu().numpy())
  # Stream common endpoints to disk; weights integer-exact. Original15 verbatim.
  targets=np.lib.format.open_memmap(D/'targets60.i32.npy',mode='w+',dtype='i4',shape=(G.N,60))
  for lo in range(0,G.N,32768):
   hi=min(lo+32768,G.N);a=np.array(old[lo:hi]);b=np.array(ii[lo:hi]);v=np.concatenate([a,b],1);assert (v>=0).all() and (v<G.N).all() and not (v==np.arange(lo,hi)[:,None]).any() and (np.diff(np.sort(v,axis=1),axis=1)>0).all(),'common support invalid';targets[lo:hi]=v
  targets.flush();src=np.repeat(np.arange(G.N,dtype='i4'),60)
  for arm in C.ARMS:
   w=G.weights(arm,G.N);assert np.all(w.sum(1,dtype='f8')==90);np.savez(D/f'{arm}-edges.npz',sources=src,targets=targets.reshape(-1),weights=w.reshape(-1),n_nodes=np.int64(G.N));del w;limits()
  files={p.name:G.sha(p) for p in D.iterdir() if p.suffix in ['.npy','.npz']}
  m={'PASS':True,'complete':True,'n':G.N,'k_extra':45,'graph_builder_sha':G.sha(__file__),'runtime_sha':runtime,'inputs':inputs,'files':files,'search_identity':identity,'audit':{'n':len(panel),'extra_recall45_mean':float(recall.mean()),'extra_recall45_min':float(recall.min()),'material_misses':int(bad.sum()),'max_score_gap':float(gap.max()),'original15_recall_mean':float(old_recall.mean()),'original15_material_disagreement':int(old_bad.sum()),'new_extra45_FP32_boundary_tie_rows':int(np.sum(sc[:,44]==sc[:,45]))},'weights':{'original15':[6,0],'mixture':[3,1],'row_total':90,'control_zero_weights':G.N*45},'original15_preservation':'verbatim_ordered_training_local_IDs','wall_s':time.monotonic()-start,'limitations':'New closest45 outside preservedoriginal15 under declaredFP32 search; FP64 sampled audit,tolerance4e-6. Not certified globalranks16-60.'}
  G.write(D/'manifest.json',m);G.write(C.O/'card088-graph-ready.json',{'PASS':True,'runtime_sha':runtime,'data_manifest_sha':G.sha(D/'manifest.json')})
if __name__=='__main__':main()
