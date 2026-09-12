"""Exact FP32 normalized-stored-input graph with direct FP64 radii; admitted isolated GPU stage."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,time,hashlib,mmap,resource,datetime as dt
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];O=ROOT.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card033-scale4m');N=4000000;Q=4096;B=131072;K=15

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def write(p,r):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 start=time.monotonic();release=json.loads((O/'card033-launch-release.json').read_text());assert release['PASS'];draw=json.loads((D/'draw-manifest.json').read_text());assert draw['PASS'] and draw['checks']['memory_below32GiB'];assert sha(D/'draw-manifest.json')==release['draw_manifest_sha'];runtime=json.loads((ROOT/'card033-runtime-sha.json').read_text());assert sha(ROOT/'card033-runtime-sha.json')==release['runtime_manifest_sha']
 for n,h in runtime.items():assert sha(ROOT/n)==h,n
 for n,h in draw['files'].items():assert sha(D/n)==h,n
 if (D/'manifest.json').exists():
  m=json.loads((D/'manifest.json').read_text());assert m['PASS'] and m['graph_builder_sha']==sha(__file__) and m['draw_manifest_sha']==sha(D/'draw-manifest.json');assert all(sha(D/n)==h for n,h in m['files'].items());print('existing graph verified',flush=True);return
 assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();mm=np.load(D/'substrate.f16.npy',mmap_mode='r');assert mm.shape==(N,1536)
 # Load only FP16 resident storage, in chunks; release source mapping residency.
 db=torch.empty((N,1536),device='cuda',dtype=torch.float16)
 for lo in range(0,N,65536):db[lo:lo+65536]=torch.from_numpy(np.array(mm[lo:lo+65536],copy=True)).cuda();mm._mmap.madvise(mmap.MADV_DONTNEED)
 def check_mem():
  free,total=torch.cuda.mem_get_info();used=(total-free)/2**30;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert used<30 and rss<49152,(used,rss);return used,rss
 def search(rows,ref_limit=N,k=15):
  query=torch.nn.functional.normalize(db[rows].float(),dim=1);assert torch.isfinite(query).all() and (torch.linalg.vector_norm(query,dim=1)>0).all();best=torch.full((len(rows),k),-torch.inf,device='cuda');bi=torch.full((len(rows),k),-1,device='cuda',dtype=torch.long)
  for lo in range(0,ref_limit,B):
   hi=min(lo+B,ref_limit);r=torch.nn.functional.normalize(db[lo:hi].float(),dim=1);assert torch.isfinite(r).all() and (torch.linalg.vector_norm(r,dim=1)>0).all();sim=query@r.T;mask=(rows>=lo)&(rows<hi);sim[torch.nonzero(mask).flatten(),rows[mask]-lo]=-torch.inf;s,i=torch.topk(sim,k,dim=1);best,order=torch.topk(torch.cat([best,s],1),k,dim=1);bi=torch.gather(torch.cat([bi,i+lo],1),1,order);check_mem()
  assert (bi>=0).all() and not (bi==rows[:,None]).any();assert (torch.diff(torch.sort(bi,dim=1).values,dim=1)>0).all(), 'duplicate neighbors';neighbors=torch.nn.functional.normalize(db[bi].float(),dim=2);dist=torch.square(neighbors.double()-query[:,None,:].double()).sum(2);assert torch.isfinite(dist).all() and (dist>=0).all();return bi.cpu().numpy().astype('i4'),dist.cpu().numpy(),best.cpu().numpy()
 with torch.inference_mode():
  # Tiny real-data CPU/exhaustive canary before expensive full-data timing.
  rows=torch.arange(32,device='cuda');got,gd,gs=search(rows,4096);x=torch.nn.functional.normalize(db[:4096].float(),dim=1).cpu().numpy().astype('f8');exact=x[:32]@x.T;exact[np.arange(32),np.arange(32)]=-np.inf;truth=np.argsort(-exact,axis=1,kind='stable')[:,:15];threshold=np.take_along_axis(exact,truth,axis=1)[:,-1];actual=np.take_along_axis(exact,got,axis=1);assert np.all(actual>=threshold[:,None]-4e-6);check_mem()
  timings=[]
  for lo in [0,2000000,N-Q]:
   t=time.monotonic();i,dd,ss=search(torch.arange(lo,lo+Q,device='cuda'));torch.cuda.synchronize();timings.append(time.monotonic()-t)
  cursor=0;hist=[];identity=sha(D/'draw-manifest.json')+':'+sha(__file__)+':'+sha(ROOT/'card033-runtime-sha.json');progress=D/'graph-progress.json'
  if progress.exists():
   old=json.loads(progress.read_text());assert old['identity']==identity;cursor=old['cursor'];hist=old['history']
  estimate=max(timings)*np.ceil((N-cursor)/Q)*1.2+120;ledger=json.loads((O/'card033-ledger.json').read_text());prior=sum(e['wall_s'] for e in ledger['entries'] if e.get('tag')=='graph');remaining=min(1800-prior-(time.monotonic()-start),32400-ledger['batch_spent_s']-(time.monotonic()-start),dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp()-time.time());ok=estimate<=remaining
  write(O/'card033-graph-preflight.json',{'PASS':bool(ok),'full4096query_block_s':timings,'estimated_remaining_s':float(estimate),'remaining_s':remaining,'global_vram_gib':check_mem()[0],'TF32':False})
  if not ok:raise SystemExit(3)
  ii=np.lib.format.open_memmap(D/'knn-indices.i32.npy',mode='r+' if progress.exists() else 'w+',dtype='i4',shape=(N,15));dd=np.lib.format.open_memmap(D/'knn-sqdist.f64.npy',mode='r+' if progress.exists() else 'w+',dtype='f8',shape=(N,15))
  for h in hist:
   assert hashlib.sha256(ii[h['lo']:h['hi']].tobytes()).hexdigest()==h['indices_sha'];assert hashlib.sha256(dd[h['lo']:h['hi']].tobytes()).hexdigest()==h['dist_sha']
  for lo in range(cursor,N,Q):
   hi=min(lo+Q,N);idx,dist,sim=search(torch.arange(lo,hi,device='cuda'));ii[lo:hi]=idx;dd[lo:hi]=dist;ii.flush();dd.flush();hist.append({'lo':lo,'hi':hi,'indices_sha':hashlib.sha256(idx.tobytes()).hexdigest(),'dist_sha':hashlib.sha256(dist.tobytes()).hexdigest()});write(progress,{'cursor':hi,'identity':identity,'history':hist});check_mem()
   if len(hist)%32==0:print('graph',hi,'/',N,'elapsed',time.monotonic()-start,flush=True)
  # Independent full-reference FP64 dot products on a preselected128-row panel, with ID-self excluded.
  source=np.load(D/'draw_source.npy');panel=np.load(D/'graph-audit-panel.npy');assert sha(D/'graph-audit-panel.npy')==release['graph_audit_panel_sha'] and len(panel)==128 and len(np.unique(panel))==128;query=torch.nn.functional.normalize(db[torch.from_numpy(panel).cuda()].float(),dim=1).double();best=torch.full((128,16),-torch.inf,device='cuda',dtype=torch.float64);bid=torch.full((128,16),-1,device='cuda',dtype=torch.long)
  for lo in range(0,N,16384):
   hi=min(lo+16384,N);r=torch.nn.functional.normalize(db[lo:hi].float(),dim=1).double();sim=query@r.T;here=(panel>=lo)&(panel<hi);sim[np.flatnonzero(here),panel[here]-lo]=-torch.inf;s,i=torch.topk(sim,16,dim=1);best,order=torch.topk(torch.cat([best,s],1),16,dim=1);bid=torch.gather(torch.cat([bid,i+lo],1),1,order)
  ni=np.asarray(ii[panel]);neigh=torch.nn.functional.normalize(db[torch.from_numpy(ni.astype('i8')).cuda()].float(),dim=2).double();score=torch.sum(query[:,None,:]*neigh,2).cpu().numpy();cut=best[:,14].cpu().numpy();bad=score<cut[:,None]-4e-6;relative=np.array([len(set(a)&set(b))/15 for a,b in zip(ni,bid[:,:15].cpu().numpy())]);radius_distance=torch.square(neigh-query[:,None,:]).sum(2).cpu().numpy();assert not bad.any() and np.max(np.abs(radius_distance-dd[panel]))<1e-10
  np.savez(D/'neighbor-audit.npz',panel_local=panel,panel_ids=np.load(D/'draw_ids.npy')[panel],stored_neighbors=ni,FP64_neighbors=bid.cpu().numpy(),FP64_scores=best.cpu().numpy(),recall15=relative,material_wrong=bad)
  raw=np.sqrt(np.asarray(dd).mean(1));p95=float(np.percentile(raw,95));assert np.isfinite(p95) and p95>0;unclipped=raw/p95;floor=float(np.mean(unclipped<1e-6));assert floor<=.001;half=np.sqrt(np.maximum(unclipped,1e-6)).astype('f4');np.save(D/'r_raw.npy',raw);np.save(D/'r_actual.npy',half);np.savez(D/'edges-fixed15.npz',sources=np.repeat(np.arange(N,dtype='i4'),15),targets=np.asarray(ii).reshape(-1),weights=np.ones(N*15,'f4'),n_nodes=np.int64(N));check_mem()
  files={n:sha(D/n) for n in ['substrate.f16.npy','draw_ids.npy','draw_source.npy','knn-indices.i32.npy','knn-sqdist.f64.npy','edges-fixed15.npz','r_raw.npy','r_actual.npy','neighbor-audit.npz']};result={'PASS':True,'complete':True,'n':N,'dimensions':1536,'draw_manifest_sha':sha(D/'draw-manifest.json'),'graph_builder_sha':sha(__file__),'runtime_manifest_sha':sha(ROOT/'card033-runtime-sha.json'),'TF32':False,'files':files,'radius':{'definition':'sqrt(max(sqrt(mean(d_HD_squared))/train_p95,1e-6)) for half-strength HookC','train_p95':p95,'floor_fraction':floor},'neighbor_audit':{'n':128,'exact_FP64_recall15_mean':float(relative.mean()),'material_misses_above4e-6':int(bad.sum())},'graph_wall_s':time.monotonic()-start,'global_vram_gib':check_mem()[0],'max_rss_mib':check_mem()[1],'source_counts':draw['source_counts']};write(D/'manifest.json',result);write(O/'card033-data-manifest.json',result);print(json.dumps({k:result[k] for k in ['PASS','graph_wall_s','radius','neighbor_audit','global_vram_gib']},indent=2))
if __name__=='__main__':main()
