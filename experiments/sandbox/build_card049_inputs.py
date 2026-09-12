"""CPU-only fixed CLIP inputs. No graph, encoder truth or training is produced here."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib,time,datetime as dt,mmap,resource
import numpy as np
import torch
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from basemap.pumap.parametric_umap.core import ParametricUMAP
O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card049-clip-scale');POOL=Path('/data2/monet/pool-20m');SEAL=Path('/data2/monet/eval-common-v2');PARENT=Path('/data/latent-basemap/substrates/card010-adaptive');RAW=POOL/'clip512.f32.npy';CHAMP=R.parent/'dino-arrival-t0/champion-bs16k/model.pt'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def fingerprint(p):
 s=Path(p).stat();return {k:getattr(s,k) for k in ['st_size','st_mtime_ns','st_ino','st_dev']}
def write(p,r):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 start=time.monotonic();D.mkdir(exist_ok=True)
 assert not (D/'inputs-manifest.json').exists(),'existing fixed inputs; inspect before rebuild'
 inputs={'train':np.load(PARENT/'draw_ids.npy'),'reference':np.load(SEAL/'ref_idx.npy'),'query':np.load(SEAL/'val_idx.npy')};n={'train':300000,'reference':250000,'query':10660};checks=[]
 def ck(v,s):assert bool(v),s;checks.append(s)
 allids=np.concatenate(list(inputs.values()));ck(len(np.unique(allids))==len(allids),'train/ref/query mutually disjoint');ck(int(allids.min())>=0 and int(allids.max())<19344847,'all IDs in existing CLIP pool')
 before={str(p):fingerprint(p) for p in [RAW,POOL/'source.npy',POOL/'id.npy']};manifest_sha=sha(POOL/'clip512-manifest.json');source=np.load(POOL/'source.npy',mmap_mode='r',allow_pickle=True);x=np.load(RAW,mmap_mode='r');ck(x.shape==(19344847,512) and x.dtype==np.dtype('f4'),'raw shape/dtype')
 saved={};raw_hash={};counts={}
 for tag,ids in inputs.items():
  ck(ids.shape==(n[tag],),tag+' declared count');np.save(D/(tag+'-ids.npy'),ids);groups=np.asarray(source[ids]).astype(str);np.save(D/(tag+'-source.npy'),groups);counts[tag]={str(g):int((groups==g).sum()) for g in np.unique(groups)}
  if tag=='train':ck(np.array_equal(groups,np.load(PARENT/'draw_source.npy',allow_pickle=True).astype(str)),'original training source identity')
  if tag=='query':ck(np.array_equal(groups,np.load(SEAL/'val_source.npy',allow_pickle=True).astype(str)),'original validation source identity')
  out=np.lib.format.open_memmap(D/(tag+'.f16.npy'),mode='w+',dtype='f2',shape=(len(ids),512));digest=hashlib.sha256();norm_lo=float('inf');norm_hi=0.
  for lo in range(0,len(ids),8192):
   hi=min(len(ids),lo+8192);buf=np.array(x[ids[lo:hi]],dtype='f4',copy=True);digest.update(buf.tobytes());norm=np.linalg.norm(buf,axis=1,keepdims=True);ck(np.isfinite(buf).all() and (norm>0).all(),tag+f' finite nonzero block{lo}');norm_lo=min(norm_lo,float(norm.min()));norm_hi=max(norm_hi,float(norm.max()));buf/=norm;out[lo:hi]=buf.astype('f2');x._mmap.madvise(mmap.MADV_DONTNEED)
  out.flush();del out;raw_hash[tag]=digest.hexdigest();saved[tag]={'normalized_fp16_sha':sha(D/(tag+'.f16.npy')),'raw_selected_ordered_sha':digest.hexdigest(),'raw_norm_min':norm_lo,'raw_norm_max':norm_hi}
  # Re-read fixed first/last source rows independently; compare exact stored bytes and native source IDs.
  bank=np.load(D/(tag+'.f16.npy'),mmap_mode='r');sel=np.unique(np.concatenate([np.flatnonzero(groups==g)[[0,-1]] for g in np.unique(groups)]))
  for j in sel:
   v=np.array(x[int(ids[j])],dtype='f4');v/=np.linalg.norm(v);ck(np.array_equal(v.astype('f2'),bank[j]),tag+f' independent byte join row{j}')
 panel=np.load(O/'card013-scoring/closeout-persist.npz')['panel_local'];ck(len(panel)==1800 and len(np.unique(panel))==1800,'fixed continuity panel');np.save(D/'panel-local.npy',panel)
 rng=np.random.default_rng(49049);train_audit=np.unique(np.r_[0,299999,rng.choice(np.arange(1,299999),126,replace=False)]);qgroups=np.load(D/'query-source.npy');query_audit=np.concatenate([np.flatnonzero(qgroups==g)[[0,-1]] for g in np.unique(qgroups)]);ck(len(train_audit)==128 and len(query_audit)==18,'fixed graph/truth audits');np.savez(D/'neighbor-audit-selection.npz',train_local=train_audit,query_local=query_audit)
 torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);p=ParametricUMAP.load(str(CHAMP),device='cpu');p.model=None;p.n_components=3;p.hidden_dim=2048;p._init_model(512)
 state={k:v.detach().cpu().clone() for k,v in p.model.state_dict().items()};h=hashlib.sha256()
 for k in sorted(state):h.update(k.encode());h.update(np.ascontiguousarray(state[k].numpy()).tobytes())
 probe=torch.from_numpy(np.array(np.load(D/'train.f16.npy',mmap_mode='r')[:32],dtype='f4'));out=p.model(probe);ck(out.shape==(32,3) and bool(torch.isfinite(out).all()),'fresh512->3 forward');out.square().mean().backward();ck(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in p.model.parameters()),'fresh512->3 gradients')
 torch.save({'model_state':state,'seed':42,'input_dim':512,'n_components':3,'hidden_dim':2048,'n_params':sum(v.numel() for v in state.values()),'state_sha256':h.hexdigest()},D/'init.pt')
 after={p:fingerprint(p) for p in before};ck(before==after,'raw file fingerprints unchanged');ck(sha(POOL/'clip512-manifest.json')==manifest_sha,'CLIP source manifest unchanged')
 files={p.name:sha(p) for p in D.iterdir() if p.is_file() and p.suffix in ['.npy','.npz','.pt']};result={'status':'CPU_INPUTS_READY_GRAPH_PENDING','PASS':True,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'n_train':300000,'dim':512,'n_components':3,'n_params':sum(v.numel() for v in state.values()),'init_state_sha256':h.hexdigest(),'files':files,'normalization':saved,'raw_fingerprints':before,'clip_manifest_sha':manifest_sha,'source_counts':counts,'checks':checks,'n_checks':len(checks),'wall_s':time.monotonic()-start,'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'builder_sha':sha(Path(__file__)),'protocol_sha':sha(O/'card049-clip-scale-transfer.md'),'input_contract':'Original selected CLIP FP32 rows normalized in FP32, then stored FP16. Graph/encoder truth will cosine-normalize stored vectors; CPU map scoring normalizes the same stored vectors in FP32. Training uses device FP16.','alignment_scope':'Reuses existing provenance-aligned CLIP pool; row/source joins checked here, original upstream shard alignment not re-derived.'};ck(result['max_rss_mib']<6144,'RSS under6GiB');result['n_checks']=len(checks);write(D/'inputs-manifest.json',result);write(O/'card049-inputs-ready.json',{'PASS':True,'input_manifest_sha':sha(D/'inputs-manifest.json'),'wall_s':result['wall_s'],'max_rss_mib':result['max_rss_mib'],'n_checks':len(checks),'GPU_s':0});print(json.dumps({k:result[k] for k in ['status','n_params','init_state_sha256','wall_s','max_rss_mib','n_checks']},indent=2))
if __name__=='__main__':main()
