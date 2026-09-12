"""Bounded CPU preparation of fixed original training/evaluation inputs and continuity instrument."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,time,json,resource,datetime as dt
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import numpy as np,torch
from scipy.spatial.distance import cdist
import card040_common as V
D=V.DATA;O=V.O

def main():
 start=time.monotonic();torch.set_num_threads(2);D.mkdir(exist_ok=True);assert not (D/'inputs.json').exists(),'inputs already frozen';assert V.sha(V.PCA)==V.PCA_SHA and V.sha(V.BODY)==V.BODY_SHA and V.sha(V.LEAKY)==V.LEAKY_SHA
 draw=np.load(V.DRAW);src=np.load(V.SOURCE);assert draw.shape==(300000,) and len(np.unique(draw))==300000 and src.shape==draw.shape
 z=np.load(V.C/'per-query.npz');qid=z['query_ids'];rid=z['reference_ids'];assert len(qid)==10110 and len(rid)==200000;checks={}
 excluded={'card021_confirmation_queries':qid,'card032_fresh':np.load(O/'card032-density/queries.npz')['query_ids'],'card037_fresh':np.load(O/'card037-confirmation/selection-ids.npz')['query_ids']}
 for name,ids in excluded.items():checks['no_train_overlap_'+name]=not np.isin(draw,ids).any()
 assert all(checks.values()),checks
 np.save(D/'draw_ids.npy',draw);np.save(D/'draw_source.npy',src);np.savez(D/'evaluation-ids.npz',**{k:z[k] for k in ['query_ids','reference_ids','general','failure','source','truth_local']});pc=np.load(V.PCA);mean=torch.from_numpy(pc['mean']);comp=torch.from_numpy(pc['components'])
 for label,ids in [('train',draw),('reference',rid),('query',qid)]:
  out=np.lib.format.open_memmap(D/f'{label}-input.f32.npy',mode='w+',dtype='f4',shape=(len(ids),768))
  with torch.inference_mode():
   for lo in range(0,len(ids),4096):
    raw=V.raw(ids[lo:lo+4096]);y=torch.nn.functional.normalize((torch.from_numpy(raw)-mean)@comp,dim=1);assert y.shape==(len(raw),768) and torch.isfinite(y).all();out[lo:lo+len(y)]=y.numpy()
  out.flush();del out;print('prepared',label,len(ids),flush=True)
 # Independent FP64 preprocessing on fixed training/evaluation identity spots.
 for label,ids in [('train',draw),('reference',rid),('query',qid)]:
  ix=np.unique(np.r_[0,len(ids)//2,len(ids)-1]);x=V.raw(ids[ix]).astype('f8');y=(x-pc['mean'].astype('f8'))@pc['components'].astype('f8');y/=np.linalg.norm(y,axis=1,keepdims=True);got=np.load(D/f'{label}-input.f32.npy',mmap_mode='r')[ix];checks[label+'_PCA_FP64_fidelity']=bool(np.allclose(got,y,rtol=1e-5,atol=1e-5))
 rng=np.random.default_rng(40041);panel=np.sort(rng.choice(np.flatnonzero(z['general']),1800,replace=False));hd=V.raw(qid[panel]).astype('f8');hd/=np.linalg.norm(hd,axis=1,keepdims=True);dist=cdist(hd,hd,'cosine');np.fill_diagonal(dist,np.inf);near=np.argsort(dist,axis=1,kind='stable')[:,:15];np.savez(D/'continuity-panel.npz',local=panel,query_ids=qid[panel],encoder_hd=hd,encoder15=near)
 torch.manual_seed(V.SEED);init=V.Readout();torch.save({'model':init.state_dict(),'seed':V.SEED,'configuration':'Linear2048_64-GELU-Linear64_2; outputzero'},D/'init.pt');checks['zero_output_init']=bool((init(torch.ones(3,2048))==0).all());checks['readout_parameters']=sum(p.numel() for p in init.parameters())==131266
 sources={}
 for name in ['pool-20m','pool-complement-88m']:
  p=Path('/data2/monet')/name/'dino1536.f16.npy';st=p.stat();sources[str(p)]={k:getattr(st,k) for k in ['st_size','st_mtime_ns','st_dev','st_ino']}
 assert all(checks.values()),checks;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert rss<8192
 manifest={'PASS':True,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'checks':checks,'n_train':len(draw),'n_reference':len(rid),'n_queries':len(qid),'train_draw_sha':V.sha(V.DRAW),'teacher_sha':V.BODY_SHA,'leaky_sha':V.LEAKY_SHA,'PCA_sha':V.PCA_SHA,'init_state_sha':V.state_sha(init.state_dict()),'readout_parameters':sum(p.numel() for p in init.parameters()),'prior_arrays_sha':V.sha(V.C/'per-query.npz'),'source_fingerprints':sources,'files':{p.name:V.sha(p) for p in D.glob('*.n*')},'init_file_sha':V.sha(D/'init.pt'),'builder_sha':V.sha(__file__),'plan_sha':V.sha(O/'card040-preactivation-readout.md'),'cpu_s':time.monotonic()-start,'max_rss_mib':rss,'scope':'CPU PCA inputs and IDs only; no040body features/targets/candidate scoring,GPUcanary ortraining. Original300K draw reused, freshquery reserves excluded.'};V.atomic(D/'inputs.json',manifest);V.atomic(O/'card040-inputs.json',manifest);print(json.dumps({'PASS':True,'checks':checks,'cpu_s':manifest['cpu_s'],'RAMMiB':rss}))
if __name__=='__main__':main()
