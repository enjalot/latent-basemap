"""Actual strictFP32 device feature/target bank, with complete manifest last."""
from pathlib import Path
import sys,json,time,resource
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import numpy as np,torch
import card040_common as V
D=V.DATA

def main():
 start=time.monotonic();runtime=V.runtime_check();inputs=V.validate_inputs()
 if (D/'bank.json').exists():V.validate_bank();print('bank already valid');return 0
 bank_names=['pre.f32.npy','teacher-xy.f32.npy','fixed-xy.f32.npy','target-xy.f32.npy','delta.f32.npy','inactive-rows.npy','active-rows.npy','positive-final.npy','normalization.npz']
 partial=[D/n for n in bank_names if (D/n).exists()]
 if partial:
  archive=D/('bank-attempt-'+str(time.time_ns()));archive.mkdir()
  for p in partial:p.rename(archive/p.name)
  print('preserved partial bank',archive,flush=True)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');assert torch.cuda.is_available();t=V.load_body('teacher','cuda');l=V.load_body('leaky','cuda');th=V.state_sha(t.state_dict());lh=V.state_sha(l.state_dict());x=np.load(D/'train-input.f32.npy',mmap_mode='r');n=len(x);assert n==300000
 outputs={'pre.f32.npy':(n,2048),'teacher-xy.f32.npy':(n,2),'fixed-xy.f32.npy':(n,2),'target-xy.f32.npy':(n,2),'delta.f32.npy':(n,2)};mm={k:np.lib.format.open_memmap(D/k,mode='w+',dtype='f4',shape=shape) for k,shape in outputs.items()};positive=np.zeros(n,'i4')
 with torch.inference_mode():
  for lo in range(0,n,V.FORWARD_BATCH):
   xb=torch.from_numpy(np.array(x[lo:lo+V.FORWARD_BATCH])).cuda();ty,f,z=V.frozen_views(t,xb);ly=l(xb);c=(ty.double()+V.ALPHA*(ly.double()-ty.double())).float();delta=(c.double()-f.double()).float();vals=[z,ty,f,c,delta]
   for name,v in zip(outputs,vals):assert torch.isfinite(v).all();mm[name][lo:lo+len(xb)]=v.cpu().numpy()
   positive[lo:lo+len(xb)]=(z>0).sum(1).cpu().numpy()
 for v in mm.values():v.flush()
 assert V.state_sha(t.state_dict())==th and V.state_sha(l.state_dict())==lh,'frozen body changed';assert all(p.grad is None for m in [t,l] for p in m.parameters());inactive=np.flatnonzero(positive==0);active=np.flatnonzero(positive>0);np.save(D/'inactive-rows.npy',inactive);np.save(D/'active-rows.npy',active);np.save(D/'positive-final.npy',positive);target_scale=float(np.sqrt(np.mean(np.square(mm['delta.f32.npy'].astype('f8')))))
 if len(inactive)<40 or len(active)<40 or not np.isfinite(target_scale) or target_scale<=0:
  V.atomic(V.O/'card040-bank-stop.json',{'status':'ADMISSION_STOP','inactive_n':len(inactive),'active_n':len(active),'target_scale':target_scale if np.isfinite(target_scale) else None,'reason':'unchanged bank strata/target viability rule failed; no reselection'});return 3
 stats={}
 for basis in V.ARMS:
  total=np.zeros(2048,'f8');sq=np.zeros(2048,'f8')
  for lo in range(0,n,4096):
   z=np.array(mm['pre.f32.npy'][lo:lo+4096],'f8');z=np.maximum(z,0) if basis=='post' else z;total+=z.sum(0);sq+=np.square(z).sum(0)
  mean=total/n;std=np.sqrt(np.maximum(sq/n-mean*mean,0));stats[basis+'_mean']=mean.astype('f4');stats[basis+'_std']=np.maximum(std,.001).astype('f4');assert np.isfinite(stats[basis+'_std']).all()
 np.savez(D/'normalization.npz',**stats);ids=np.load(D/'draw_ids.npy');source=np.load(D/'draw_source.npy');free,total=torch.cuda.mem_get_info();used=(total-free)/2**30;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert used<30 and rss<12288
 files=list(outputs)+['inactive-rows.npy','active-rows.npy','positive-final.npy','normalization.npz'];m={'PASS':True,'input_manifest_sha':V.sha(D/'inputs.json'),'teacher_sha':V.BODY_SHA,'leaky_sha':V.LEAKY_SHA,'runtime_manifest_sha':runtime,'files':{name:V.sha(D/name) for name in files},'target_scale':target_scale,'inactive_n':len(inactive),'active_n':len(active),'inactive_by_source':{str(g):int((source[inactive]==g).sum()) for g in np.unique(source)},'ordered_draw_sha':V.sha(D/'draw_ids.npy'),'precision':'float32','TF32':False,'batch_size':V.FORWARD_BATCH,'global_used_gib':used,'max_rss_mib':rss,'stage_wall_s':time.monotonic()-start,'loaded_modules':V.loaded_check(),'scope':'Actual frozen teacher/continuation GPU outputs on original300K training rows; no evaluation outputs used for bank/normalization/selection. Target residual isC-F, not an all-coordinate model.'};V.atomic(D/'bank.json',m);V.validate_bank();print(json.dumps({k:m[k] for k in ['PASS','inactive_n','active_n','target_scale','stage_wall_s']}));return 0
if __name__=='__main__':raise SystemExit(main())
