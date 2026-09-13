"""Fresh constant-norm projection, controls and provenance; CPU scorer owns gates."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import time,signal,datetime as dt,fcntl,subprocess,resource
from pathlib import Path
import numpy as np,torch
import card062_common as C
O=C.O;R=C.R;D=O/'card063-projection';DATA=O/'card061-inputs';START=None;RC=999
def verify():
 rel=C.read(O/'card063-release.json');assert rel['PASS'] and all(C.sha(p)==h for p,h in rel['files'].items());m=C.read(R/'card063-runtime-sha.json');assert all(C.sha(R/p)==h for p,h in m.items());return C.sha(O/'card063-release.json')
def collect(m,x,c):
 out={k:[] for k in ['xy','teacher','delta64','inactive','inactive_preactivation','inactive_rows']}
 for lo in range(0,len(x),4096):
  p=C.project(m,x[lo:lo+4096],c,collect=True);idx=np.flatnonzero(p['inactive'])
  for k in ['xy','teacher','delta64','inactive']:out[k].append(p[k])
  out['inactive_preactivation'].append(p['preactivation'][idx]);out['inactive_rows'].append(idx+lo)
 return {k:np.concatenate(v) for k,v in out.items()}
def main():
 global START,RC
 START=time.monotonic();ledger=O/'card063-ledger.json'
 if not ledger.exists():C.write(ledger,{'batch_cap_s':300,'batch_spent_s':0.,'entries':[]})
 end=dt.datetime.fromisoformat(C.read(O/'owner-autonomous-24h-20260912.json')['deadline_utc'].replace('Z','+00:00')).timestamp();rem=min(300-C.read(ledger)['batch_spent_s'],165491-C.read(O/'cards-24h-window-ledger.json')['spent_s'],end-time.time());assert rem>30
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('projection cap')));signal.alarm(int(rem)-5);release=verify();C.device_setup();eng=C.read(O/'card062-engineering-result.json');assert eng['speed_PASS'] and eng['correctness_PASS'];panel=np.load(O/'card061-panel/panel.npz');D.mkdir(exist_ok=False);manifest={};shas={}
 for hi,name in enumerate(C.HEADS):
  m,c,radius=C.load(name,'cuda');stored=np.load(O/f'card059-scoring/{name}-coords.npz');exposed=np.load(O/f'card057-inputs/{name}-normalized.f32.npy');probe=C.project(m,exposed[:32],c)['xy'];assert np.array_equal(probe,stored['target_direction_constant_norm'][:32]),'isolated runtime exposed canary'
  saved={'output_weight':m.proj_out.weight.cpu().numpy()};parts={}
  for pi,(pop,key,pkey) in enumerate([('reference','reference','reference_ids'),('target',name,name+'_query_ids'),('general','general','general_query_ids')]):
   x=np.load(DATA/f'{key}-normalized.f32.npy',mmap_mode='r');p=collect(m,x,c);ids=panel[pkey];teacher=p['teacher'];mask=p['inactive'];idx=p['inactive_rows'];pred=np.load(O/f'card035-full/{name}-positive-final.npy',mmap_mode='r')[ids]==0;assert np.array_equal(mask,pred),'actual support drift';stored_key='query' if pop=='target' else pop;old=panel[f'{name}_{stored_key}_teacher'];assert np.all(np.abs(teacher.astype('f8')-old)<=2e-4+2e-5*np.abs(old)),'stored teacher fidelity';assert np.array_equal(p['xy'][~mask].view('u4'),teacher[~mask].view('u4'));assert np.linalg.norm(p['xy'].astype('f8')-teacher,axis=1).max()/radius<=.0011
   # Original057 control: scalar independent of the new fixed-norm candidate.
   z=p['inactive_preactivation'].astype('f8');v=z@saved['output_weight'].astype('f8').T;mar=np.maximum(-z.max(1),0);delta=1e-4*(mar/(mar+1))[:,None]*v;n=np.linalg.norm(delta,axis=1);delta*=np.minimum(1,.001*radius/np.maximum(n,np.finfo('f8').tiny))[:,None];orig=teacher.copy();orig[idx]=(teacher[idx].astype('f8')+delta).astype('f4')
   saved.update({pop+'_ids':ids,pop+'_teacher':teacher,pop+'_repair':p['xy'],pop+'_original':orig,pop+'_inactive':mask,pop+'_inactive_rows':idx,pop+'_inactive_preactivation':p['inactive_preactivation'],pop+'_delta64':p['delta64'],pop+'_original_delta64':delta})
   for draw in range(3):
    rng=np.random.default_rng(630630+100*hi+10*pi+draw);u=rng.normal(size=(len(idx),3));u/=np.linalg.norm(u,axis=1,keepdims=True);iso=teacher.copy();iso[idx]=(teacher[idx].astype('f8')+c*u).astype('f4');label='isotropic' if draw==0 else f'isotropic{draw}';saved[pop+'_'+label]=iso;saved[pop+'_'+label+'_direction']=u
   parts[pop]={'stored_teacher_max_abs_difference':float(np.abs(teacher.astype('f8')-old).max()),'inactive_rows':len(idx)}
   free,total=torch.cuda.mem_get_info();assert (total-free)/2**30<30
  np.savez(D/f'{name}.npz',**saved);timing=eng['heads'][name]['timing'];timing={k:{**v,'ratio':v['median_ratio']} for k,v in timing.items()};manifest[name]={'sha':C.sha(D/f'{name}.npz'),'radius':radius,'constant_norm':c,'parts':parts,'timings':timing};del m;torch.cuda.empty_cache()
 verify();C.write(D/'manifest.json',{'status':'PROJECTED_VALIDATED','heads':manifest,'release_sha':release,'runtime_sha':C.sha(R/'card063-runtime-sha.json'),'precision':'FP32 inputs and teacher; FP64 correction; FP32 candidate/controls;256neuralbatch; exact062operator; no training','wall_s':time.monotonic()-START});assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<8192;C.write(O/'card063-execution.json',{'status':'PROJECTED_BENCHMARKED_VALIDATED','batch_spent_s':time.monotonic()-START,'manifest_sha':C.sha(D/'manifest.json'),'quality':'NOT_SCORED','benchmark_reused':'unchanged062 operator/implementation'});RC=0
 subprocess.run(['/home/enjalot/code/latent-basemap/.venv/bin/python',str(O/'notify.py'),'post','codex-overseer',str(O/'card063-execution.json'),'Card063 fresh-query confined-repair projection validated; CPU confirmation scoring/audit next. No gate from trainer; no full-corpus projection.'],timeout=30)
if __name__=='__main__':
 try:main()
 except Exception as e:C.write(O/'card063-execution.json',{'status':'FAILED','error':repr(e)});raise
 finally:
  if START is not None:
   sec=time.monotonic()-START
   with (O/'window-ledger-write.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    for p,key in [(O/'card063-ledger.json','batch_spent_s'),(O/'cards-24h-window-ledger.json','spent_s')]:
     r=C.read(p);r[key]+=sec;r.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'063','tag':'fresh_projection','event':'exclusive_GPU_stage','wall_s':sec,'rc':RC});C.write(p,r)
