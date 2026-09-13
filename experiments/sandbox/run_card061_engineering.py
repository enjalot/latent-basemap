"""Exposed-panel device proof + frozen end-to-end benchmark; no fresh outcomes."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import time,datetime as dt,signal,fcntl,subprocess,resource
import numpy as np,torch
import card061_common as C
START=None;RC=999;CAP=300;OUT=C.O/'card061-engineering';OUT.mkdir(exist_ok=True)
def bench(m,x,c,reps):
 for _ in range(3):C.project(m,x,c,'teacher');C.project(m,x,c,'repair')
 vals={'teacher':[],'repair':[]}
 for i in range(reps):
  for mode in ['teacher','repair'] if i%2==0 else ['repair','teacher']:
   torch.cuda.synchronize();start=time.perf_counter();C.project(m,x,c,mode);torch.cuda.synchronize();vals[mode].append(time.perf_counter()-start)
 return {**vals,'median_ratio':float(np.median(vals['repair'])/np.median(vals['teacher'])),'rows':len(x)}
def main():
 global START,RC
 START=time.monotonic();ledger=C.O/'card061-ledger.json';window=C.O/'cards-24h-window-ledger.json'
 if not ledger.exists():C.write(ledger,{'batch_cap_s':900,'engineering_cap_s':300,'batch_spent_s':0.,'entries':[]})
 prior=C.read(ledger)['batch_spent_s'];end=dt.datetime.fromisoformat(C.read(C.O/'owner-autonomous-24h-20260912.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
 remaining=min(CAP-prior,end-time.time(),165491-C.read(window)['spent_s']);assert remaining>30
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('phase A cap')));signal.alarm(max(1,int(remaining)-5));runtime=C.verify();C.device_setup()
 input_ref=np.load(C.O/'card057-inputs/reference-normalized.f32.npy',mmap_mode='r');indices=np.load(C.O/'card057-benchmark.npz')['mixed_reference_rows'];assert len(indices)==8192
 records={};checks=[]
 for name in C.HEADS:
  m,c,radius=C.load(name,'cuda');old=np.load(C.O/f'card057-projection/{name}.npz');expected=np.load(C.O/f'card059-scoring/{name}-coords.npz');target=np.load(C.O/f'card057-inputs/{name}-normalized.f32.npy');mixed=np.array(input_ref[indices]);saved={};ph={}
  for pop,x,ref_teacher,ref_mask,ref_y in [('target',target,old['target_teacher'],old['target_inactive'],expected['target_direction_constant_norm']),('mixed',mixed,old['reference_teacher'][indices],old['reference_inactive'][indices],expected['reference_direction_constant_norm'][indices])]:
   rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state().clone();p=C.project(m,x,c,collect=True);assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_gpu,torch.cuda.get_rng_state());checks.append(name+pop+' no RNG consumption')
   assert np.array_equal(p['teacher'].view('u4'),ref_teacher.view('u4'));assert np.array_equal(p['inactive'],ref_mask);assert np.array_equal(p['xy'].view('u4'),ref_y.view('u4'));assert np.array_equal(p['xy'][~ref_mask].view('u4'),ref_teacher[~ref_mask].view('u4'));checks.append(name+pop+' teacher/mask/candidate/active exact')
   rows=np.flatnonzero(ref_mask);zz=p['preactivation'][rows].astype(np.longdouble);w=m.proj_out.weight.cpu().numpy().astype(np.longdouble);v=zz@w.T;n=np.sqrt((v*v).sum(1));d=np.divide(v*c,n[:,None],out=np.zeros_like(v),where=n[:,None]>0);yy=(p['teacher'][rows].astype(np.longdouble)+d).astype('f4');assert np.array_equal(yy,p['xy'][rows]);checks.append(name+pop+' independent scalar FP32 output')
   movement=np.linalg.norm(p['xy'].astype('f8')-p['teacher'].astype('f8'),axis=1)/radius;assert movement.max()<=.0011;ph[pop]={'inactive':len(rows),'max_native_movement':float(movement.max()),'zero_direction_rows':int((n==0).sum())}
   saved.update({pop+'_'+k:p[k] for k in ['xy','teacher','delta64','inactive']});saved[pop+'_inactive_preactivation']=p['preactivation'][rows];saved[pop+'_inactive_rows']=rows
   # Same-shaped permutation changes grouping/order, not numerical batch size.
   perm=np.arange(len(x)-1,-1,-1);pp=C.project(m,x[perm],c)['xy'];assert np.array_equal(pp[np.argsort(perm)],p['xy']);checks.append(name+pop+' matched-shape ordering')
  saved['weight']=m.proj_out.weight.cpu().numpy();np.savez(OUT/(name+'-canary.npz'),**saved)
  timing={'mixed':bench(m,mixed,c,20),'collapsed':bench(m,target,c,10)};records[name]={'constant_norm':c,'radius':radius,'movement':ph,'timing':timing,'speed_PASS':timing['mixed']['median_ratio']<=1.10}
  del m;torch.cuda.empty_cache()
  free,total=torch.cuda.mem_get_info();assert (total-free)/2**30<30
 C.verify();r={'status':'ENGINEERING_COMPLETE','heads':records,'correctness_PASS':True,'checks':checks,'n_checks':len(checks),'speed_PASS':all(v['speed_PASS'] for v in records.values()),'fresh_projection_authorized':False,'runtime_sha':runtime,'release_sha':C.sha(C.O/'card061-engineering-release.json'),'rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'wall_s_before_closeout':time.monotonic()-START};assert r['rss_MiB']<8192;C.write(C.O/'card061-engineering-result.json',r);RC=0
 subprocess.run(['/home/enjalot/code/latent-basemap/.venv/bin/python',str(C.O/'notify.py'),'post','codex-overseer',str(C.O/'card061-engineering-result.json'),f'Card061 engineering-only test complete: correctness PASS; mixed-speed pass={r["speed_PASS"]}. Previously exposed inputs only; no fresh projection or promotion. Root owns next release.'],timeout=30)
if __name__=='__main__':
 try:main()
 except Exception as e:C.write(C.O/'card061-engineering-execution-failure.json',{'error':repr(e),'at':dt.datetime.now(dt.timezone.utc).isoformat(),'fresh_projection_authorized':False});raise
 finally:
  if START is not None:
   seconds=time.monotonic()-START
   with (C.O/'window-ledger-write.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    for p,key in [(C.O/'card061-ledger.json','batch_spent_s'),(C.O/'cards-24h-window-ledger.json','spent_s')]:
     r=C.read(p);r[key]+=seconds;r.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'061','tag':'engineering_phase_A','event':'exclusive_GPU_stage','wall_s':seconds,'rc':RC});C.write(p,r)
