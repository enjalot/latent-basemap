"""Bounded full-corpus3D projections with an unmodified-model activation sensor."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,time,resource,tempfile,datetime as dt,hashlib,mmap,copy
import numpy as np
import torch
import card091_common as C
O=C.O;R=C.R;D=C.D;B=C.B;sys.path[:0]=[str(C.S),str(R)]
from card032_engine import sha,atomic,save_checkpoint,load_checkpoint
from card035_projection import Projector
from card035_buffered_projection import BufferedProjector
from basemap.pumap.parametric_umap.core import ParametricUMAP
LOW=19344847;HIGH=103816750;CHUNK=32768;HEAD_BATCH=256
RAW=[(0,LOW,Path('/data2/monet/pool-20m/dino1536.f16.npy')),(LOW,HIGH,Path('/data2/monet/pool-complement-88m/dino1536.f16.npy'))]
def raw(ids):
 x=np.empty((len(ids),1536),'f2')
 for lo,hi,p in RAW:
  take=np.flatnonzero((ids>=lo)&(ids<hi))
  if len(take):
   mm=np.load(p,mmap_mode='r');mm._mmap.madvise(mmap.MADV_RANDOM);x[take]=mm[ids[take]-lo];mm._mmap.madvise(mmap.MADV_DONTNEED);del mm
 return x
def main():
 start=time.monotonic();release=C.require_release();D.mkdir(exist_ok=True)
 runtime_sha=C.source_check();selection_sha=sha(D/'selection.json');release_sha=sha(O/'card091-release.json')
 def proof(path,payload):
  payload.update(runtime_sha=runtime_sha,selection_sha=selection_sha,release_sha=release_sha);atomic(path,payload)
 s=json.loads((D/'selection.json').read_text());assert sha(D/'selection.json')==release['selection_sha'];C.validate_selection(s)
 def sources():
  for p,ex in s['raw_file_fingerprints'].items():
   st=Path(p).stat();assert {k:getattr(st,k) for k in ex}==ex,('source changed',p)
  for p,h in s['source_manifest_hashes'].items():assert sha(p)==h
 sources();C.verify_sources(s);assert sha(B/'selection.json')==s['card032_selection_sha'] and sha(B/'progress.json')==s['card032_progress_sha'] and sha(B/'execution.json')==s['card032_execution_sha']
 hist32=json.loads((B/'progress.json').read_text())['history'];raw_expected={(h['lo'],h['hi']):h['raw_sha'] for h in hist32};assert len(raw_expected)==len(hist32)
 from card035_stream_layout import validate_raw_history
 assert validate_raw_history(hist32,LOW,HIGH,CHUNK)['PASS']
 assert C.source_check()==s['runtime_manifest_sha']
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');assert torch.cuda.is_available()
 import subprocess
 loader_start=time.time_ns();subprocess.run(['/home/enjalot/code/latent-basemap/.venv/bin/python',str(C.S/'card091_loader_reference.py')],check=True,timeout=120)
 loader=C.receipt(D/'loader-reference.json',loader_start,runtime_sha,selection_sha,release_sha);assert sha(D/'loader-reference.npz')==loader['output_sha'];loader_arrays=np.load(D/'loader-reference.npz');models={}
 for a,m in s['models'].items():assert sha(m['path'])==m['sha'];models[a]=ParametricUMAP.load(m['path'],device='cpu').model.cuda().eval().requires_grad_(False)
 def resources():
  free,total=torch.cuda.mem_get_info();used=(total-free)/2**30;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert used<30 and rss<32768,(used,rss)
  from run_card091_chain import usage
  tree_rss,global_vram=usage(os.getpid());return {'global_vram_gib':global_vram,'tree_rss_bytes':tree_rss,'process_peak_rss_mib':rss}
 # Pre-output engineering selection: arithmetic/inner batch unchanged, bulk CPU copies only.
 bench_ids=np.load(B/'reference-250000-ids.npy')[:CHUNK];bx=torch.nn.functional.normalize(torch.tensor(raw(bench_ids).astype('f4'),device='cuda'),dim=1)
 def bench(kind):
  pp=kind(models,HEAD_BATCH);torch.cuda.synchronize();t=time.monotonic();yy=pp(bx);torch.cuda.synchronize();elapsed=time.monotonic()-t;pp.close();resources();return yy,elapsed
 legacy,_=bench(Projector);buffered,_=bench(BufferedProjector);assert all(np.array_equal(legacy[k],buffered[k]) for k in legacy),'buffered coordinates/support counts differ';bt={'legacy':[],'buffered':[]}
 for rep in range(3):
  order=[('legacy',Projector),('buffered',BufferedProjector)]
  if rep%2:order.reverse()
  for label,kind in order:
   yy,elapsed=bench(kind);assert all(np.array_equal(legacy[k],yy[k]) for k in legacy),'benchmark repeat differs';bt[label].append(elapsed)
 speed=float(np.median(bt['legacy'])/np.median(bt['buffered']));use_buffered=bool(speed>=1/0.9);ProjectionClass=BufferedProjector if use_buffered else Projector
 proof(D/'buffered-projection-admission.json',{'PASS':True,'all_projected_coordinates_counts_bitwise':True,'probe_ids_sha':hashlib.sha256(bench_ids.tobytes()).hexdigest(),'n':len(bench_ids),'head_batch':HEAD_BATCH,'timings_s':bt,'median_speedup':speed,'selected':'buffered' if use_buffered else 'legacy','selection_rule':'buffered if >=10percent less median time; sameFP32 inner256batches and bitwise coordinates/counts required','helper_sha':sha(C.S/'card035_buffered_projection.py'),'global_resources':resources()})
 del bx,legacy,buffered,yy;projector=ProjectionClass(models,HEAD_BATCH)
 def predict(buf):
  x=torch.as_tensor(buf.astype('f4'),device='cuda');assert bool(torch.isfinite(x).all()) and bool((torch.linalg.vector_norm(x,dim=1)>0).all());x=torch.nn.functional.normalize(x,dim=1);return projector(x)
 def allocate(path,n,mode='w+'):
  return {key:np.lib.format.open_memmap(path/(key+'.npy'),mode=mode,dtype=dtype,shape=shape) for a in models for key,dtype,shape in [(a,'f4',(n,3)),(a+'-positive-final','u2',(n,))]}
 with torch.inference_mode():
  # Predetermined reference rows only. No900query outcomes are consulted by admission.
  canids=np.load(B/'reference-250000-ids.npy')[:1024];buf=raw(canids);gpu_x=torch.nn.functional.normalize(torch.tensor(buf.astype('f4'),device='cuda'),dim=1);cpu_x=torch.nn.functional.normalize(torch.tensor(buf.astype('f4')),dim=1);normgap=float((gpu_x.cpu()-cpu_x).abs().max());assert normgap<=s['numeric_tolerance']['normalized_input_abs'];ys=projector(gpu_x);projector.close();can={};saved={'ids':canids,'normalized_gpu':gpu_x.cpu().numpy(),'normalized_cpu':cpu_x.numpy()}
  abs_tol,rel_tol=s['numeric_tolerance']['forward_abs_plus_relative']
  for a,m in models.items():
   unhook=np.concatenate([m(gpu_x[lo:lo+HEAD_BATCH]).cpu().numpy() for lo in range(0,len(buf),HEAD_BATCH)]);assert np.array_equal(unhook,ys[a]),'hook changed coordinates'
   statehash=hashlib.sha256()
   for k,v in sorted(m.state_dict().items()):statehash.update(k.encode());statehash.update(v.cpu().numpy().tobytes())
   C.validate_loader_output(canids,unhook,statehash.hexdigest(),loader_arrays['ids'],loader_arrays[a],loader['states'][a])
   try:C.validate_loader_output(canids[::-1],unhook,statehash.hexdigest(),loader_arrays['ids'],loader_arrays[a],loader['states'][a]);raise AssertionError('wrong probe order accepted')
   except AssertionError as e:assert 'original090 loader query order mismatch' in str(e)
   cpu=copy.deepcopy(m).cpu();y32=np.concatenate([cpu(cpu_x[lo:lo+HEAD_BATCH]).numpy() for lo in range(0,len(buf),HEAD_BATCH)]);cpu=cpu.double();x64=torch.nn.functional.normalize(torch.tensor(buf.astype('f8')),dim=1);y64=np.concatenate([cpu(x64[lo:lo+HEAD_BATCH]).numpy() for lo in range(0,len(buf),HEAD_BATCH)]);err=np.abs(ys[a].astype('f8')-y64);bound=abs_tol+rel_tol*np.abs(y64);assert np.all(err<=bound),(a,float(err.max()),float((err/bound).max()));can[a]={'fp64_max_abs':float(err.max()),'fp32_max_abs':float(np.max(np.abs(ys[a]-y32))),'max_error_over_componentwise_bound':float((err/bound).max()),'hook_bitexact':True,'wrong_probe_id_order_rejected':True,'constant_probe_outputs':bool(np.array_equal(unhook,unhook[::-1]))};saved[a+'_gpu']=ys[a];saved[a+'_cpu32']=y32;saved[a+'_cpu64']=y64;del cpu,x64,y32,y64
  projector=ProjectionClass(models,HEAD_BATCH);np.savez(D/'device-fidelity.npz',**saved)
  with tempfile.TemporaryDirectory(dir=str(D),prefix='canary-') as td:
   td=Path(td);out=allocate(td,1024);history=[]
   for lo in range(0,1024,256):
    hi=lo+256;y=predict(buf[lo:hi]);h={'lo':lo,'hi':hi,'sha':{}}
    for a,v in y.items():out[a][lo:hi]=v;h['sha'][a]=hashlib.sha256(v.tobytes()).hexdigest()
    history.append(h)
    if hi==512:
     fixture_identity=selection_sha+':'+release_sha;save_checkpoint(td,hi,{},out,fixture_identity,history)
     for v in out.values():v.flush()
     del out;out=allocate(td,1024,'r+');cursor,st,history=load_checkpoint(td,fixture_identity,out,'cuda');assert cursor==512 and st=={}
     try:load_checkpoint(td,'wrong-model-or-input',out,'cuda');raise AssertionError('wrong identity accepted')
     except ValueError as e:assert 'admission-identity mismatch' in str(e)
   expected=predict(buf);assert all(np.array_equal(out[a],v) for a,v in expected.items()),'resumed output/count mismatch'
  proof(D/'device-canary.json',{'PASS':True,'models':can,'normalization_max_abs':normgap,'head_batch':HEAD_BATCH,'resumed_coordinates_counts_bitexact':True,'wrong_identity_rejected':True,'projection_sha':sha(C.S/'card035_projection.py'),'fidelity_sha':sha(D/'device-fidelity.npz'),'resources':resources()});del gpu_x,cpu_x,ys,buf,saved,expected,out
  # Full path measured at multiple source positions; checkpoint metadata padded to final history size.
  timings=[]
  with tempfile.TemporaryDirectory(dir=str(D),prefix='preflight-') as td:
   td=Path(td);out=allocate(td,CHUNK)
   for lo in [0,CHUNK,LOW+1000000,LOW+20000000,HIGH-2*CHUNK,HIGH-CHUNK]:
    t=time.monotonic();buf=raw(np.arange(lo,lo+CHUNK));digest=hashlib.sha256(buf.tobytes()).hexdigest();y=predict(buf);h={'lo':0,'hi':CHUNK,'raw_sha':digest,'sha':{}}
    for a,v in y.items():out[a][:]=v;h['sha'][a]=hashlib.sha256(v.tobytes()).hexdigest()
    save_checkpoint(td,CHUNK,{},out,'preflight',[h]*len(hist32));torch.cuda.synchronize();timings.append(time.monotonic()-t);resources()
  del out,y,buf
  from run_card091_chain import cpu_state
  import shutil
  done=0
  if (D/'progress.json').exists():
   prior= C.read(D/'progress.json');assert prior['identity']==selection_sha+':'+release_sha,'resume admission identity mismatch';done=prior['cursor']
   expected_prefix=[h for h in hist32 if h['hi']<=done]
   assert len(prior['history'])==len(expected_prefix) and (not expected_prefix and done==0 or expected_prefix and expected_prefix[-1]['hi']==done),'resume cursor/layout mismatch'
   assert all((a['lo'],a['hi'],a['raw_sha'])==(b['lo'],b['hi'],b['raw_sha']) for a,b in zip(prior['history'],expected_prefix)),'resume raw history mismatch'
  estimate=float(max(timings)*sum(h['hi']>done for h in hist32)*1.2+240)
  elapsed=time.monotonic()-float(os.environ['CARD091_OCCUPANCY_START'])
  import card091_budget as Budget
  reservation=float(os.environ['CARD091_ACTIVE_RESERVATION']);spent=C.settled_equivalent(C.read(Budget.L)['spent_s'],reservation,elapsed);window=C.settled_equivalent(C.read(Budget.W)['spent_s'],reservation,elapsed)
  cpustate=cpu_state();cpu_remaining=C.cpu090_bound(cpustate,time.time());checks=C.joint_checks(estimate,spent,window,cpu_remaining,time.time(),shutil.disk_usage(D).free)
  proof(D/'preflight.json',{'PASS':all(checks.values()),'checks':checks,'chunks_s':timings,'estimate_s_max_chunk_plus20pct_and240':estimate,'head_batch':HEAD_BATCH,'models':list(models),'resources':resources(),'cpu090_state':cpustate,'cpu090_remaining_bound_s':cpu_remaining,'settled_equivalent_actual_spent_s':spent,'window_actual_spent_s':window,'active_reservation_excluded':True,'remaining_exact_stream_chunks':sum(h['hi']>done for h in hist32),'note':'Complete four-head path, hashing/checkpoint metadata;240s final output hashing/closeout/resume validation reserve. Cache coldness uncontrolled.'})
  assert all(checks.values()),'entire four-head corpus fails measured joint admission'
  identity=sha(D/'selection.json')+':'+sha(O/'card091-release.json');resuming=(D/'progress.json').exists()
  if not resuming:assert not any((D/(a+suffix+'.npy')).exists() for a in C.ARMS for suffix in ['', '-positive-final']),'orphan outputs require root preservation/review'
  outputs=allocate(D,HIGH,'r+' if resuming else 'w+')
  if resuming:cursor,st,history=load_checkpoint(D,identity,outputs,'cuda');assert st=={}
  else:cursor=0;history=[]
  for base,end,p in RAW:
   if cursor>=end:continue
   source=np.load(p,mmap_mode='r')
   for lo in range(max(base,cursor),end,CHUNK):
    hi=min(lo+CHUNK,end);buf=np.array(source[lo-base:hi-base],dtype='f2');source._mmap.madvise(mmap.MADV_DONTNEED);digest=hashlib.sha256(buf.tobytes()).hexdigest();assert raw_expected[(lo,hi)]==digest,('raw032stream differs',lo,hi);y=predict(buf);h={'lo':lo,'hi':hi,'raw_sha':digest,'sha':{}}
    for a,v in y.items():outputs[a][lo:hi]=v;h['sha'][a]=hashlib.sha256(v.tobytes()).hexdigest()
    history.append(h);save_checkpoint(D,hi,{},outputs,identity,history);cursor=hi
    for m in outputs.values():m._mmap.madvise(mmap.MADV_DONTNEED)
    resources();assert time.time()<C.END-1200,'GPU CPU-reserve deadline STOP'
    if len(history)%32==0:print(json.dumps({'cursor':cursor,'rows':HIGH,'elapsed_s':time.monotonic()-start}),flush=True)
   del source
  assert cursor==HIGH;sources();qids=np.load(B/'queries.npz')['query_ids'];counts={}
  for a in models:
   np.save(D/f'{a}-query.npy',np.array(outputs[a][qids]));count=outputs[a+'-positive-final'];hist=np.zeros(models[a].up[0].out_features+1,dtype='i8');zero_parts=[]
   for lo in range(0,HIGH,1000000):
    v=np.array(count[lo:lo+1000000]);hist+=np.bincount(v,minlength=len(hist));zero_parts.append(np.flatnonzero(v==0)+lo);count._mmap.madvise(mmap.MADV_DONTNEED)
   zero=np.concatenate(zero_parts);np.save(D/f'{a}-inactive-global-ids.npy',zero);np.save(D/f'{a}-positive-count-hist.npy',hist);assert int(hist.sum())==HIGH;counts[a]={'all_inactive_n':len(zero),'fraction':len(zero)/HIGH,'query_sha':sha(D/f'{a}-query.npy'),'histogram_sha':sha(D/f'{a}-positive-count-hist.npy'),'zero_ids_sha':sha(D/f'{a}-inactive-global-ids.npy')}
  C.verify_sources(s)
  receipt={'status':'PROJECTED_NOT_SCORED','rows':HIGH,'models':s['models'],'head_batch':HEAD_BATCH,'TF32':False,'FP32':True,'projection_copy_policy':'buffered' if use_buffered else 'legacy','buffered_admission_sha':sha(D/'buffered-projection-admission.json'),'selection_sha':sha(D/'selection.json'),'release_sha':sha(O/'card091-release.json'),'progress_sha':sha(D/'progress.json'),'producer_sha':sha(__file__),'outputs':{a:sha(D/(a+'.npy')) for a in outputs},'inactive_counts':counts,'wall_s':time.monotonic()-start,'resources':resources(),'limits':'Counts diagnose one exact constant-output mechanism, not all bands. Root091 independent audit/scoring pending; historical42 context only; no promotion.'};receipt['PASS']=True;proof(D/'execution.json',receipt);atomic(O/'card091-execution.json',receipt);print(json.dumps(receipt,indent=2));projector.close()
if __name__=='__main__':main()
