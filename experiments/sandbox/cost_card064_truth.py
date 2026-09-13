"""Cost only: bounded FP64 exhaustive encoder/low-D scans; no quality selection."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
import time,signal,datetime as dt,fcntl,resource,hashlib
from pathlib import Path
import numpy as np
import torch
import card062_common as C
O=C.O;START=None;RC=999;LOW=19344847;HIGH=103816750;CHUNK=32768
def main():
    global START,RC
    START=time.monotonic();release=C.read(O/'card064-cost-release.json');assert release['PASS'] and all(C.sha(p)==h for p,h in release['files'].items())
    end=dt.datetime.fromisoformat(C.read(O/'owner-autonomous-24h-20260912.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
    rem=min(180,300-C.read(O/'card064-ledger.json')['batch_spent_s'],165491-C.read(O/'cards-24h-window-ledger.json')['spent_s'],end-time.time());assert rem>30
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('cost probe cap')));signal.alarm(int(rem)-5);C.device_setup()
    panel=np.load(O/'card061-panel/panel.npz');qids=np.concatenate([panel['ordinary2m_query_ids'],panel['half4m_query_ids'],panel['general_query_ids']]);assert len(qids)==2948 and len(np.unique(qids))==2948
    paths=[Path('/data2/monet')/p/'dino1536.f16.npy' for p in ('pool-20m','pool-complement-88m')];raw=[np.load(p,mmap_mode='r') for p in paths]
    fingerprints=C.read(O/'card064-inputs/manifest.json')['raw_fingerprints']
    def unchanged():
        for p,fp in fingerprints.items():assert all(getattr(os.stat(p),k)==v for k,v in fp.items())
    def get(ids):
        x=np.empty((len(ids),1536),'f2');mask=ids<LOW;x[mask]=raw[0][ids[mask]];x[~mask]=raw[1][ids[~mask]-LOW];return x
    unchanged();q=torch.tensor(get(qids).astype('f8'),device='cuda');q=torch.nn.functional.normalize(q,dim=1)
    times=[];d_times=[];starts=[0,CHUNK,LOW+1000000,LOW+20000000,HIGH-2*CHUNK,HIGH-CHUNK]
    # Each probe evaluates ALL 2948 actual query vectors, but no neighbors/scores persist.
    with torch.inference_mode():
        for lo in starts:
            torch.cuda.synchronize();t=time.monotonic();buf=get(np.arange(lo,lo+CHUNK));x=torch.tensor(buf.astype('f8'),device='cuda');x=torch.nn.functional.normalize(x,dim=1)
            for k in range(0,len(q),256):
                sim=q[k:k+256]@x.T;take=torch.topk(sim,16,dim=1);assert torch.isfinite(take.values).all()
            torch.cuda.synchronize();times.append(time.monotonic()-t);del x,sim,take
        # Exact subtraction squared distances avoid dot-product cancellation at tiny offsets.
        maps={name:np.load(O/f'card064-full/{name}.npy',mmap_mode='r') for name in C.HEADS}
        for lo in starts[:3]:
            torch.cuda.synchronize();t=time.monotonic()
            for name in C.HEADS:
                qxy=torch.tensor(np.array(maps[name][qids]),device='cuda',dtype=torch.float64);rxy=torch.tensor(np.array(maps[name][lo:lo+CHUNK]),device='cuda',dtype=torch.float64)
                for k in range(0,len(qxy),128):
                    diff=qxy[k:k+128,None,:]-rxy[None,:,:];dist=(diff*diff).sum(2);take=torch.topk(dist,2000,dim=1,largest=False);assert torch.isfinite(take.values).all()
            torch.cuda.synchronize();d_times.append(time.monotonic()-t)
    unchanged();free,total=torch.cuda.mem_get_info();assert (total-free)/2**30<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<7500
    chunks=int(np.ceil(LOW/CHUNK)+np.ceil((HIGH-LOW)/CHUNK));result={'status':'COSTED_NOT_QUALITY_SCORED','query_n':len(qids),'reference_n_excluding_all_queries':HIGH-len(qids),'encoder_FP64_chunks_s':times,'lowD_two_repaired_heads_chunks_s':d_times,'chunk':CHUNK,'full_chunks':chunks,'encoder_two_pass_max_chunk_plus20pct_s':2*max(times)*chunks*1.2,'lowD_four_paths_two_pass_upper_estimate_s':4*max(d_times)*chunks*1.2,'scope':'FP64 raw normalization and exhaustive cosine top16, all2948 fixed queries; lowD exact subtraction/top2000. Timings only, no outcomes saved, no full scan admission. Estimates include conservative double passes and double lowD modes but need checkpoint/tie/output reserves. Cache coldness uncontrolled.','source_sha':C.sha(__file__),'release_sha':C.sha(O/'card064-cost-release.json'),'wall_s':time.monotonic()-START,'global_vram_GiB':(total-free)/2**30,'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    C.write(O/'card064-full-reference-cost.json',result);print(result,flush=True);RC=0
if __name__=='__main__':
    try:main()
    except Exception as e:C.write(O/'card064-full-reference-cost.json',{'status':'FAILED','error':repr(e)});raise
    finally:
        if START is not None:
            seconds=time.monotonic()-START
            with (O/'window-ledger-write.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX)
                for p,key in [(O/'card064-ledger.json','batch_spent_s'),(O/'cards-24h-window-ledger.json','spent_s')]:
                    r=C.read(p);r[key]+=seconds;r.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'064','tag':'full_reference_cost_only','event':'exclusive_GPU_stage','wall_s':seconds,'rc':RC});C.write(p,r)
