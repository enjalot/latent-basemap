"""Project only the immutable inactive census with the confirmed 062 operator."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
import time, signal, datetime as dt, fcntl, resource
import numpy as np
import torch
import card062_common as C
O=C.O; R=C.R; D=O/'card064-projection'; START=None; RC=999
def verify():
    rel=C.read(O/'card064-release.json')
    assert rel['PASS'] and all(C.sha(p)==h for p,h in rel['files'].items())
    assert all(C.sha(R/p)==h for p,h in C.read(R/'card064-runtime-sha.json').items())
    return C.sha(O/'card064-release.json')
def main():
    global START,RC
    START=time.monotonic(); end=dt.datetime.fromisoformat(C.read(O/'owner-autonomous-24h-20260912.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
    remaining=min(300-C.read(O/'card064-ledger.json')['batch_spent_s'],165491-C.read(O/'cards-24h-window-ledger.json')['spent_s'],end-time.time())
    assert remaining>30
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('card064 cap'))); signal.alarm(int(remaining)-5)
    release=verify(); C.device_setup(); assert C.read(O/'card063-scoring/independent-audit.json')['PASS']
    assert C.read(O/'card063-score.json')['passing_operator_candidates']==C.HEADS
    inputs=C.read(O/'card064-inputs/manifest.json'); D.mkdir(exist_ok=False); heads={}
    for name in C.HEADS:
        row=inputs['heads'][name]; m,c,radius=C.load(name,'cuda'); assert c==row['constant_norm']
        exposed=np.load(O/f'card057-inputs/{name}-normalized.f32.npy',mmap_mode='r')[:256]
        expected=np.load(O/f'card059-scoring/{name}-coords.npz')['target_direction_constant_norm'][:256]
        assert np.array_equal(C.project(m,exposed,c)['xy'],expected),'256-row frozen exposed canary'
        x=np.load(O/f'card064-inputs/{name}-normalized.f32.npy',mmap_mode='r'); ids=np.load(O/f'card064-inputs/{name}-ids.npz'); n=len(x)
        z=np.lib.format.open_memmap(D/f'{name}-preactivation.npy',mode='w+',dtype='f4',shape=(n,m.up[0].out_features))
        xy=np.empty((n,3),'f4'); teacher=xy.copy(); delta=np.empty((n,3),'f8')
        for lo in range(0,n,4096):
            hi=min(n,lo+4096); p=C.project(m,x[lo:hi],c,collect=True)
            assert p['inactive'].all(),'historical census no longer all-inactive on exact CPU-normalized inputs'
            assert np.array_equal(p['teacher'],np.broadcast_to(ids['teacher_bias'],(hi-lo,3))),'teacher bias differs'
            xy[lo:hi]=p['xy']; teacher[lo:hi]=p['teacher']; delta[lo:hi]=p['delta64']; z[lo:hi]=p['preactivation']
            free,total=torch.cuda.mem_get_info(); assert (total-free)/2**30<30
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<7500
        z.flush(); del z
        movement=np.linalg.norm(xy.astype('f8')-teacher,axis=1)/radius
        assert movement.max()<=.0011 and np.isfinite(xy).all()
        np.savez(D/f'{name}.npz',ids=ids['ids'],teacher=teacher,xy=xy,delta64=delta,output_weight=m.proj_out.weight.cpu().numpy())
        heads[name]={'n':n,'constant_norm':c,'radius':radius,'projection_sha':C.sha(D/f'{name}.npz'),'preactivation_sha':C.sha(D/f'{name}-preactivation.npy'),'max_native_movement_R':float(movement.max()),'changed_rows':int(np.any(xy.view('u4')!=teacher.view('u4'),axis=1).sum())}
        del m; torch.cuda.empty_cache()
    verify(); C.write(D/'manifest.json',{'status':'CENSUS_PROJECTED_NOT_PATCHED','heads':heads,'release_sha':release,'wall_s':time.monotonic()-START})
    C.write(O/'card064-execution.json',{'status':'CENSUS_PROJECTED_NOT_PATCHED','manifest_sha':C.sha(D/'manifest.json'),'quality':'Full-reference quality not yet measured; CPU patch audit required'})
    RC=0
if __name__=='__main__':
    try: main()
    except Exception as e: C.write(O/'card064-execution.json',{'status':'FAILED','error':repr(e)}); raise
    finally:
        if START is not None:
            seconds=time.monotonic()-START
            with (O/'window-ledger-write.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX)
                for p,key in [(O/'card064-ledger.json','batch_spent_s'),(O/'cards-24h-window-ledger.json','spent_s')]:
                    r=C.read(p);r[key]+=seconds;r.setdefault('entries',[]).append({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'064','tag':'census_projection','event':'exclusive_GPU_stage','wall_s':seconds,'rc':RC});C.write(p,r)
