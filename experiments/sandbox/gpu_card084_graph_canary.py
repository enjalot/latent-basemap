"""Production fit/full-state resumes on a512-node fixture; no quality evaluation."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc
import numpy as np,torch
import card084_common as C
from card084_fit import fit

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

KEYS=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen',
      'loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node',
      'loader_node_at_rank','rankneg_scale','replay_gen','card081_noise_pairs',
      'mn_gen','mn_rng','dens_gen','dens_rng','hold_gen','hold_rng','deriv_gen',
      'global_step','epoch','config','current_replay_bank_path']

def main():
    C.require_gpu_stage();start=time.monotonic();checks=[];hashes={}
    with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card084-graph-canary-') as path:
        td=Path(path);n=512
        X=np.array(np.load(C.D/'train.f16.npy',mmap_mode='r')[:n],dtype='f4')
        graph=td/'edges.npz';src=np.repeat(np.arange(n,dtype='i4'),15)
        dst=np.array([(i+j)%n for i in range(n) for j in range(1,16)],dtype='i4')
        np.savez(graph,sources=src,targets=dst,weights=np.ones(len(src),'f4'),n_nodes=n)
        rp=td/'radii.npy';np.save(rp,np.load(C.D/'radii.npy')[:n])
        qp=td/'q.npy';np.save(qp,np.full(n,1/n))
        kw=dict(X=X,graph=graph,radius_path=rp,noise_q_path=qp,checkpoints=[2,4,7,9,18])
        for arm in C.ARMS:
            p,full,r=fit(arm,18,td/arm,**kw);hashes[arm]=r['state_sha']
            assert r['actual_attraction']=={k:r['identity'][k] for k in ('family','coefficient','delta')}
            del p;gc.collect();torch.cuda.empty_cache();mids=[];epochs=[]
            for cp in (td/arm/'ckpts').glob('*.pt'):
                ck=torch.load(cp,map_location='cpu',weights_only=False)
                if 0<ck['global_step']<18:
                    if ck['step_checkpoint'] and 0<ck['loader_pos_idx']<len(ck['loader_perm']):mids.append((ck['global_step'],cp))
                    if not ck['step_checkpoint'] and ck['epoch']<full['epoch']:epochs.append((ck['global_step'],cp))
            assert mids and epochs
            for tag,cp in [('mid',min(mids)[1]),('epoch',min(epochs)[1])]:
                p,res,report=fit(arm,18,td/(arm+'-'+tag),resume=cp,**kw)
                assert all(same(full[k],res[k]) for k in KEYS),arm+' '+tag+' full-state differs'
                assert full['card012_identity']==res['card012_identity']
                for k in ('positive_lr_optimizer_steps','executed_iters','optimizer_steps_succeeded','attempted_batches','amp_overflow_skips','nonfinite_loss_skips','nonfinite_gradient_skips'):
                    assert full['train_stats'][k]==res['train_stats'][k],k
                assert report['actual_attraction']==r['actual_attraction'],'hook not reconstructed'
                checks.append(arm+' '+tag+' exact full-state resume with hook reconstructed')
                del p,res;gc.collect();torch.cuda.empty_cache()
            for tag,override in [('family',{'family':'pseudo_huber' if arm=='quadratic' else 'quadratic'}),
                ('coefficient',{'coefficient':r['identity']['coefficient']*2}),('delta',{'delta':C.DELTA*2}),
                ('calibration',{'calibration_sha':'0'*64}),('source',{'runtime_manifest_sha':'0'*64}),
                ('bank',{'bank_sha':'0'*64}),('dose',{'dose':19}),('batches',{'calibration_batches_sha':'0'*64})]:
                # Validation runs before model configuration or fit; no updates on rejection.
                try:fit(arm,18,td/(arm+'-reject-'+tag),resume=min(mids)[1],ident_override=override,**kw)
                except (AssertionError,ValueError):checks.append(arm+' wrong '+tag+' rejected')
                else:raise AssertionError('wrong identity accepted: '+tag)
        off={'coefficient':0.,'canary_off_control':True}
        p,base,_=fit('quadratic',18,td/'absent',ident_override=off,omit_hook=True,**kw)
        del p;gc.collect();torch.cuda.empty_cache()
        p,zero,_=fit('quadratic',18,td/'zero',ident_override=off,**kw)
        assert all(same(base[k],zero[k]) for k in KEYS)
        assert base['card012_identity']==zero['card012_identity']
        checks.append('absent vs coefficient0 exact model/Adam/scaler/scheduler/sampler/RNG identity')
        assert len(set([*hashes.values(),C.state_sha(base['model'])]))==3
        checks.append('both enabled families diverge from zero control and each other')
    C.write(C.O/'card084-graph-canary.json',{'PASS':True,'checks':checks,'wall_s':time.monotonic()-start,
        'runtime_sha':C.source_check(),'calibration_sha':C.sha(C.CAL),'endpoints':hashes,
        'scope':'Actual production1536D/2048hidden/3D batch16384 fit,512-node synthetic graph with real training features/radii; mid+real epoch full-state resume. Full2M throughput measured separately; not a full2M epoch-resume twin.'})
if __name__=='__main__':main()
