"""Budgeted, lease-protected inference diagnostic; quality scoring is CPU-only."""
from pathlib import Path
import sys,json,time,datetime as dt,signal,resource
import numpy as np
import torch
from card057_common import O,R,DATA,HEADS,sha,atomic,load_head,project,kernel,device_setup,loaded_files
sys.path.insert(0,str(O))
from card057_confined_repair import repair,shuffled_direction_control

D=O/'card057-projection'
START=time.monotonic();CHARGED=0.;STAGES=[]
def verify():
    release=json.loads((O/'card057-release.json').read_text())
    assert release['PASS'] and all(sha(p)==h for p,h in release['files'].items()),'release/source identity'
    manifest=json.loads((R/'card057-runtime-sha.json').read_text())
    assert all(sha(R/p)==h for p,h in manifest.items()),'runtime identity'
    return release

def resources():
    free,total=torch.cuda.mem_get_info();used=(total-free)/2**30
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    assert used<30 and rss<14000,(used,rss)
    return {'global_GPU_GiB':used,'max_RSS_MiB':rss}

def charge(tag,seconds,rc):
    global CHARGED
    entry={'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'057','tag':tag,'event':'exclusive_GPU_stage','wall_s':seconds,'rc':rc}
    window=json.loads((O/'cards-24h-window-ledger.json').read_text())
    window['spent_s']+=seconds;window['entries'].append(entry)
    atomic(O/'cards-24h-window-ledger.json',window)
    CHARGED+=seconds;STAGES.append(entry)
    atomic(O/'card057-ledger.json',{'batch_cap_s':900,'batch_spent_s':CHARGED,'entries':STAGES})

def stage(name,fn):
    start=time.monotonic();rc=1
    try:
        result=fn();resources();rc=0;return result
    finally:charge(name,time.monotonic()-start,rc)

def expired(*args):raise TimeoutError('card057 occupancy/window deadline')

def check_stored(name,key,result,panel,ids):
    stored_key='reference' if key=='reference' else 'general' if key=='general' else 'query'
    stored=panel[name+'_'+stored_key+'_teacher']
    assert np.all(np.abs(result['teacher'].astype('f8')-stored)<=2e-4+2e-5*np.abs(stored)),(name,key,'stored teacher fidelity')
    count=np.load(O/f'card035-full/{name}-positive-final.npy',mmap_mode='r')
    assert np.array_equal(result['inactive'],np.asarray(count[ids])==0),(name,key,'original support mask')
    live=result['teacher'];out=result['xy'];active=~result['inactive']
    assert np.array_equal(out[active].view('u4'),live[active].view('u4')),(name,key,'active bytes')

def canary():
    checks=[];saved={}
    panel=np.load(O/'card057-panel/panel.npz')
    for name in HEADS:
        m,radius=load_head(name,'cuda')
        q=np.load(DATA/f'{name}-normalized.f32.npy')[:32]
        x=torch.from_numpy(q.copy()).cuda();held={}
        h=m.up[0].register_forward_hook(lambda module,args,z:held.update(z=z))
        with torch.inference_mode():
            teacher=m(x);z=held['z'];fast,d,mask=kernel(teacher,z,m.proj_out.weight,radius)
            ref=repair(teacher,z,m.proj_out.weight,radius)
            assert torch.equal(fast,ref['xy32']),'device reference FP32 mismatch'
            assert torch.allclose(d,ref['delta64'],rtol=1e-12,atol=1e-14)
            assert mask.all(),'targeted canary no longer collapsed'
        h.remove()
        assert np.all(np.abs(teacher.cpu().numpy().astype('f8')-panel[name+'_query_teacher'][:32])<=2e-4+2e-5*np.abs(panel[name+'_query_teacher'][:32]))
        saved[name+'_teacher']=teacher.cpu().numpy();saved[name+'_z']=z.cpu().numpy()
        saved[name+'_fast']=fast.cpu().numpy();saved[name+'_reference']=ref['xy32'].cpu().numpy()
        # Mixed synthetic preactivations exercise active, zero-boundary and
        # capped paths on this device, independently of real quality outcomes.
        gen=torch.Generator(device='cuda').manual_seed(570577)
        zz=torch.randn((64,m.proj_out.in_features),generator=gen,device='cuda')
        zz[:16]=-zz[:16].abs()-1;zz[16:24]=-zz[16:24].abs();zz[16:24,0]=0
        tt=torch.randn((64,3),generator=gen,device='cuda')
        yy,dd,ii=kernel(tt,zz,m.proj_out.weight,radius);rr=repair(tt,zz,m.proj_out.weight,radius)
        assert torch.equal(yy,rr['xy32']) and torch.equal(yy[~ii],tt[~ii]) and torch.equal(yy[16:24],tt[16:24])
        checks.append(name+' device real/synthetic reference,mask,boundary,active PASS')
        del m,x,teacher,z,fast,d,ref,zz,tt,yy,dd,rr;torch.cuda.empty_cache()
    np.savez(D/'canary-arrays.npz',**saved)
    return {'PASS':True,'checks':checks,'arrays_sha':sha(D/'canary-arrays.npz')}

def timed(m,x,radius,mode):
    torch.cuda.synchronize();start=time.monotonic();out=project(m,x,radius,mode)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    return elapsed,out

def preflight():
    ref=np.load(DATA/'reference-normalized.f32.npy',mmap_mode='r');rows={}
    for name in HEADS:
        m,radius=load_head(name,'cuda');project(m,ref[:512],radius)
        t1,_=timed(m,ref[:2048],radius,'repair');t2,_=timed(m,ref[:8192],radius,'repair')
        slope=(t2-t1)/6144
        assert np.isfinite(slope) and slope>0,'degenerate throughput fit'
        # Includes all250K projections, both benchmark conditions, validation
        # transfers, serialization and fixed overhead. No truncation on miss.
        estimate=max(slope,t2/8192)*400000*1.5+30
        rows[name]={'t2048_s':t1,'t8192_s':t2,'seconds_per_row':slope,'conservative_remaining_s':estimate}
        del m;torch.cuda.empty_cache()
    required=sum(x['conservative_remaining_s'] for x in rows.values())+60
    assert time.monotonic()-START+required<900,'complete workload does not fit cap'
    return {'PASS':True,'heads':rows,'required_remaining_s':required}

def production():
    panel=np.load(O/'card057-panel/panel.npz');bench=np.load(O/'card057-benchmark.npz');records={}
    for hi,name in enumerate(HEADS):
        m,radius=load_head(name,'cuda');arrays={'output_weight':m.proj_out.weight.cpu().numpy()};parts={}
        for si,key in enumerate(('reference',name,'general')):
            x=np.load(DATA/f'{key}-normalized.f32.npy',mmap_mode='r')
            ids=panel['reference_ids'] if key=='reference' else panel['general_query_ids'] if key=='general' else panel[name+'_query_ids']
            result=project(m,x,radius,collect=True);check_stored(name,key,result,panel,ids)
            # CPU independent reference reconstruction from persisted FP32
            # preactivations; require actual rounded output equality.
            ix=result['inactive_rows'];z=torch.from_numpy(result['inactive_preactivation'])
            with torch.inference_mode():
                cpu=repair(torch.from_numpy(result['teacher'][ix]),z,m.proj_out.weight.cpu(),radius)
            assert np.array_equal(cpu['xy32'].numpy(),result['xy'][ix]),(name,key,'CPU reference rounded result')
            norms=np.linalg.norm(result['delta64'],axis=1);valid=np.flatnonzero(result['inactive']&(norms>0))
            perm=np.random.default_rng(570570+si+hi).permutation(len(valid)).astype('i8')
            if key=='general' and len(valid)<2:
                shuffled=result['teacher'].copy();shuffle_delta=np.zeros_like(result['delta64']);uninformative=True
            else:
                control=shuffled_direction_control(torch.from_numpy(result['teacher']).cuda(),torch.from_numpy(result['delta64']).cuda(),torch.from_numpy(result['inactive']).cuda(),torch.from_numpy(perm).cuda())
                shuffled=control['xy32'].cpu().numpy();shuffle_delta=control['delta64'].cpu().numpy();uninformative=False
            prefix='target' if key==name else key
            arrays[prefix+'_ids']=ids
            for label,value in result.items():arrays[prefix+'_'+('repair' if label=='xy' else label)]=value
            arrays[prefix+'_shuffled']=shuffled;arrays[prefix+'_shuffle_delta64']=shuffle_delta;arrays[prefix+'_permutation']=perm
            parts[prefix]={'nonzero_donors':len(valid),'permutation_fixed_points':int((perm==np.arange(len(perm))).sum()),'general_control_unchanged':uninformative}
        timings={}
        for label,x in [('mixed',np.asarray(np.load(DATA/'reference-normalized.f32.npy',mmap_mode='r')[bench['mixed_reference_rows']])),('collapsed',np.load(DATA/f'{name}-normalized.f32.npy'))]:
            for mode in ('teacher','repair'):project(m,x,radius,mode)
            times={'teacher':[],'repair':[]}
            for repeat in range(5):
                for mode in (('teacher','repair') if repeat%2==0 else ('repair','teacher')):
                    elapsed,_=timed(m,x,radius,mode);times[mode].append(elapsed)
            times['ratio']=float(np.median(times['repair'])/np.median(times['teacher']))
            times['rows']=len(x);timings[label]=times
        timings['mixed']['inactive_rows']=int(arrays['reference_inactive'][bench['mixed_reference_rows']].sum())
        np.savez(D/f'{name}.npz',**arrays)
        records[name]={'sha':sha(D/f'{name}.npz'),'radius':radius,'parts':parts,'timings':timings,'resources':resources()}
        atomic(D/f'{name}-receipt.json',records[name]);del m,arrays;torch.cuda.empty_cache()
    return records

def main():
    status='EXECUTION_FAILED';error=None
    try:
        release=verify();window=json.loads((O/'owner-autonomous-24h-20260912.json').read_text())
        deadline=dt.datetime.fromisoformat(window['deadline_utc'].replace('Z','+00:00'))
        room=(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()
        ledger=json.loads((O/'cards-24h-window-ledger.json').read_text())
        assert room>=900 and ledger['spent_s']+900<=window['gpu_window_cap_s'],'owner window admission'
        signal.signal(signal.SIGTERM,expired);signal.signal(signal.SIGALRM,expired);signal.alarm(890)
        D.mkdir(exist_ok=False);device_setup()
        c=stage('device_canary',canary);atomic(D/'device-canary.json',c)
        p=stage('throughput_preflight',preflight);atomic(D/'preflight.json',p)
        heads=stage('full_projection_and_benchmark',production)
        verify();status='PROJECTED_BENCHMARKED_VALIDATED'
        atomic(D/'manifest.json',{'status':status,'heads':heads,'release_sha':sha(O/'card057-release.json'),'loaded_modules':loaded_files(),'torch':torch.__version__,'numpy':np.__version__,'GPU_name':torch.cuda.get_device_name(0),'precision':'FP32 inputs/teacher, TF32off, FP64 correction, actual FP32 output;256batch; no training'})
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        charge('controller_reconciliation',max(0.,time.monotonic()-START-CHARGED),0 if error is None else 1)
        atomic(O/'card057-execution.json',{'status':status,'error':error,'stages':STAGES,'batch_spent_s':CHARGED,'at':dt.datetime.now(dt.timezone.utc).isoformat()})

if __name__=='__main__':main()
