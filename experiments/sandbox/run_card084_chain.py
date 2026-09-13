"""Root-released, dual-lease, resource-monitored durable Card084 chain. No scoring."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import subprocess,datetime as dt,time,signal,sys
from pathlib import Path
import card084_common as C
import card084_budget as B
PY='/home/enjalot/code/latent-basemap/.venv/bin/python'
S=C.R/'experiments/sandbox'
STAGE_TIME=0.
STAGE_RECEIPTS={}

def verify():
    release=C.O/'card084-release.json'
    assert release.exists(),'NO ROOT GPU RELEASE'
    r=C.read(release)
    assert r['PASS'] and r['card']=='084' and r['limits']==B.LIMITS
    assert r['runtime_sha']==C.source_check() and all(C.sha(p)==h for p,h in r['files'].items())
    return C.sha(release)

LEASE_PATHS=('/data/latent-basemap/.gpu_lease','/data/latent-basemap/sandbox/.gpu.lock')

def verify_external_leases():
    ancestors=set();pid=os.getpid()
    while pid>1 and pid not in ancestors:
        ancestors.add(pid)
        status=Path(f'/proc/{pid}/status').read_text()
        pid=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
    held=set()
    for line in Path('/proc/locks').read_text().splitlines():
        fields=line.split()
        if len(fields)>=8 and fields[1]=='FLOCK' and fields[3]=='WRITE' and int(fields[4]) in ancestors:
            major,minor,inode=fields[5].split(':')
            held.add((int(major,16),int(minor,16),int(inode)))
    for path in LEASE_PATHS:
        st=os.stat(path)
        assert (os.major(st.st_dev),os.minor(st.st_dev),st.st_ino) in held, 'missing actual external flock: '+path

def usage(pid):
    rows=subprocess.check_output(['ps','-eo','pid=,ppid=,rss='],text=True,timeout=2)
    processes={int(p):(int(parent),int(rss)*1024) for p,parent,rss in (line.split() for line in rows.splitlines())}
    descendants={pid}
    while True:
        expanded=descendants|{p for p,(parent,_) in processes.items() if parent in descendants}
        if expanded==descendants:break
        descendants=expanded
    rss=sum(processes.get(p,(0,0))[1] for p in descendants)
    output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
    vram=sum(float(line.strip()) for line in output.splitlines())/1024
    assert rss<32*2**30,'RSS32GiB STOP'
    assert vram<30,'global VRAM30GiB STOP'
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    assert mem['MemAvailable']>4*2**30,'host available RAM below4GiB STOP'
    return rss,vram

def validate_stage_receipt(path, started_ns, runtime_sha, calibration_sha):
    path=Path(path)
    assert path.is_file(), 'missing stage receipt'
    assert path.stat().st_mtime_ns>=started_ns, 'stale stage receipt'
    receipt=C.read(path)
    assert receipt.get('PASS') is True, 'stage receipt lacks explicit PASS'
    assert receipt.get('runtime_sha')==runtime_sha, 'stage receipt runtime mismatch'
    assert receipt.get('calibration_sha')==calibration_sha, 'stage receipt calibration mismatch'
    return receipt

def stage(tag,script,cap,args=(),arm=None,receipt_path=None):
    global STAGE_TIME
    release=verify();limit=min(cap,B.available(arm)-2)
    assert limit>5,'budget/deadline STOP'
    B.transact(tag,limit,arm,check=True);start=time.monotonic();child=None;rc=999
    peak_rss=0;peak_vram=0;stage_error=None;proof=None
    runtime_sha=C.source_check();calibration_sha=C.sha(C.CAL) if receipt_path is not None else None
    started_ns=time.time_ns()
    try:
        env=dict(os.environ,CARD084_RELEASE_SHA=release)
        child=subprocess.Popen([PY,str(S/script),*args],cwd=C.R,env=env,start_new_session=True)
        while child.poll() is None:
            if time.monotonic()-start>=limit-1 or time.time()>=B.END-1:raise TimeoutError('bounded stage timeout; no dose truncation acceptance')
            rss,vram=usage(child.pid);peak_rss=max(peak_rss,rss);peak_vram=max(peak_vram,vram)
            time.sleep(.5)
        rc=child.returncode
        assert rc==0,tag+' failed with '+str(rc)
        if receipt_path is not None:
            assert C.source_check()==runtime_sha, 'stage runtime changed while running'
            assert C.sha(C.CAL)==calibration_sha, 'stage calibration changed while running'
            receipt=validate_stage_receipt(receipt_path,started_ns,runtime_sha,calibration_sha)
            proof={'path':str(receipt_path),'sha':C.sha(receipt_path),'started_ns':started_ns,
                   'mtime_ns':Path(receipt_path).stat().st_mtime_ns}
            STAGE_RECEIPTS[tag]={'receipt':receipt,'proof':proof}
    except BaseException as error:
        stage_error=repr(error)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
        elapsed=time.monotonic()-start
        STAGE_TIME+=elapsed
        B.transact(tag,elapsed-limit,arm,kind='settlement')
        C.write(C.O/f'card084-stage-{tag}.json',{'PASS':rc==0 and stage_error is None,'rc':rc,'error':stage_error,'receipt_proof':proof,'stage_started_ns':started_ns,'wall_s':elapsed,'peak_tree_rss_bytes':peak_rss,'peak_global_vram_gib':peak_vram,'runtime_sha':C.source_check()})
    verify()
    return elapsed

def main():
    verify() # no lease or GPU activity without root's immutable release
    verify_external_leases() # inspect real ancestor-owned locks; never reacquire
    start=time.monotonic();stage_time=0.;reserved=False
    try:
        # Shared controller time distributed equally to the two cumulative arm caps.
        B.transact('controller',120.,check=True);reserved=True
        C.input_check()
        assert all(C.read(C.TD/a/'preparation.json')['READY'] for a in C.ARMS)
        if C.CAL.exists():
            assert C.read(C.CAL)['PASS'] and C.read(C.CAL)['runtime_sha']==C.source_check()
        else:stage_time+=stage('calibration','gpu_card084_calibrate.py',180)
        stage_time+=stage('graph_canary','gpu_card084_graph_canary.py',400,receipt_path=C.O/'card084-graph-canary.json')
        for a in C.ARMS:stage_time+=stage('preflight-'+a,'gpu_card084_preflight.py',300,(a,),a,receipt_path=C.O/f'card084-preflight-{a}.json')
        estimates={}
        for a in C.ARMS:
            pf=STAGE_RECEIPTS['preflight-'+a]['receipt']
            assert pf['PASS'] and pf['runtime_sha']==C.source_check() and pf['calibration_sha']==C.sha(C.CAL)
            estimates[a]=pf['estimate']['complete_arm_s']
        ledger=C.read(B.L);window=C.read(B.W)
        checks={'card_cap':ledger['batch_spent_s']+sum(estimates.values())<=4200,
                'window_cap':window['spent_s']+sum(estimates.values())<=165491,
                'deadline':sum(estimates.values())+120<=B.END-time.time()}
        checks.update({a:ledger['arm_spent_s'][a]+estimates[a]<=1800 for a in C.ARMS})
        C.write(C.O/'card084-preflight.json',{'PASS':all(checks.values()),'checks':checks,'estimates':estimates,
            'ledger_at_admission':ledger,'runtime_sha':C.source_check(),'calibration_sha':C.sha(C.CAL)})
        if not all(checks.values()):
            C.write(C.O/'card084-execution.json',{'status':'ADMISSION_STOP','reason':'Full60K measured dose does not fit cumulative limits; no truncation'})
            return
        for a in C.ARMS:
            assert B.available(a)>=estimates[a],'remaining full dose no longer fits'
            stage_time+=stage(a,'run_card084_arm.py',1800,(a,),a)
        C.write(C.O/'card084-execution.json',{'status':'TRAINED_VALIDATED','quality':'NOT_SCORED',
             'runtime_sha':C.source_check(),'calibration_sha':C.sha(C.CAL)})
    except BaseException as e:
        C.write(C.O/'card084-execution.json',{'status':'EXECUTION_STOP','error':repr(e)})
        raise
    finally:
        if reserved:B.transact('controller',max(0.,time.monotonic()-start-STAGE_TIME)-120.,kind='settlement')
if __name__=='__main__':main()
