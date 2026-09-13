"""Root-released one-stage projection; all initial checks/failures occupy budget."""
import os,sys,time,subprocess,signal,shutil
from pathlib import Path
import card091_common as C
import card091_budget as B
PY='/home/enjalot/code/latent-basemap/.venv/bin/python'
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
    descendants={os.getpid(),pid}
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


def active(unit):
 p=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True,timeout=5)
 assert p.returncode in (0,3),'cannot establish CPU service state'
 return p.stdout.strip() in ('active','activating','deactivating')
def cpu_state():
 out={'active':active('basemap-card090-score.service')}
 if out['active']:
  props=subprocess.check_output(['systemctl','--user','show','basemap-card090-score.service','-p','MemoryMax','-p','MemorySwapMax'],text=True,timeout=5)
  values=dict(line.split('=',1) for line in props.splitlines() if '=' in line)
  out['memory_max_bytes']=int(values['MemoryMax']);out['memory_swap_max_bytes']=int(values['MemorySwapMax'])
 for k,n in [('release','card090-score-release.json'),('launch','card090-score-launch.json'),('terminal','card090-cpu-execution.json')]:
  p=C.O/n
  if p.exists():out[k]=C.read(p)
 if 'release' in out and 'launch' in out:
  out['launch_release_match']=out['launch'].get('cmd',[])[-1:]==[C.sha(C.O/'card090-score-release.json')]
 if 'terminal' in out and 'launch' in out:
  out['terminal_fresh']=(C.O/'card090-cpu-execution.json').stat().st_mtime_ns>=(C.O/'card090-score-launch.json').stat().st_mtime_ns
 return out

def main():
 start=time.monotonic();reserved=0.;child=None;rc=999;error=None;peak_rss=0;peak_vram=0
 try:
  release=C.O/'card091-release.json';assert release.exists(),'NO ROOT GPU RELEASE'
  os.environ['CARD091_RELEASE_SHA']=C.sha(release);r=C.require_release();verify_external_leases()
  # Settle/replay journal before snapshot. Reserve remaining budget only once.
  B.transact(0.,'recovery');prior=C.read(B.L)['spent_s'];window=C.read(B.W)['spent_s']
  limit=min(3600-prior,165491-window,C.END-time.time()-1200)
  assert limit>120,'no full-stage budget'
  assert shutil.disk_usage(C.O).free>=16*2**30,'disk16GiB STOP'
  C.cpu090_bound(cpu_state(),time.time())
  B.transact(limit,'reservation',check=True);reserved=limit
  env=dict(os.environ,CARD091_OCCUPANCY_START=str(start),CARD091_PRIOR_SPENT=str(prior),CARD091_PRIOR_WINDOW=str(window),CARD091_ACTIVE_RESERVATION=str(limit))
  began=time.time_ns();child=subprocess.Popen([PY,str(C.S/'run_card091_gpu.py')],cwd=C.R,env=env,start_new_session=True)
  while child.poll() is None:
   assert time.monotonic()-start<limit-2 and time.time()<C.END-1202,'GPU deadline/cumulative timeout'
   rss,vram=usage(child.pid);peak_rss=max(peak_rss,rss);peak_vram=max(peak_vram,vram);time.sleep(.5)
  rc=child.returncode;assert rc==0,'projection subprocess failed'
  for n in ['loader-reference.json','buffered-projection-admission.json','device-canary.json','preflight.json','execution.json']:
   C.receipt(C.D/n,began,r['runtime_sha'],r['selection_sha'],C.sha(release))
  ex=C.read(C.D/'execution.json');assert ex['status']=='PROJECTED_NOT_SCORED' and ex['rows']==C.HIGH and list(ex['models'])==C.ARMS,'incomplete projection'
  C.require_release()
 except BaseException as e:error=repr(e)
 finally:
  if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
  actual=time.monotonic()-start
  B.transact(actual-reserved,'settlement')
  passed=error is None and C.read(B.L)['spent_s']<=3600 and C.read(B.W)['spent_s']<=165491 and time.time()<=C.END-1200
  C.write(C.O/'card091-chain-execution.json',{'PASS':passed,'status':'PROJECTED_NOT_SCORED' if passed else 'EXECUTION_STOP','error':error,'rc':rc,'wall_s':actual,'peak_tree_rss_bytes':peak_rss,'peak_global_vram_gib':peak_vram,'prior_costs_preserved':True})
  if not passed:C.write(C.O/'card091-execution.json',{'status':'EXECUTION_STOP','error':error,'partial_outputs':'preserved incomplete'})
 return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
