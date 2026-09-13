"""Card091 immutable contract. Import has no data/device/ledger side effects."""
from pathlib import Path
import json,hashlib,os,math,time,datetime as dt
R=Path(__file__).resolve().parents[2];S=R/'experiments/sandbox';O=R.parent/'overseer-codex';D=O/'card091-full';B=O/'card032-density'
R090=R.parent/'card090-code';TD=R.parent/'card090-train'
ARMS=['ranked43','uniform43','ranked44','uniform44'];LOW=19344847;HIGH=103816750;CHUNK=32768;HEAD_BATCH=256
RUNTIME090='f91697b587a02111234a2caf1bc698c692c6577bd308f6448d75a5d1f1ded2a8'
PARENT='cde4a26b385eea214270aca80633e49a54a9d90efa25bf282562b52f016bd6b9'
END=dt.datetime.fromisoformat('2026-09-13T23:50:55+00:00').timestamp()
LIMITS={'card_gpu_s':3600,'one_cpu_reserve_s':1200,'gpu_tree_rss_gib':32,'cpu_rss_gib':16,'aggregate_rss_gib':48,'global_vram_gib':30,'free_disk_gib':16,'deadline':'2026-09-13T23:50:55Z'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def source_check():
 p=R/'card091-runtime-sha.json';m=read(p)
 assert all(sha(R/n)==h for n,h in m.items()),'Card091 runtime changed'
 return sha(p)
def require_release():
 p=O/'card091-release.json';assert p.exists(),'NO ROOT GPU RELEASE';r=read(p)
 assert r.get('PASS') is True and r.get('card')=='091' and r['limits']==LIMITS,'root release contract mismatch'
 assert os.environ.get('CARD091_RELEASE_SHA')==sha(p),'root release environment mismatch'
 assert r['runtime_sha']==source_check(),'release runtime mismatch'
 assert all(sha(p)==h for p,h in r['files'].items()),'release input changed'
 assert r['selection_sha']==sha(D/'selection.json'),'release selection mismatch'
 return r

def validate_completed(ex,arms):
 assert ex.get('status')=='TRAINED_VALIDATED' and ex.get('runtime_sha')==RUNTIME090,'four090 incomplete or wrong runtime'
 assert set(arms)==set(ARMS),'missing or extra090 arm'
 for arm in ARMS:
  v,m,a,p=arms[arm]
  assert v.get('PASS') is True and v.get('arm')==arm and v.get('runtime_sha')==RUNTIME090,'canonical090 validation mismatch'
  assert a['card']=='090' and a['arm']==arm and a['seed']==int(arm[-2:]) and a['dose']==60000 and a['lr']==.0001,'090 dose/seed mismatch'
  assert a['parent_sha']==a['original_init_sha']==PARENT,'090 parent mismatch'
  assert a['runtime_manifest_sha']==RUNTIME090,'090 identity runtime mismatch'
  assert a['rankneg_window']==(500000 if arm.startswith('ranked') else 0),'090 policy mismatch'
  assert m['identity']==v['identity']==a and m['model_sha']==v['model_sha'],'090 manifest/validation mismatch'
  assert m['dose']==60000 and m['train_stats']['positive_lr_optimizer_steps']==60000,'090 endpoint incomplete'
  assert p['READY'] is True and p['runtime_sha']==RUNTIME090 and p['parent_sha']==PARENT,'090 preparation mismatch'
 return True

def validate_selection(s):
 assert list(s['models'])==ARMS and s['full_rows']==HIGH,'selection full four-head contract'
 assert s['head_batch']==HEAD_BATCH and s['chunk']==CHUNK and s['tiers']==[250000,1000000,4000000,16000000,103815850],'selection precision/row contract'
 assert s['precision']=='FP32 normalized raw full1536; TF32off','selection precision mismatch'
 old=read(O/'card078-full/selection.json')
 for k in ['numeric_tolerance','raw_file_fingerprints','source_manifest_hashes','card032_selection_sha','card032_progress_sha','card032_execution_sha','card032_truth','source_counts']:
  assert s[k]==old[k],'selection inherited input/tolerance mismatch: '+k
 assert not any(k.startswith('reused_') for k in s),'historical coordinate reuse forbidden'
 assert s['runtime']==str(R) and s['runtime_manifest_sha']==source_check(),'selection runtime mismatch'
 assert s['protocol_sha']==sha(O/'card091-full-reference-seeds.md') and s['quality_sha']==sha(O/'card091-quality-prereg.md'),'selection protocol mismatch'
 assert s['baseline_runtime_sha']==RUNTIME090,'090 runtime mismatch'
 for a in ARMS:
  assert s['models'][a]['path']==str(TD/a/'model.pt'),'090 canonical path mismatch'
  assert s['models'][a]['sha']==s['endpoint_bindings'][str(TD/a/'model.pt')]==s['canonical090_validation'][a]['model_sha'],'090 selected model binding mismatch'
 return s

def verify_sources(s):
 for p,x in s['raw_file_fingerprints'].items():
  st=Path(p).stat();assert {k:getattr(st,k) for k in x}==x,'raw fingerprint changed'
 for p,h in s['source_manifest_hashes'].items():assert sha(p)==h,'source manifest changed'
 for name,key in [('selection.json','card032_selection_sha'),('progress.json','card032_progress_sha'),('execution.json','card032_execution_sha')]:assert sha(B/name)==s[key],'032 contract changed'
 for n,h in s['card032_truth'].items():assert sha(B/n)==h,'032 truth changed'
 for n,h in read(B/'selection.json')['files'].items():assert sha(B/n)==h,'032 query/reference binding changed'
 arm_metadata={a:tuple(read(TD/a/n) for n in ['validation.json','manifest.json','admission.json','preparation.json']) for a in ARMS}
 validate_completed(read(O/'card090-execution.json'),arm_metadata)
 for p,h in s['endpoint_bindings'].items():assert sha(p)==h,'090 endpoint binding changed'
 assert sha(R090/'card090-runtime-sha.json')==RUNTIME090,'original090 runtime identity changed'
 assert all(sha(R090/n)==h for n,h in read(R090/'card090-runtime-sha.json').items()),'original090 source changed'
 for a,names in s['retained_checkpoint_files'].items():assert sorted(str(p) for p in (TD/a/'ckpts').glob('*.pt'))==names,'090 retained checkpoint set changed'

def cpu090_bound(state,now):
 # State is assembled from actual unit and durable root launch/release/terminal receipts.
 active=state['active'];launch=state.get('launch');release=state.get('release');terminal=state.get('terminal')
 if active:assert state.get('memory_max_bytes',float('inf'))<=16*2**30 and state.get('memory_swap_max_bytes')==0,'090 CPU memory/swap bound missing'
 if terminal:
  assert release and launch and launch.get('status')=='CPU_STARTED','090 terminal without bound launch'
  assert state.get('terminal_fresh') is True,'090 stale terminal'
 if release:
  assert release.get('PASS') is True and release.get('runtime_sha')==RUNTIME090,'090 CPU runtime mismatch'
  assert state.get('launch_release_match') is True,'090 CPU launch release mismatch'
 if terminal and terminal.get('status') in ('SCORED_AUDITED','CPU_FAILED') and not active:return 0.
 if launch and launch.get('status')!='CPU_STARTED':raise AssertionError('090 CPU launch failure/stuck state requires root')
 if release:
  assert launch and launch.get('status')=='CPU_STARTED','090 CPU freeze without launch requires root'
  assert active,'090 CPU launched but inactive without terminal requires root'
  end=dt.datetime.fromisoformat(release['cpu_absolute_deadline']).timestamp();started=dt.datetime.fromisoformat(release['cpu_budget_started_at']).timestamp()
  assert 0<end-started<=1800 and end<=END,'090 CPU invalid deadline'
  assert end>now,'090 CPU deadline expired requires root'
  return end-now
 assert not launch and not active,'090 CPU unbound active/launch state requires root'
 return 1800.

def joint_checks(estimate,spent,window,cpu_remaining,now,free_disk):
 assert all(math.isfinite(v) and v>=0 for v in [spent,window,cpu_remaining,now,free_disk]) and math.isfinite(estimate) and estimate>0,'invalid admission measurement'
 return {'card_cap':spent+estimate<=3600,'window_cap':window+estimate<=165491,
         'joint_deadline':now+max(estimate,cpu_remaining)+1200<=END,'disk':free_disk>=16*2**30}

def receipt(path,started_ns,runtime,selection,release):
 p=Path(path);assert p.is_file(),'missing stage receipt';assert p.stat().st_mtime_ns>=started_ns,'stale stage receipt';r=read(p)
 assert r.get('PASS') is True,'stage receipt lacks PASS'
 assert r.get('runtime_sha')==runtime,'stage runtime mismatch'
 assert r.get('selection_sha')==selection,'stage selection mismatch'
 assert r.get('release_sha')==release,'stage release mismatch'
 return r

def settled_equivalent(ledger_spent,active_reservation,elapsed):
 assert all(math.isfinite(v) and v>=0 for v in [ledger_spent,active_reservation,elapsed]),'invalid active reservation accounting'
 assert ledger_spent+1e-6>=active_reservation,'reservation absent from ledger'
 return max(0.,ledger_spent-active_reservation)+elapsed

def validate_loader_output(ids,output,state,expected_ids,expected_output,expected_state):
 import numpy as np
 assert np.array_equal(ids,expected_ids),'original090 loader query order mismatch'
 assert output.dtype==expected_output.dtype==np.dtype('f4'),'original090 loader output precision mismatch'
 assert output.shape==expected_output.shape==(len(ids),3),'original090 loader output shape mismatch'
 assert np.array_equal(output,expected_output),'original090 loader output mismatch'
 assert state==expected_state,'original090 loader state mismatch'
