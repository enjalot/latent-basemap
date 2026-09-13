"""CPU contracts for canary-only guarded admission caching; no model/GPU."""
import tempfile,types,os,json,hashlib
from pathlib import Path
from card088_admission_cache import guarded_canary_admission
checks={}
for case in ['valid','bound_change','release_change','env_change','returned_object_mutation','exception']:
 with tempfile.TemporaryDirectory() as td:
  O=Path(td);R=O/'runtime';R.mkdir();f=R/'source.py';f.write_text('first');(R/'card088-runtime-sha.json').write_text(json.dumps({'source.py':'dummy'}));rp=O/'card088-release.json';rp.write_text('release');n=[0]
  sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  rel={'files':{str(f):sha(f)}}
  def original():
   n[0]+=1
   assert sha(f)==rel['files'][str(f)],'full hash changed'
   return rel
  C=types.SimpleNamespace(O=O,R=R,require_release=original,sha=sha,read=lambda p:json.loads(Path(p).read_text()),source_check=lambda:None);os.environ['CARD088_RELEASE_SHA']=sha(rp);failed=False
  try:
   with guarded_canary_admission(C) as report:
    result=C.require_release()
    if case=='bound_change':f.write_text('other')
    if case=='release_change':rp.write_text('changed')
    if case=='env_change':os.environ['CARD088_RELEASE_SHA']='bad'
    if case=='returned_object_mutation':result['files'].clear()
    if case=='exception':raise ValueError('body failure')
    C.require_release()
  except (AssertionError,ValueError):failed=True
  checks[case+'_outcome']=failed==(case in ['bound_change','release_change','env_change','exception'])
  checks[case+'_restore']=C.require_release is original
  if not failed:checks[case+'_full_prepost']=n[0]==2 and report['postcheck_PASS'] and report['calls']==2
assert all(checks.values()),checks
O=Path('/data/latent-basemap/sandbox/overseer-codex');(O/'card088-admission-cache-cpu.json').write_text(json.dumps({'PASS':True,'checks':checks,'scope':'Mocked small bound files; full pre/post calls, per-call mutations and exception restoration. NoGPU.'},indent=2)+'\n');print('PASS',len(checks))
