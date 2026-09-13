"""Canary-only admission cache: full pre/post hashes, per-call file fingerprints."""
from contextlib import contextmanager
from pathlib import Path
import hashlib,os,time,copy

def signature(path):
 st=Path(path).stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)

@contextmanager
def guarded_canary_admission(C):
 original=C.require_release;started=time.monotonic();release=original();frozen=copy.deepcopy(release)
 release_path=C.O/'card088-release.json';release_sha=C.sha(release_path)
 # Bound files plus every loaded runtime source, not only the manifest pathname.
 paths=set(release['files'])|{str(C.R/n) for n in C.read(C.R/'card088-runtime-sha.json')}|{str(release_path)}
 stamps={n:signature(n) for n in paths};report={'mode':'canary_only_full_pre_post_hashes_per_call_metadata','precheck_s':time.monotonic()-started,'calls':0,'file_count':len(paths),'postcheck_PASS':False}
 def verify_metadata():
  assert os.environ.get('CARD088_RELEASE_SHA')==release_sha,'cached admission release environment drift'
  assert C.sha(release_path)==release_sha,'cached admission release content drift'
  assert release==frozen,'cached admission object drift'
  assert all(signature(p)==v for p,v in stamps.items()),'cached admission bound file changed'
 def cached():
  verify_metadata();report['calls']+=1;return copy.deepcopy(frozen)
 C.require_release=cached
 try:
  yield report
  verify_metadata();t=time.monotonic();post=original();assert post==frozen,'cached admission full postcheck drift';C.source_check();report['postcheck_s']=time.monotonic()-t;report['postcheck_PASS']=True
 finally:C.require_release=original
