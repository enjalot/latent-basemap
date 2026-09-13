"""Canary-only immutable admission; production checks remain unchanged."""
from contextlib import contextmanager
from pathlib import Path
import copy,os,time
import card089_common as C

def metadata(path):
 p=Path(path);a=p.lstat();b=p.stat()
 return (str(p.resolve()),a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns,b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns)

@contextmanager
def admitted_canary(record):
 original_require=C.require_release;original_source=C.source_check;original_sha=C.sha
 paths=set();pre_hashes={};pid=os.getpid();start=time.monotonic();env=os.environ.get('CARD089_RELEASE_SHA')
 def capture(p):
  name=str(Path(p).absolute());paths.add(name);value=original_sha(p);pre_hashes[name]=value;return value
 C.sha=capture
 try:
  release=original_require();runtime=original_source();C.input_check()
 finally:C.sha=original_sha
 fingerprints={p:metadata(p) for p in sorted(paths)}
 record.update(pre_full_fingerprint_PASS=True,full_pre_files=pre_hashes,paths=len(paths),runtime_sha=runtime,release_sha=original_sha(C.O/'card089-release.json'),calls=0,scope='Only this process, maximum300s; fixture/checkpoint identity hashes remain actual on every fit')
 def guard():
  assert os.getpid()==pid,'canary admission PID changed'
  assert time.monotonic()-start<300,'canary admission expired'
  assert os.environ.get('CARD089_RELEASE_SHA')==env,'canary release environment changed'
  assert all(metadata(p)==v for p,v in fingerprints.items()),'canary admission metadata changed'
  record['calls']+=1
 def cached_require():guard();return copy.deepcopy(release)
 def cached_source():guard();return runtime
 C.require_release=cached_require;C.source_check=cached_source
 try:yield
 finally:
  C.require_release=original_require;C.source_check=original_source
  # Full fingerprints are mandatory even when a device/control assertion fails.
  post_hashes={}
  def post_sha(p):
   value=original_sha(p);post_hashes[str(Path(p).absolute())]=value;return value
  C.sha=post_sha
  try:
   post=original_require();assert post==release,'canary release changed'
   assert original_source()==runtime,'canary runtime changed';C.input_check()
  finally:C.sha=original_sha
  assert post_hashes==pre_hashes,'canary full fingerprints differ'
  assert all(metadata(p)==v for p,v in fingerprints.items()),'canary post metadata changed'
  guard()
  record.update(post_full_fingerprint_PASS=True,full_post_files=post_hashes,wall_s=time.monotonic()-start)
