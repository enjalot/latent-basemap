"""Canonical baseline binding contract; no implicit checkpoint selection."""
from pathlib import Path
import card089_common as C

def bound_baseline(release):
 base=C.R.parent/'card086-observer-train/all_one';bindings=release.get('baseline_files');assert isinstance(bindings,dict),'missing baseline file bindings'
 epoch=Path(release.get('baseline_epoch_checkpoint','')).resolve();assert epoch.is_relative_to((base/'ckpts').resolve()) and epoch.name.startswith('ckpt-epoch') and epoch.suffix=='.pt','invalid bound baseline epoch'
 required=[base/'validation.json',base/'admission.json',base/'model.pt',base/'prepared.pt',base/'preparation.json',base/'manifest.json',epoch]
 for path in required:
  assert str(path) in bindings,'missing baseline binding: '+path.name
  assert path.is_file() and C.sha(path)==bindings[str(path)],'baseline binding mismatch: '+path.name
 return base,epoch,{str(p):bindings[str(p)] for p in required}
