import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,resource,time
import torch
import card085_common as C
from card085_fit import fit,validate_arm
C.require_release();arm=sys.argv[1];assert arm in C.ARMS;C.source_check();C.input_check();dest=C.TD/arm
if (dest/'manifest.json').exists():C.write(dest/'validation.json',validate_arm(arm));print('ALREADY_VALIDATED');raise SystemExit(0)
resume=None
if (dest/'ckpts').exists():
 candidates=[]
 for path in (dest/'ckpts').glob('*.pt'):
  ck=torch.load(path,map_location='cpu',weights_only=False);step=C.validate_ckpt(ck,C.identity(arm));candidates.append((step,path))
 if candidates:resume=max(candidates,key=lambda v:v[0])[1]
p,ck,r=fit(arm,C.DOSES[arm],dest,checkpoints=C.SNAPS[arm],resume=resume);p.save(str(dest/'model.pt'));r.update(model_sha=C.sha(dest/'model.pt'),max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,trainer_sha=C.sha(Path(__file__)));assert r['max_rss_MiB']<32768;C.source_check();C.write(dest/'manifest.json',r);C.write(dest/'validation.json',validate_arm(arm));print(arm,'TRAINED_VALIDATED',r['wall_s'],flush=True)
