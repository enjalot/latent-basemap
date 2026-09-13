import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,time
import torch
import card065_common as C
import card065_prepare as P
arm=sys.argv[1];assert arm in C.ARMS;C.source_check();C.input_check();start=time.monotonic();P.setup();data=P.load_data();dest=C.TD/arm;dest.mkdir(parents=True,exist_ok=True)
if (dest/'preparation.json').exists():
 r=C.read(dest/'preparation.json');assert r['prepared_sha']==C.sha(dest/'prepared.pt') and r['READY'] and r['identity']==P.prep_identity(arm,0 if arm=='random' else C.PREP_STEPS);raise SystemExit(0)
if arm=='random':
 ident=P.prep_identity(arm,0);C.write(dest/'prep-admission.json',ident);m=P.fresh();r={'step':0,'identity':ident}
else:
 paths=list(dest.glob('prep-step*.pt'));resume=max(paths,key=lambda p:int(p.stem.split('step')[1])) if paths else None
 m,r,_=P.run(arm,C.PREP_STEPS,dest,data,resume=resume,save_steps=(500,1000,1500,2000))
r=P.finish(arm,m,r,data,dest);r['wall_s']=time.monotonic()-start;C.write(dest/'preparation.json',r);C.source_check();print(arm,r['READY'],r['check_PCA_R2'],flush=True)
if not r['READY']:raise SystemExit(3)
