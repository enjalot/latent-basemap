import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc
import torch
import card065_common as C
import card065_prepare as P
from gpu_card060_canary import same
start=time.monotonic();C.source_check();C.input_check();P.setup();data=P.load_data();checks=[];states={};draws={}
with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card065-prep-canary-') as path:
 td=Path(path)
 for arm in ('pca','shuffled'):
  m,r,gen=P.run(arm,16,td/arm,data,save_steps=(8,16));full=torch.load(td/arm/'prep-step16.pt',map_location='cpu',weights_only=False);states[arm]=r['state_sha'];draws[arm]=r['draw_digest'];del m;gc.collect();torch.cuda.empty_cache()
  m,rr,_=P.run(arm,16,td/(arm+'-resume'),data,resume=td/arm/'prep-step8.pt');again=torch.load(td/(arm+'-resume')/'prep-step16.pt',map_location='cpu',weights_only=False)
  assert same(full,again),'supervised full-state resume mismatch';checks.append(arm+' actual full-state midpoint resume');del m;gc.collect();torch.cuda.empty_cache()
  for key,value in [('arm','wrong'),('target_sha','0'*64),('seed',123),('dose',17)]:
   ck=torch.load(td/arm/'prep-step8.pt',map_location='cpu',weights_only=False);ck['identity'][key]=value;p=td/f'{arm}-{key}.pt';torch.save(ck,p)
   try:P.run(arm,16,td/f'{arm}-reject-{key}',data,resume=p)
   except ValueError as e:assert 'supervised admission-identity mismatch' in str(e);checks.append(arm+' wrong '+key+' rejected')
   else:raise AssertionError('wrong supervised identity accepted')
 assert states['pca']!=states['shuffled'];checks.append('target permutation changes weights');assert draws['pca']==draws['shuffled'];checks.append('matched supervised row draws')
C.write(C.O/'card065-prep-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'endpoints':states,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check()});print('PASS',len(checks),flush=True)
