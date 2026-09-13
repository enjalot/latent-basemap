"""Actual full300K checkpointed fit costs; no dose reduction on admission miss."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc,math,datetime as dt
import torch
import card065_common as C
from card065_fit import fit
END=dt.datetime.fromisoformat('2026-09-13T23:50:55+00:00').timestamp()
def main():
 start=time.monotonic();C.source_check();C.input_check();fits={};est={};checks={}
 with tempfile.TemporaryDirectory(dir=str(C.R.parent),prefix='card065-preflight-') as td:
  for arm in C.ARMS:
   fits[arm]={}
   for steps in [500,3500]:
    before=time.monotonic();p,ck,r=fit(arm,steps,Path(td)/f'{arm}-{steps}',checkpoints=[500,steps] if steps>500 else [steps]);torch.cuda.synchronize()
    epochs=list((Path(td)/f'{arm}-{steps}'/'ckpts').glob('ckpt-epoch*.pt'))
    if steps==3500:assert epochs,'long timing missed real epoch checkpoint'
    fits[arm][str(steps)]={'seconds':time.monotonic()-before,'positive_updates':r['train_stats']['positive_lr_optimizer_steps'],'global_vram_GiB':r['global_vram_GiB'],'epoch_checkpoints':len(epochs)}
    del p,ck;gc.collect();torch.cuda.empty_cache()
   w1=fits[arm]['500']['seconds'];w2=fits[arm]['3500']['seconds'];assert math.isfinite(w1+w2) and w2>w1>0
   slope=max((w2-w1)/3000,w2/3500);setup=max(0.,w1-500*slope);epochs=math.ceil(C.DOSE/math.ceil(C.N*15/int(C.BATCH*.1)));reserve=3*epochs+6*len(C.SNAPS)+60
   est[arm]={'per_step_s':slope,'setup_s':setup,'reserve_s':reserve,'complete_arm_s':setup+slope*C.DOSE+reserve}
   checks[arm+'_time']=est[arm]['complete_arm_s']<=1500;checks[arm+'_vram']=max(x['global_vram_GiB'] for x in fits[arm].values())<30
 spent=C.read(C.O/'card065-ledger.json')['batch_spent_s'];win=C.read(C.O/'cards-24h-window-ledger.json')['spent_s'];used=time.monotonic()-start;need=sum(v['complete_arm_s'] for v in est.values())+120
 checks.update(card_cap=spent+used+need<=5400,window_cap=win+used+need<=165491,deadline=need<=END-time.time())
 r={'PASS':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'fits':fits,'estimates':est,'fits_plus_reserve_s':need,'preflight_elapsed_s':used,'runtime_sha':C.source_check(),'scope':'Actual1536-D300K two-arm500/3500 successful-update fits with real epoch checkpoint, conservative repeated serialization, unchanged60K doses.'};C.write(C.O/'card065-preflight.json',r);print(r,flush=True);return 0 if r['PASS'] else 3
if __name__=='__main__':raise SystemExit(main())
