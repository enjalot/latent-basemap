import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc,math,sys
import torch
import card090_common as C
from card090_fit import fit

def estimate(fits):
 w1=fits['500']['seconds'];w2=fits['3500']['seconds'];assert math.isfinite(w1+w2) and w2>w1>0,'invalid timing slope'
 slope=max((w2-w1)/3000,w2/3500);setup=max(0.,w1-500*slope);epochs=math.ceil(C.DOSE/math.ceil(C.N*15/int(C.BATCH*.1)));reserve=2*max(x['serialization_s'] for x in fits.values())*(epochs+len(C.SNAPS))+90
 return {'per_step_s':slope,'setup_s':setup,'reserve_s':reserve,'complete_arm_s':setup+slope*C.DOSE+reserve}
def main():
 arm=sys.argv[1];assert arm in ['ranked43','uniform43'];C.require_release();start=time.monotonic();C.input_check();fits={}
 with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card090-preflight-') as td:
  for steps in [500,3500]:
   before=time.monotonic();p,ck,r=fit(arm,steps,Path(td)/str(steps),checkpoints=[500,steps] if steps>500 else [steps]);torch.cuda.synchronize();serial=time.monotonic();torch.save(ck,Path(td)/'serialization-probe.pt');serial=time.monotonic()-serial
   fits[str(steps)]={'seconds':time.monotonic()-before,'positive_updates':r['train_stats']['positive_lr_optimizer_steps'],'attempted_batches':r['train_stats']['attempted_batches'],'amp_skips':r['train_stats']['amp_overflow_skips'],'pipeline':r['pipeline_info'],'serialization_s':serial,'global_vram_GiB':r['global_vram_GiB'],'actual_rank_scale':p._rankneg_scale,'epoch_checkpoints':len(list((Path(td)/str(steps)/'ckpts').glob('ckpt-epoch*.pt')))}
   del p,ck;gc.collect();torch.cuda.empty_cache()
 e=estimate(fits);C.write(C.O/f'card090-preflight-{arm}.json',{'PASS':True,'measurement_only':True,'arm':arm,'fits':fits,'estimate':e,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.D/'inputs-manifest.json'),'scope':'Full2M500/3500 successful updates per policy; checkpoint serialization measured, epoch18316 not reached so repeated epoch serialization reserved. No budget admission until all stage/controller reservations settle; seed44 uses same-policy estimate.'})
if __name__=='__main__':main()
