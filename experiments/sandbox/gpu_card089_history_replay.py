"""Fail closed on absent/unmatched086 all_one; no replacement allocation."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import tempfile,subprocess,sys,time
import torch
import card089_common as C
from card089_baseline import bound_baseline
from gpu_card060_canary import same
KEYS=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen','mn_gen','mn_rng','dens_gen','dens_rng','hold_gen','hold_rng','deriv_gen','global_step','epoch','train_stats']
def main():
 own_release=C.require_release();bound_base,bound_epoch,bound_files=bound_baseline(own_release);deep=C.read(C.O/'card089-baseline-deep-validation.json');assert deep['PASS'] and deep['baseline_files']==bound_files and deep['runtime_sha']==C.source_check(),'baseline deep validation mismatch'
 start=time.monotonic();base=C.R.parent/'card086-observer-train/all_one';assert (base/'validation.json').exists(),'baseline incomplete STOP';v=C.read(base/'validation.json');assert v['PASS'] and v['model_sha']==C.sha(base/'model.pt'),'baseline validation STOP'
 release=C.read(C.O/'card086-release.json');assert all(C.sha(p)==h for p,h in release['files'].items()),'baseline release changed';checks={};evidence={}
 with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card089-history-') as td:
  for mode in ['fresh','epoch']:
   states=[]
   for runtime in ['original','proposed']:
    dest=Path(td)/(mode+'-'+runtime);subprocess.run([sys.executable,str(Path(__file__).with_name('gpu_card089_history_worker.py')),runtime,mode,str(dest)],check=True);receipt=C.read(dest/'receipt.json');assert receipt['PASS'] and receipt['baseline_files']==bound_files;states.append(torch.load(receipt['endpoint'],map_location='cpu',weights_only=False));evidence[dest.name]=receipt
   states[1]['train_stats'].pop('card089_exposure',None)
   for k in KEYS:checks[mode+'-'+k]=same(states[0][k],states[1][k]);assert checks[mode+'-'+k],'historical compatibility STOP: '+mode+'-'+k
 C.write(C.O/'card089-history-device.json',{'PASS':True,'checks':checks,'evidence':evidence,'baseline_files':bound_files,'baseline_deep_validation_sha':C.sha(C.O/'card089-baseline-deep-validation.json'),'baseline_validation_sha':C.sha(base/'validation.json'),'baseline_model_sha':C.sha(base/'model.pt'),'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.GD/'manifest.json'),'scope':'Full2M actual all_one path separate-source processes,fresh8 and retained-epoch+8. Only new089 exposure metadata removed for fresh-state comparison; no other numeric state omission. New089 instrumented MID/EPOCH twins separate.'})
if __name__=='__main__':main()
