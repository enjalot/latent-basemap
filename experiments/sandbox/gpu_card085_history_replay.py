"""Fail closed on full-support historical replay mismatch; no substitute control."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import subprocess,tempfile,time,sys
from pathlib import Path
import torch
import card085_common as C
from gpu_card060_canary import same
KEYS=['model','optimizer','scheduler','scaler','torch_rng','cuda_rng','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','replay_gen','mn_gen','mn_rng','dens_gen','dens_rng','hold_gen','hold_rng','deriv_gen','global_step','epoch','train_stats']
def main():
 C.require_release();start=time.monotonic();checks={};evidence={}
 with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card085-history-replay-') as td:
  for card in ['023','075']:
   for mode in ['fresh','epoch']:
    outcomes=[]
    for runtime in ['original','proposed']:
     dest=Path(td)/(card+'-'+mode+'-'+runtime)
     subprocess.run([sys.executable,str(Path(__file__).with_name('gpu_card085_history_worker.py')),card,runtime,mode,str(dest)],check=True)
     receipt=C.read(dest/'receipt.json');assert receipt['PASS'];outcomes.append(torch.load(receipt['endpoint'],map_location='cpu',weights_only=False));evidence[dest.name]=receipt
    for key in KEYS:
     name=card+'-'+mode+'-'+key;checks[name]=same(outcomes[0][key],outcomes[1][key]);assert checks[name], 'historical compatibility STOP: '+name
    del outcomes
 C.write(C.O/'card085-history-device.json',{'PASS':True,'checks':checks,'evidence':evidence,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'history_cpu_sha':C.sha(C.O/'card085-history-compatibility.json'),'scope':'Full2M support; original and proposed source in separate processes; fresh8 steps and genuine historical epoch1 resume+8; complete numerical state exact. No historical controls retrained.'})
if __name__=='__main__':main()
