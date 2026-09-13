import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,subprocess,sys,time
import torch
import card090_common as C
from card090_state import compare

def main():
 C.require_release();start=time.monotonic();checks={};evidence={}
 with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card090-history-') as td:
  for policy in ['ranked','uniform']:
   for mode in ['fresh','epoch']:
    states=[]
    for runtime in ['original','proposed']:
     dest=Path(td)/(policy+'-'+mode+'-'+runtime);subprocess.run([sys.executable,str(Path(__file__).with_name('gpu_card090_history_worker.py')),policy,runtime,mode,str(dest)],check=True);receipt=C.read(dest/'receipt.json');assert receipt['PASS'];states.append(torch.load(receipt['endpoint'],map_location='cpu',weights_only=False));evidence[dest.name]=receipt
    assert evidence[policy+'-'+mode+'-original']['core_sha']==evidence[policy+'-'+mode+'-proposed']['core_sha'],'core bytes changed'
    checks[policy+'-'+mode]=compare(*states);del states
    print(policy,mode,'historical exact state PASS',flush=True)
 C.write(C.O/'card090-history-device.json',{'PASS':True,'checks':checks,'evidence':evidence,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.D/'inputs-manifest.json'),'scope':'Actual full2M, each original073/075 versus090 configuration path in separate imported-core processes; seed42 fresh8 and actual retained epoch2+8, every listed numerical state field exactly equal; legacy admission identity intentionally retained for historical replay only.'})
if __name__=='__main__':main()
