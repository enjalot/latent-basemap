"""Re-run original086 canonical deep validation without changing its files."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import sys,time
import torch
import card089_common as C
from card089_baseline import bound_baseline

def main():
 release=C.require_release();start=time.monotonic();base,epoch,bindings=bound_baseline(release);root=C.R.parent/'card086-code';sys.path[:0]=[str(root/'experiments/sandbox'),str(root)]
 assert C.sha(root/'card086-runtime-sha.json')==C.read(C.O/'card089-cpu-contracts.json')['baseline_runtime_sha'],'baseline runtime changed since CPU review'
 import card086_common as H
 from card086_fit import validate_arm
 import basemap.pumap.parametric_umap.core as core
 assert str(core.__file__).startswith(str(root)+'/'),'canonical validation import escaped'
 validated=validate_arm('all_one');assert validated['PASS'],'canonical deep validation failed'
 ck=torch.load(epoch,map_location='cpu',weights_only=False);assert ck['step_checkpoint'] is False and ck['global_step']<60000,'bound baseline not retained epoch';H.validate_ckpt(ck,H.identity('all_one'))
 assert C.read(base/'admission.json')==validated['identity'],'baseline admission identity mismatch';assert all(C.sha(p)==h for p,h in bindings.items()),'baseline changed during deep validation'
 C.write(C.O/'card089-baseline-deep-validation.json',{'PASS':True,'canonical_result':validated,'baseline_files':bindings,'bound_epoch':str(epoch),'bound_epoch_step':ck['global_step'],'canonical_runtime_sha':H.source_check(),'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.GD/'manifest.json'),'wall_s':time.monotonic()-start})
if __name__=='__main__':main()
