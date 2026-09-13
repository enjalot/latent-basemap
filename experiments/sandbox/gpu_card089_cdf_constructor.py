"""First released device stage: diagnose actual full-support raw CUDA CDF only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import time,torch
import card089_common as C
from card089_constructor_diagnostic import inspect_arm

def main():
 start=time.monotonic();release=C.require_release();assert release.get('cdf_lost_probability_cap')==1e-10
 assert torch.cuda.is_available();torch.set_num_threads(2)
 folder=C.O/f'card089-cdf-constructor-{time.time_ns()}'
 results={arm:inspect_arm(arm,'cuda',folder) for arm in C.ARMS}
 receipt={'PASS':all(v['PASS'] for v in results.values()),'arms':results,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.GD/'manifest.json'),'scope':'Two actual constructors per arm on full30M stored weights/endpoints, zero stand-in dataset; no encoder features, iter, draws, model, optimizer or production. Raw CUDA normalization unchanged.'}
 C.write(C.O/'card089-cdf-constructor-device.json',receipt)
 assert receipt['PASS'],'raw CUDA CDF diagnostic STOP; root review required, no automatic repair'
if __name__=='__main__':main()
