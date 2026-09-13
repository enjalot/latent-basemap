"""Bounded CPU profiling of stored weights and actual CPU constructor CDF."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import time,resource,torch
import card089_common as C
from card089_constructor_diagnostic import inspect_arm

def main():
 start=time.monotonic();torch.set_num_threads(2)
 folder=C.O/'card089-final-readiness/cpu-cdf-details'
 results={arm:inspect_arm(arm,'cpu',folder) for arm in C.ARMS}
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert rss<8*2**30,'CPU profile RSS8GiB STOP'
 out={'PASS':all(v['PASS'] for v in results.values()),'arms':results,'wall_s':time.monotonic()-start,'peak_rss_bytes':rss,'threads':2,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.GD/'manifest.json'),'scope':'Actual CPU constructor CDF, unchanged raw weights and normalization. No claim of CUDA equivalence or quality. No feature/query/model access; stored graph/weight reads only.'}
 C.write(C.O/'card089-final-readiness/cpu-cdf-profile.json',out);assert out['PASS'];print('CPU CDF PROFILE PASS',out['wall_s'],rss)
if __name__=='__main__':main()
