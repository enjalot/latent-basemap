"""Full2M weighted fits, actual CDF and repeated epoch-construction cost, no dose truncation."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,tempfile,time,gc,math
from pathlib import Path
import torch
import card086_common as C
import card086_budget as B
from card086_fit import fit
from card086_sampler_probe import probe

def main():
 C.require_release();arm=sys.argv[1];assert arm in C.ARMS;start=time.monotonic();C.input_check();fits={}
 with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card086-preflight-') as td:
  for dose in [500,3500]:
   record={};before=time.monotonic()
   p,ck,r=fit(arm,dose,Path(td)/str(dose),checkpoints=[dose],probe_record=record)
   stamp=time.monotonic();torch.save(ck,Path(td)/'serialize.pt');serial=time.monotonic()-stamp
   fits[str(dose)]={'seconds':time.monotonic()-before,'serialization_s':serial,'positive_updates':r['train_stats']['positive_lr_optimizer_steps'],'global_vram_GiB':r['global_vram_GiB'],'sampler':record,'cdf_proof':r['cdf_proof']}
   del p,ck;gc.collect();torch.cuda.empty_cache()
 w1=fits['500']['seconds'];w2=fits['3500']['seconds'];slope=max((w2-w1)/3000,w2/3500);assert math.isfinite(slope) and slope>0;setup=max(0,w1-500*slope)
 epoch_cost=max(t for v in fits.values() for t in v['sampler']['epoch_construction_s']);n_epochs=math.ceil(C.DOSE/math.ceil(C.N*15/int(C.BATCH*.1)));reserve=n_epochs*epoch_cost+2*max(v['serialization_s'] for v in fits.values())*(n_epochs+len(C.SNAPS))+90
 estimate=setup+slope*C.DOSE+reserve
 release=C.require_release();assert release.get('cdf_lost_probability_cap')==1e-10,'root CDF lost-mass acceptance must be 1e-10'
 checks={'stage_cap':estimate+C.read(B.L)['arm_spent_s'][arm]<=B.LIMITS['stage_gpu_s'][arm],'vram':all(v['global_vram_GiB']<30 for v in fits.values()),'cdf_lost_mass':all(v['sampler']['cdf']['positive_zero_width_probability']<=release['cdf_lost_probability_cap'] for v in fits.values()),'successful_doses':all(v['positive_updates']==int(k) for k,v in fits.items())}
 out={'PASS':all(checks.values()),'checks':checks,'estimate':{'complete_arm_s':estimate,'per_step_s':slope,'setup_s':setup,'reserve_s':reserve,'epoch_construction_s_max':epoch_cost,'reserved_epoch_constructions':n_epochs},'fits':fits,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.GD/'manifest.json')};C.write(C.O/f'card086-preflight-{arm}.json',out);assert out['PASS'],'weighted full-dose preflight STOP'
if __name__=='__main__':main()
