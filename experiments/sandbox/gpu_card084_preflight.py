"""Measure full2M production fits; estimate complete60K dose, never truncate."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import tempfile,time,gc,math,sys
import torch
import card084_common as C
from card084_fit import fit

def main(arm):
    C.require_gpu_stage();assert arm in C.ARMS;start=time.monotonic();fits={}
    with tempfile.TemporaryDirectory(dir=C.R.parent,prefix='card084-preflight-') as td:
        for steps in (500,3500):
            before=time.monotonic()
            p,ck,r=fit(arm,steps,Path(td)/str(steps),checkpoints=[500,steps] if steps>500 else [steps])
            torch.cuda.synchronize();serialization=time.monotonic()
            torch.save(ck,Path(td)/'serialization.pt');serial_s=time.monotonic()-serialization
            fits[str(steps)]={'seconds':time.monotonic()-before,'positive_updates':r['train_stats']['positive_lr_optimizer_steps'],
                'global_vram_GiB':r['global_vram_GiB'],'serialization_s':serial_s,
                'hook':r['actual_attraction'],'loaded_modules':r['loaded_modules']}
            del p,ck;gc.collect();torch.cuda.empty_cache()
    a=fits['500']['seconds'];b=fits['3500']['seconds']
    assert math.isfinite(a+b) and b>a>0
    slope=max((b-a)/3000,b/3500);setup=max(0.,a-500*slope)
    epochs=math.ceil(C.DOSE/math.ceil(C.N*15/int(C.BATCH*.1)))
    reserve=2*max(v['serialization_s'] for v in fits.values())*(epochs+len(C.SNAPS))+90
    estimate=setup+slope*C.DOSE+reserve
    assert math.isfinite(estimate) and estimate>0
    C.write(C.O/f'card084-preflight-{arm}.json',{'PASS':True,'arm':arm,'fits':fits,
        'estimate':{'per_step_s':slope,'setup_s':setup,'reserve_s':reserve,'complete_arm_s':estimate},
        'runtime_sha':C.source_check(),'calibration_sha':C.sha(C.CAL),'wall_s':time.monotonic()-start,
        'scope':'Actual full2M500/3500 successful-update fits with enabled hook; full checkpoint serialization measured. Epoch18316 not reached; conservative checkpoint reserve. Chain admits full60K only after cumulative budgets.'})
if __name__=='__main__':main(sys.argv[1])
