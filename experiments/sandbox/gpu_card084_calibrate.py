"""Released GPU stage; fixed-head production gradients, no training updates."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import time,datetime as dt
import card084_common as C
from card084_fit import fit
from card084_calibration import FixedHeadCalibration,CalibrationComplete

def main():
    C.require_gpu_stage();C.source_check();C.input_check()
    assert not C.CAL.exists(),'calibration is immutable; root must review failed attempts'
    start=time.monotonic();tag=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    cb=FixedHeadCalibration(C.O/'card084-calibration-batches'/tag)
    try:
        fit('quadratic',C.DOSE,C.R.parent/'card084-calibration-work'/tag,calibration_callback=cb)
    except CalibrationComplete:
        result=cb.result();result['wall_s']=time.monotonic()-start;C.write(C.CAL,result)
        print('CALIBRATION_PASS',result['arms'],flush=True)
        return
    raise RuntimeError('calibration failed to stop at eight fixed-head batches')
if __name__=='__main__':main()
