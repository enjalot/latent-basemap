"""Measure actual readout optimizer/checkpoint cost before admitting either full fit."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import torch
import card040_common as V
from card040_engine import Bank,Engine

def main():
 start=time.monotonic();bank=Bank();est={}
 for a in V.ARMS:
  e=Engine(a,bank,2000);torch.cuda.synchronize();begin=time.monotonic()
  while e.step<500:e.advance()
  torch.cuda.synchronize();t500=time.monotonic()-begin
  while e.step<2000:e.advance()
  torch.cuda.synchronize();t2000=time.monotonic()-begin;p=V.DATA/('preflight-'+a+'.pt');before=time.monotonic();e.save(p);ckpt=time.monotonic()-before;p.unlink();per=(t2000-t500)/1500
  est[a]={'per_step_s':per,'setup_and_warm_s':max(0,t500-500*per),'checkpoint_s':ckpt,'fit_bound_s':1.2*(V.DOSE*per+max(0,t500-500*per)+4*ckpt)+20}
  del e
 spent=json.loads((V.O/'card040-ledger.json').read_text())['batch_spent_s'];elapsed=time.monotonic()-start;need=sum(x['fit_bound_s'] for x in est.values())+180
 from datetime import datetime,timezone
 room=(datetime.fromisoformat('2026-09-13T01:52:44+00:00')-datetime.now(timezone.utc)).total_seconds()
 passed=all(x['fit_bound_s']<=600 for x in est.values()) and spent+elapsed+need<=1800 and need<=room
 r={'PASS':bool(passed),'estimates':est,'remaining_bound_s':need,'measured_s':elapsed,'prior_charged_s':spent,'runtime_manifest_sha':V.runtime_check(),'bank_sha':bank.sha,'resources':V.resources(),'no_dose_truncation':True};V.atomic(V.O/'card040-preflight.json',r);print(json.dumps(r));return 0 if passed else 3
if __name__=='__main__':raise SystemExit(main())
