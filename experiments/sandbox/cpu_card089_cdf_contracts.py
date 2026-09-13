"""Actual persistence helper faults; no GPU and no frozen-receipt mutation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import tempfile,json
import torch
import card089_common as C
from card089_cdf_diagnostics import persist_and_check

def main():
 checks={}
 with tempfile.TemporaryDirectory() as td:
  for name,values,failure in [('valid',[.25,.5,1.],None),('decreasing',[.5,.25,1.],'nondecreasing'),('terminal',[.25,.5,1.00000000001],'terminal_1e12'),('nonfinite',[.25,float('nan'),1.],'finite')]:
   p=Path(td)/(name+'.json')
   try:persist_and_check(torch.tensor(values,dtype=torch.float64),torch.ones(3,dtype=torch.float64),p,{'fixture':name})
   except AssertionError as e:assert failure and str(e)=='card089 CDF '+failure+' failed'
   else:assert failure is None
   assert p.exists();r=json.loads(p.read_text());assert r['fixture']==name
   if failure:assert r['checks'][failure] is False
   if name=='decreasing':assert r['min_interval']==-.25 and r['decreasing_interval_count']==1 and r['decreasing_interval_input_mass']==1
   checks[name+'_persisted_before_specific_rejection']=True
 C.write(C.O/'card089-cdf-cpu-contracts.json',{'PASS':True,'checks':checks,'scope':'Same helper called by actual probe; injected decreasing/terminal/nonfiniteCDFs persist measurements before failure,unchanged1e-12 terminal tolerance. NoGPU.'});print('CDF CPU PASS',len(checks))
if __name__=='__main__':main()
