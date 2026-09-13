"""Actual preparation function with tiny synthetic parent; no actual parent/data reads."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from unittest.mock import patch
import tempfile,torch
import card090_common as C
import prepare_card090 as P

def main():
 torch.set_num_threads(2);real_sha=C.sha;checks=[]
 with tempfile.TemporaryDirectory() as tmp:
  td=Path(tmp);parent=td/'parent.pt';torch.save({'model_state_dict':{'w':torch.arange(8,dtype=torch.float32)}},parent)
  def sha(p):return 'cde4a26b385eea214270aca80633e49a54a9d90efa25bf282562b52f016bd6b9' if Path(p)==parent else 'input' if Path(p).name=='inputs-manifest.json' else real_sha(p)
  with patch.multiple(C,CHAMP=parent,TD=td/'train',O=td,D=td,sha=sha,require_release=lambda:None,source_check=lambda:'runtime',input_check=lambda:None):
   P.main();before={a:((C.TD/a/'prepared.pt').read_bytes(),(C.TD/a/'prepared.pt').stat().st_mtime_ns) for a in C.ARMS};checks.append('actual preparation clones every small parent tensor')
   marker=C.TD/C.ARMS[0]/'retained-checkpoint-marker';marker.write_text('preserve')
   P.main();assert marker.read_text()=='preserve';assert all(before[a]==((C.TD/a/'prepared.pt').read_bytes(),(C.TD/a/'prepared.pt').stat().st_mtime_ns) for a in C.ARMS);checks.append('idempotent current-runtime preparation preserves existing files')
   p=C.TD/C.ARMS[0]/'preparation.json';r=C.read(p);r['runtime_sha']='old';C.write(p,r)
   try:P.main()
   except AssertionError as e:assert str(e)=='existing preparation identity mismatch'
   else:raise AssertionError('old runtime preparation accepted')
   assert C.read(p)['runtime_sha']=='old' and marker.exists();checks.append('old runtime rejected, evidence preserved')
 C.write(C.O/'card090-readiness/preparation-contracts.json',{'PASS':True,'n_checks':len(checks),'checks':checks,'scope':'Actual prepare.main on tiny synthetic parent only; real parent tensors remain released-stage gate.'});print('PREPARATION PASS',len(checks))
if __name__=='__main__':main()
