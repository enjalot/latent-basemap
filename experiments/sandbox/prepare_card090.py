"""Released parent clones; verify existing same-runtime preparation, never overwrite it."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import time,torch
import card090_common as C

def main():
 start=time.monotonic();C.require_release();C.source_check();C.input_check();parent_sha=C.sha(C.CHAMP)
 assert parent_sha=='cde4a26b385eea214270aca80633e49a54a9d90efa25bf282562b52f016bd6b9','wrong original023 parent'
 base=torch.load(C.CHAMP,map_location='cpu',weights_only=False)['model_state_dict'];base_sha=C.state_sha(base);receipts={}
 for arm in C.ARMS:
  dest=C.TD/arm;dest.mkdir(parents=True,exist_ok=True);prep=dest/'preparation.json';warm=dest/'prepared.pt'
  if prep.exists():
   r=C.read(prep);assert r['READY'] and r['arm']==arm and r['seed']==C.seed(arm) and r['parent_sha']==parent_sha and r['runtime_sha']==C.source_check(),'existing preparation identity mismatch'
   assert r['prepared_sha']==C.sha(warm),'existing prepared file changed'
   saved=torch.load(warm,map_location='cpu',weights_only=False);sd=saved['model_state'];assert saved['READY'] and saved['arm']==arm and saved['prepared_state_sha']==base_sha,'existing prepared tensor identity mismatch'
  else:
   assert not any(dest.iterdir()),'preserve incomplete/old outputs; root recovery required'
   sd={k:v.clone() for k,v in base.items()}
   torch.save({'READY':True,'arm':arm,'model_state':sd,'prepared_state_sha':base_sha},warm)
   r={'READY':True,'arm':arm,'seed':C.seed(arm),'fresh_optimizer':'no optimizer serialized; legacy fit creates fresh AdamW','parent_sha':parent_sha,'parent_path':str(C.CHAMP),'full_parent_tensors_bit_identical':True,'prepared_state_sha':base_sha,'prepared_sha':C.sha(warm),'runtime_sha':C.source_check()}
   C.write(prep,r)
  assert set(base)==set(sd) and all(torch.equal(base[k],sd[k]) for k in base),'prepared tensors differ from original023'
  assert C.state_sha(sd)==r['prepared_state_sha']==base_sha
  receipts[arm]=r;del sd
 C.write(C.O/'card090-initialization.json',{'PASS':True,'arms':receipts,'wall_s':time.monotonic()-start,'runtime_sha':C.source_check(),'data_manifest_sha':C.sha(C.D/'inputs-manifest.json')});print('MATCHED INITIALIZATION PASS',flush=True)
if __name__=='__main__':main()
