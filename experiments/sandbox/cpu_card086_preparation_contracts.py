"""CPU actual preparation payload validation; no real preparation or model change."""
import tempfile,copy
from pathlib import Path
from unittest.mock import patch
import torch
import card086_common as C
checks={}
with tempfile.TemporaryDirectory() as td:
 root=Path(td);parent=root/'parent.pt';base={'a':torch.tensor([1.,2.]),'b':torch.tensor([[3.]])};torch.save({'model_state_dict':base},parent)
 with patch.multiple(C,TD=root,CHAMP=parent,source_check=lambda:'runtime'):
  for arm in C.ARMS:
   dest=root/arm;dest.mkdir();p=dest/'prepared.pt';sd=copy.deepcopy(base);state=C.state_sha(sd);torch.save({'READY':True,'arm':arm,'model_state':sd,'prepared_state_sha':state},p)
   C.write(dest/'preparation.json',{'READY':True,'arm':arm,'runtime_sha':'runtime','parent_sha':C.sha(parent),'parent_path':str(parent),'prepared_sha':C.sha(p),'prepared_state_sha':state,'full_parent_tensors_bit_identical':True})
  assert set(C.validate_preparations())==set(C.ARMS);checks['actual_valid_both_arms']=True
  arm=C.ARMS[0];path=root/arm/'preparation.json';good=C.read(path)
  for key,value,msg in [('runtime_sha','wrong','preparation runtime/arm mismatch'),('arm','wrong','preparation runtime/arm mismatch'),('parent_sha','wrong','preparation parent mismatch'),('prepared_sha','wrong','preparation file mismatch'),('prepared_state_sha','wrong','preparation state mismatch')]:
   C.write(path,dict(good,**{key:value}))
   try:C.validate_preparations()
   except AssertionError as e:assert str(e)==msg;checks[key+'_rejected']=True
   else:raise AssertionError('bad preparation accepted')
   C.write(path,good)
  p=root/arm/'prepared.pt';saved=torch.load(p,weights_only=False);saved['model_state']['a'][0]=99.;saved['prepared_state_sha']=C.state_sha(saved['model_state']);torch.save(saved,p);C.write(path,dict(good,prepared_sha=C.sha(p),prepared_state_sha=saved['prepared_state_sha']))
  try:C.validate_preparations()
  except AssertionError as e:assert str(e)=='preparation parent tensors mismatch';checks['consistent_rehash_wrong_parent_tensors_rejected']=True
  else:raise AssertionError('wrong parent accepted')
C.write(C.O/'card086-cdf-final-delivery/preparation-revision/preparation-contracts.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'scope':'Synthetic CPU tensors through actual production preparation validator; no GPU or actual parent/model mutation.'});print('PREPARATION PASS',len(checks))
