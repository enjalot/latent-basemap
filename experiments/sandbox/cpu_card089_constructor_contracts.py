"""Small actual CPU constructors and mocked device entrypoint; never CUDA."""
from pathlib import Path
from unittest.mock import patch
import tempfile,numpy as np,torch
import card089_common as C
import card089_constructor_diagnostic as D
import gpu_card089_cdf_constructor as G
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceEdgeSampler

def main():
 checks={}
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);n=5;src=np.repeat(np.arange(n,dtype='i4'),15);dst=(src+1)%n
  for arm in C.ARMS:np.savez(root/f'{arm}-edges.npz',sources=src,targets=dst,weights=np.ones(n*15,dtype='f4'),n_nodes=n)
  with patch.multiple(C,GD=root,N=n,source_check=lambda:'runtime',sha=lambda p:'bound'),patch.object(DeviceEdgeSampler,'__iter__',side_effect=AssertionError('no iteration')),patch.object(DeviceEdgeSampler,'__next__',side_effect=AssertionError('no draws')):
   result=D.inspect_arm(C.ARMS[0],'cpu',root/'ok');assert result['PASS'] and result['repeats'][1]['repeat_bit_identical'];checks['actual_constructor_no_iter_draw']=True
   original=DeviceEdgeSampler.__init__
   def corrupt(self,*a,**kw):original(self,*a,**kw);self.sample_cdf[0]=.8;self.sample_cdf[1]=.2
   with patch.object(DeviceEdgeSampler,'__init__',corrupt):result=D.inspect_arm(C.ARMS[0],'cpu',root/'bad')
   assert not result['PASS'] and len(result['errors'])==2 and all('nondecreasing' in e for e in result['errors']);checks['actual_corrupt_constructor_persisted_before_stop']=True
 outputs={};calls=[]
 def inspect(arm,device,folder):calls.append((arm,device));return {'PASS':arm!=C.ARMS[0]}
 with patch.multiple(C,require_release=lambda:{'cdf_lost_probability_cap':1e-10},source_check=lambda:'runtime',sha=lambda p:'data',write=lambda p,r:outputs.update(r)),patch.object(G,'inspect_arm',inspect),patch.object(torch.cuda,'is_available',lambda:True):
  try:G.main()
  except AssertionError as e:assert str(e)=='raw CUDA CDF diagnostic STOP; root review required, no automatic repair'
  else:raise AssertionError('failed constructor admitted')
 assert calls==[(a,'cuda') for a in C.ARMS] and outputs['PASS'] is False;checks['mocked_device_entry_persists_aggregate_STOP']=True
 C.write(C.O/'card089-final-readiness/constructor-contracts.json',{'PASS':True,'n_checks':len(checks),'checks':checks,'scope':'Actual tiny CPU constructors with draw/iteration forbidden; mocked CUDA-entry inspect function only, no device execution.'});print('CONSTRUCTOR CONTRACTS PASS',len(checks))
if __name__=='__main__':main()
