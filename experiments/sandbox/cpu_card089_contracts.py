"""CPU identity, source reuse, resource and fresh receipt controls; no feature reads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,tempfile,time
import card089_common as C
import card089_budget as B
from run_card089_chain import validate_stage_receipt

def reject(f,msg):
 try:f()
 except AssertionError as e:assert str(e)==msg;return True
 raise AssertionError('negative control accepted '+msg)
def main():
 checks={}
 for key in ['arm','dose','lr','graph_sha','weight_values_sha','endpoints_sha','protocol_sha','quality_prereg_sha','mutual_mask_sha','actual_mutual_mask_values_sha','raw_ratio','row_normalization','favored_policy','parent_sha','warm_sha','reference_sha','data_manifest_sha','runtime_manifest_sha','positive_target_mode','negative_policy']:
  checks['wrong_'+key]=reject(lambda:C.assert_identity({key:'old'},{key:'new'}),'card089 identity mismatch: '+key)
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'r.json';t=time.time_ns();checks['missing']=reject(lambda:validate_stage_receipt(p,t,'r','d'),'missing stage receipt');good={'PASS':True,'runtime_sha':'r','data_manifest_sha':'d'};p.write_text(json.dumps(good));os.utime(p,ns=(t,t));assert validate_stage_receipt(p,t,'r','d')==good;checks['fresh']=True
  os.utime(p,ns=(t-1,t-1));checks['stale']=reject(lambda:validate_stage_receipt(p,t,'r','d'),'stale stage receipt')
  for k,v,msg in [('PASS',False,'stage receipt lacks explicit PASS'),('runtime_sha','bad','stage receipt runtime mismatch'),('data_manifest_sha','bad','stage receipt calibration mismatch')]:
   p.write_text(json.dumps(dict(good,**{k:v})));checks['receipt_'+k]=reject(lambda:validate_stage_receipt(p,0,'r','d'),msg)
 from card089_baseline import bound_baseline
 from unittest.mock import patch
 with tempfile.TemporaryDirectory() as td:
  root=Path(td)/'code';base=Path(td)/'card086-train/all_one';(base/'ckpts').mkdir(parents=True);epoch=base/'ckpts/ckpt-epoch1.pt';paths=[base/n for n in ['validation.json','admission.json','model.pt','prepared.pt']]+[epoch]
  for p in paths:p.write_text('fixture')
  rel={'baseline_files':{str(p):C.sha(p) for p in paths},'baseline_epoch_checkpoint':str(epoch)}
  with patch.object(C,'R',root):
   assert bound_baseline(rel)[1]==epoch;checks['baseline_all_required_bindings']=True
   bad=dict(rel,baseline_files={k:v for k,v in rel['baseline_files'].items() if k!=str(base/'admission.json')});checks['baseline_missing_admission_rejected']=reject(lambda:bound_baseline(bad),'missing baseline binding: admission.json')
   epoch.write_text('changed');checks['baseline_changed_epoch_rejected']=reject(lambda:bound_baseline(rel),'baseline binding mismatch: ckpt-epoch1.pt')
 checks['budget']=B.LIMITS['card_gpu_s']==3600 and B.LIMITS['stage_gpu_s']=={'reciprocal':1600,'rank_count_control':1600} and B.LIMITS['rss_gib']==32
 checks['no_release']=reject(C.require_release,'NO ROOT GPU RELEASE')
 old=C.R.parent/'card086-code';manifest=C.read(old/'card086-runtime-sha.json');baseline={p:h for p,h in manifest.items() if p.startswith('basemap/')};assert all(C.sha(old/p)==h and C.sha(C.R/p)==h for p,h in baseline.items());checks['all_basemap_bytes_match086']=True
 import ast
 def config(path):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='configure'),include_attributes=False)
 checks['configure_AST_matches086']=config(old/'experiments/sandbox/card086_common.py')==config(C.R/'experiments/sandbox/card089_common.py')
 assert all(checks.values());C.write(C.O/'card089-cpu-contracts.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'baseline_source_files':len(baseline),'baseline_runtime_sha':C.sha(old/'card086-runtime-sha.json'),'scope':'Static same-core proof only; historical numerical reuse requires completed086 all_one validation and released full2M separate-process fresh8/retained-epoch+8 exact state. No feature/query/model load.'});print('CONTRACTS PASS',len(checks))
if __name__=='__main__':main()
