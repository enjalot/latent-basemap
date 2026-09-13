"""Replay released observer failure and prove cleanup-only count correction on CPU."""
import importlib.util
from pathlib import Path
import torch
import card088_common as C
import cpu_card088_exposure as T
from gpu_card060_canary import same
checks={};oldpath=C.R.parent/'card088-code/experiments/sandbox/card088_exposure.py'
spec=importlib.util.spec_from_file_location('released_card088_observer',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
current=T.observe;T.observe=old.observe
try:
 try:T.run()
 except AssertionError as e:assert str(e)=='exposure loop-entry mismatch';checks['released_observer_reproduced_first_cleanup_at_CPU_attempt5']=True
 else:raise AssertionError('released bug not reproduced')
finally:T.observe=current
for tag in ['reset','unselected_increment']:
 try:T.run(counter_fault=tag)
 except AssertionError as e:assert str(e)=='exposure loop-entry mismatch';checks[tag+'_still_rejected']=True
 else:raise AssertionError('counter defect accepted')
full=T.run(fault_batches=(1,4));plain=T.run(False,fault_batches=(1,4));e=T.validate(full[0]._train_stats)
assert e['attempted_batches']==10 and e['successful_batches']==6 and e['skipped_batches']==4 and e['skipped_short_tail_batches']==2;checks['normal_and_tail_double_clear_skips_count_once']=True
assert same(full[1].state_dict(),plain[1].state_dict()) and same(full[0].model.state_dict(),plain[0].model.state_dict()) and same(full[2].state_dict(),plain[2].state_dict()) and torch.equal(full[3].gen.get_state(),plain[3].gen.get_state());checks['normal_tail_cleanup_model_Adam_scaler_RNG_parity']=True
for split in [2,5]:
 r=T.run(split=split,fault_batches=(1,4));assert r[0]._train_stats==full[0]._train_stats and same(r[1].state_dict(),full[1].state_dict()) and torch.equal(r[3].gen.get_state(),full[3].gen.get_state());checks['resume_after_cleanup_'+str(split)]=True
C.write(C.O/'card088-exposure-repair/cleanup-regression.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'released_observer_sha':C.sha(oldpath),'scope':'Actual CPU sampler/Adam/GradScaler with core-order unscale,clip,cleanup-zero,scaler-update,skip-counter; released observer replay fails on first injected cleanup. Exact production GPU step not inferred from this fixture.'});print('CLEANUP REGRESSION PASS',len(checks))
