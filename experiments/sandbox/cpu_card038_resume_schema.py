"""Validate new038 resume reader against a real038-compatible-width core checkpoint from033.
Explicit structural fixture: identity replaced, observer absent. No claim this is a038 training run.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,copy
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import torch,numpy as np
import card038_validate as V
R=Path(__file__).resolve().parents[2];O=V.OC;p=V.SB/'card033-train/ordinary3d/ckpts/ckpt-step100000.pt';ck=torch.load(p,map_location='cpu',weights_only=False);identity={**V.expected_identity('wide2048',R),'scope':'real033checkpoint schema fixture','observe_exposure':False};ck['card012_identity']=identity
V.validate_ckpt_payload(ck,'wide2048',R,identity,expect_step=100000,n_nodes=4000000);checks={'real_core_state_positive':True}
def reject(name,mutate,restore):
 mutate()
 try:V.validate_ckpt_payload(ck,'wide2048',R,identity,expect_step=100000,n_nodes=4000000)
 except (AssertionError,ValueError,RuntimeError):checks[name]=True
 else:checks[name]=False
 finally:restore()
old=ck['card012_identity'];reject('wrong_admission',lambda:ck.__setitem__('card012_identity',{}),lambda:ck.__setitem__('card012_identity',old))
old=ck['scaler'];reject('missing_scaler',lambda:ck.__setitem__('scaler',None),lambda:ck.__setitem__('scaler',old))
old=ck['loader_gen'];reject('missing_device_generator',lambda:ck.__setitem__('loader_gen',None),lambda:ck.__setitem__('loader_gen',old))
old=ck['loader_pos_idx'];reject('bad_cursor',lambda:ck.__setitem__('loader_pos_idx',-1),lambda:ck.__setitem__('loader_pos_idx',old))
pid=next(iter(ck['optimizer']['state']));old=ck['optimizer']['state'][pid]['exp_avg'];reject('wrong_moment_shape',lambda:ck['optimizer']['state'][pid].__setitem__('exp_avg',torch.zeros(1)),lambda:ck['optimizer']['state'][pid].__setitem__('exp_avg',old))
try:V.validate_ckpt_payload(ck,'compact1024',R,identity,expect_step=100000,n_nodes=4000000)
except AssertionError:checks['wrong_actual_width']=True
else:checks['wrong_actual_width']=False
out={'PASS':all(checks.values()),'checks':checks,'source_checkpoint':str(p),'source_checkpoint_sha':V.full_sha(p),'validator_sha':V.full_sha(Path(V.__file__)),'scope':'Actual033100K4M core checkpoint used as explicit structural fixture, identity replaced/observer disabled. Validates038state reader, not038training or resume bitwise behavior; actualdevice twins still mandatory.'};(O/'card038-resume-schema-canary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));assert out['PASS']
