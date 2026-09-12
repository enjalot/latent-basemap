import json,shutil
import torch
from run_card019_arm import fit,OUT,OC,ROOT,sha,write

def main():
 a=fit('relu',240,True);b=fit('leaky',240,True,ckpt_steps=[120])
 path=OUT/'canary-leaky/ckpt/ckpt-step120.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
 assert ck['global_step']==120 and ck['step_checkpoint'] and ck['config']['final_activation']=='leaky_relu_slope_0p01'
 preserved=path.with_name('resume-canary-original.pt');shutil.copy2(path,preserved)
 resumed=fit('leaky',240,True,resume_from=str(preserved),ckpt_steps=[])
 checks={'genuine_midrun_ckpt':True,'activation_changes_endpoint':a['endpoint_named_sha']!=b['endpoint_named_sha'],
         'same_warm_weights':a['warm_parameter_sha']==b['warm_parameter_sha'],'leaky_resume_bitwise':resumed['endpoint_named_sha']==b['endpoint_named_sha']}
 try:
  fit('relu',240,True,resume_from=str(preserved),ckpt_steps=[],identity_override=ck['card012_identity'])
  checks['wrong_activation_rejects']=False
 except ValueError as exc:checks['wrong_activation_rejects']='resume final_activation mismatch' in str(exc)
 r={'PASS':all(checks.values()),'checks':checks,'arms':{'relu':a,'leaky':b,'resumed':resumed},'canary_sha':sha(__file__),'checkpoint_sha':sha(preserved)}
 write(OC/'card019-gpu-canary.json',r);print(json.dumps({'PASS':r['PASS'],'checks':checks},indent=2));assert r['PASS']
if __name__=='__main__':main()
