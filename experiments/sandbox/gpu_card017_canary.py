"""Short real-path GPU checks for the new training-support inputs."""
import json
from run_card017_arm import fit,OC,write

def main():
 a=fit('uniform',100,True);b=fit('uniform',100,True);c=fit('geometric',100,True)
 checks={'same_draw_same_endpoint':a['endpoint_named_sha']==b['endpoint_named_sha'],
         'new_support_diverges':a['endpoint_named_sha']!=c['endpoint_named_sha'],
         'same_initial_weights':len({v['warm_parameter_sha'] for v in [a,b,c]})==1,
         'all_successful_dose':all(v['stats']['positive_lr_optimizer_steps']==100 for v in [a,b,c])}
 out={'PASS':all(checks.values()),'checks':checks,'arms':[a,b,c]};write(OC/'card017-gpu-canary.json',out)
 print(json.dumps(out,indent=2),flush=True);assert out['PASS']

if __name__=='__main__':main()
