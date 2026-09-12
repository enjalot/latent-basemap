"""Two setup-separated fresh fits with fullgraph checkpoints, unchanged full-dose admission."""
import time,json,math,datetime as dt
from pathlib import Path
from run_card030_arm import fit,OC,sha,write
CAP=3600;END=dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp()
def main():
 start=time.monotonic();a=fit('leaky',1000,True,ckpt_steps=[500]);b=fit('leaky',3000,True,ckpt_steps=[500]);s1=a['stage_wall_s'];s2=b['stage_wall_s'];assert s2>s1>0
 rate=max((s2-s1)/2000,s2/3000);setup=max(0,s1-1000*rate);per=setup+60000*rate+3*22+3*3+60+90
 c=json.loads((OC/'card030-ledger.json').read_text());w=json.loads((OC/'cards-24h-window-ledger.json').read_text());spent=c['batch_spent_s'];wall=time.monotonic()-start
 checks={'per_arm_fits1500':per<=1500,'both_fit_cap':2*per+spent+wall<=CAP,'window':2*per+w['spent_s']+wall<=86400,'deadline':2*per+60<=END-time.time(),'finite_rate':math.isfinite(rate) and rate>0,'vram':max(a['peak_allocated_gib'],b['peak_allocated_gib'])<30}
 out={'PASS':all(checks.values()),'checks':checks,'dose':60000,'windows_s':{'1000':s1,'3000':s2},'per_step_s':rate,'setup_s':setup,'per_arm_s':per,'two_arm_s':2*per,'spent_before_preflight':spent,'preflight_wall_s':wall,'source_sha':sha(__file__),'reserves':'Conservative max(two-fit slope,longfit average); nonnegative setup;66s epoch writes+9s step writes+60s endpoint+90s margin/arm. No dose truncation.'};write(OC/'card030-preflight.json',out);print(json.dumps(out,indent=2));assert out['PASS']
if __name__=='__main__':main()
