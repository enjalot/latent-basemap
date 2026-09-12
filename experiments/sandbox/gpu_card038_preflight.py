"""Measure unchanged full300K core path at500/3500 updates per width, with real checkpoints.
All fits plus charged prep/reserves must fit caps; no dose/width fallback.
"""
import sys,time,json,gc,math,tempfile,datetime as dt
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import numpy as np,torch
import card038_validate as V
from card038_fit import configure_pumap
from basemap.pumap.parametric_umap.core import ParametricUMAP
R=Path(__file__).resolve().parents[2];O=V.OC;CAPS={'wide2048':1800,'compact1024':3000};END=dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp()
def measured(arm,steps,d):
 gc.collect();torch.cuda.empty_cache();torch.manual_seed(V.SEED);np.random.seed(V.SEED);torch.cuda.manual_seed_all(V.SEED);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 x=np.array(np.load(V.SUB,mmap_mode='r'),'f4',copy=True);r=np.load(V.RADII);ident={**V.expected_identity(arm,R),'scope':'full300K throughput preflight','dose':steps}
 p=ParametricUMAP.load(str(V.CHAMPION),device='cuda');configure_pumap(p,arm,steps,r,ident,{min(500,steps),steps})
 p.fit(x,precomputed_edges_path=str(V.GRAPH),random_state=V.SEED,verbose=False,warm_start_state=V.expected_init(arm),checkpoint_every_epochs=1,checkpoint_dir=str(d))
 torch.cuda.synchronize();elapsed=time.monotonic()-start;ck=torch.load(d/f'ckpt-step{steps}.pt',map_location='cpu',weights_only=False);V.validate_ckpt_payload(ck,arm,R,ident,expect_step=steps)
 assert p._pipeline_info['x_residency']=='device_fp16';free,total=torch.cuda.mem_get_info();return {'seconds':elapsed,'steps':steps,'process_peak_gib':torch.cuda.max_memory_allocated()/2**30,'global_used_gib':(total-free)/2**30,'actual_attempted_batches':p._train_stats['attempted_batches'],'actual_successful_positive':p._train_stats['card038_successful_positive']}
def main():
 assert V.runtime_manifest_check(R)[0];start=time.monotonic();fits={};checks={};estimates={}
 with tempfile.TemporaryDirectory(dir=str(V.SB),prefix='card038-preflight-') as td:
  for a in V.ARMS:
   fits[a]={}
   for steps in [500,3500]:
    d=Path(td)/f'{a}-{steps}';d.mkdir();fits[a][str(steps)]=measured(a,steps,d)
   w1=fits[a]['500']['seconds'];w2=fits[a]['3500']['seconds'];assert math.isfinite(w1) and math.isfinite(w2) and w2>w1>0
   slope=max((w2-w1)/3000,w2/3500);setup=max(0,w1-500*slope);epochs=math.ceil(V.DOSE[a]/math.ceil(V.N*15/int(V.BATCH*V.POS_RATIO)))
   # Includes measured early serialization, plus explicit repeated epoch/step and endpoint reserves.
   reserve=epochs*3+len(V.SNAP[a])*6+120+180;cost=setup+slope*V.DOSE[a]+reserve;estimates[a]={'per_step_s':slope,'setup_s':setup,'epochs':epochs,'reserve_s':reserve,'complete_arm_s':cost};checks[a+'_fits_arm_cap']=cost<=CAPS[a];checks[a+'_global_memory']=max(fits[a][k]['global_used_gib'] for k in fits[a])<30
 spent=json.loads((O/'card038-ledger.json').read_text())['batch_spent_s'];win=json.loads((O/'cards-24h-window-ledger.json').read_text())['spent_s'];used=time.monotonic()-start;need=sum(x['complete_arm_s'] for x in estimates.values())+120
 checks['fits_card']=need+spent+used<=5400;checks['fits_window']=need+win+used<=86400;checks['fits_deadline']=need<=END-time.time()
 result={'PASS':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'fits':fits,'estimates':estimates,'remaining_fits_plus_benchmark_s':need,'preflight_elapsed_s':used,'runtime_manifest_sha':V.full_sha(V.runtime_manifest_path(R)),'scope':'Measured actual300K configuration, exact unchanged60K/180K doses; conservative serialization and endpoint/benchmark reserve; noautomaticdose reduction.'};(O/'card038-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0 if result['PASS'] else 3
if __name__=='__main__':raise SystemExit(main())
