"""CPU-only original-instrument scoring: teacher, frozen20K, fixed40K and80K."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from run_card020 import *
from card020_instrument import project,recall,continuity,logdensity,quality_guards,BUDGETS
from benchmark_card016 import NativeCompact
from _paths import ensure_paths
ensure_paths()
from basemap.pumap.parametric_umap.core import ParametricUMAP
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
SEAL=Path('/data2/monet/eval-common-v2');DEST=OC/'card020-scoring';TEACHER=SB/'dino-arrival-t0/champion-bs16k/model.pt';HEADS=['teacher','parent20k','step40k','step80k']
def main():
 t=time.monotonic();torch.set_num_threads(2);validate_inputs();validate_endpoint();DEST.mkdir(exist_ok=True)
 m=json.loads((DATA/'manifest.json').read_text());prior=json.loads((OC/'card016-score.json').read_text());p=np.load(OC/'card016-scoring/per-query.npz');ids=np.load(SEAL/'val_idx.npy');rid=np.load(SEAL/'ref_idx.npy');truth=np.load(SEAL/'truth_val.npy');groups=np.load(SEAL/'val_source.npy',allow_pickle=True).astype(str)
 assert np.array_equal(ids,p['val_ids']) and np.array_equal(rid,p['ref_ids']) and np.array_equal(truth,p['truth']) and np.array_equal(groups,p['val_source']);assert not np.isin(ids,rid).any()
 for n,h in prior['provenance']['instrument'].items():assert sha(SEAL/n)==h
 assert sha(TEACHER)==m['teacher_head_sha'];ref=np.load(SEAL/'ref_hd.f16.npy',mmap_mode='r');val=np.load(SEAL/'val_hd.f16.npy',mmap_mode='r');by={g:np.flatnonzero(groups==g) for g in np.unique(groups)};dec=p['decile_zero_based'];panel=p['panel_local'];hi=p['panel_encoder15_local'];grid=torch.from_numpy(np.load(DATA/'density_grid.npy'))
 arrays={k:p[k] for k in ['ref_ids','val_ids','val_source','truth','enc_radius','decile_zero_based','panel_local','panel_val_ids','panel_hd','panel_encoder15_local']};xy={};rcs={};cont={};movement={};logp={};outside={};recs={};hashes={}
 for a in HEADS:
  path=TEACHER if a=='teacher' else PARENT if a=='parent20k' else OUT/f"step-{40000 if a=='step40k' else 80000}.pt";hashes[a]=sha(path)
  if a=='teacher':model=ParametricUMAP.load(str(path),device='cpu').model.eval()
  else:
   ob=torch.load(path,map_location='cpu',weights_only=False);expected={'parent20k':20000,'step40k':40000,'step80k':80000}[a];assert ob['successful_steps']==expected
   ob.update(center=m['center'],span=m['span']);model=NativeCompact(ob).eval()
  rc=project(model,ref);vc=project(model,val);del model;np.save(DEST/f'{a}-ref-xy.npy',rc);np.save(DEST/f'{a}-val-xy.npy',vc);xy[a]=vc;recs[a]=recall(rc,vc,truth);cont[a]=continuity(vc[panel],hi);logp[a],outside[a],_=logdensity(vc,m,grid)
  dist,_=cKDTree(rc.astype('f8')).query(vc.astype('f8'),k=15,workers=2);arrays[a+'_map_radius']=np.sqrt(np.square(dist).mean(1));arrays[a+'_val_xy']=vc;arrays[a+'_continuity_per_query']=cont[a];arrays[a+'_log_density']=logp[a];arrays[a+'_outside_grid']=outside[a]
  for b in BUDGETS:arrays[a+f'_B{b}']=recs[a][b]
  if a in ['teacher','parent20k']:
   old='teacher' if a=='teacher' else 'compact_l1';assert np.array_equal(vc,p[old+'_val_xy']),'frozen coordinate control changed'
   for b in BUDGETS:assert np.array_equal(recs[a][b],p[old+f'_B{b}']),'frozen recall control changed'
 for a in HEADS[1:]:movement[a]=np.linalg.norm(xy[a].astype('f8')-xy['teacher'].astype('f8'),axis=1)/33.6717;arrays[a+'_movement_native']=movement[a]
 np.savez(DEST/'per-query.npz',**arrays)
 equal={a:{str(b):float(np.mean([recs[a][b][i].mean() for i in by.values()])) for b in BUDGETS} for a in HEADS};sources={g:{a:{str(b):float(recs[a][b][ix].mean()) for b in BUDGETS} for a in HEADS} for g,ix in by.items()};decs={str(j+1):{'n':int((dec==j).sum()),'heads':{a:{str(b):float(recs[a][b][dec==j].mean()) for b in BUDGETS} for a in HEADS}} for j in range(10)}
 pairs=[(a,'teacher') for a in HEADS[1:]]+[('step40k','parent20k'),('step80k','parent20k'),('step80k','step40k')];boot={a+'-'+b:{str(k):[] for k in [250,2000]} for a,b in pairs};mb={a:[] for a in HEADS[1:]};rng=np.random.default_rng(16016016)
 for _ in range(2000):
  ss=[rng.choice(i,len(i),replace=True) for i in by.values()];ix=np.concatenate(ss)
  for a,b in pairs:
   for k in [250,2000]:boot[a+'-'+b][str(k)].append(np.mean([(recs[a][k][i]-recs[b][k][i]).mean() for i in ss]))
  for a in HEADS[1:]:mb[a].append(np.percentile(movement[a][ix],99))
 for v in boot.values():v['continuity']=[]
 ps=[np.flatnonzero(groups[panel]==g) for g in by]
 for _ in range(2000):
  ix=np.concatenate([rng.choice(i,len(i),replace=True) for i in ps])
  for a,b in pairs:boot[a+'-'+b]['continuity'].append((cont[a]-cont[b])[ix].mean())
 cis={pair:{k:np.percentile(v,[2.5,97.5]).tolist() for k,v in metrics.items()} for pair,metrics in boot.items()};gates={};details={};frame={}
 bench=json.loads((OC/'card016-projection-benchmark.json').read_text());assert bench['model_hashes']['compact_l1']==sha(SB/'card016-train/compact_l1/model.pt')
 for a in HEADS[1:]:
  g,d=quality_guards(a,'teacher',recs,cont,by,dec);g.update(speedup_ge_2=bench['speedup_vs_teacher']['compact_l1']>=2,parameters_le_30pct=bench['parameter_counts']['compact_l1']/bench['parameter_counts']['teacher']<=.3);gates[a]={'checks':g,'PASS':all(g.values())};details[a]=d
  mu=float(movement[a].mean());q=float(np.percentile(movement[a],99));frame[a]={'mean':mu,'p99':q,'p99_ci95':np.percentile(mb[a],[2.5,97.5]).tolist(),'PASS':mu<=.01 and q<=.05}
 curves={s:json.loads((OUT/f'dev-{s}.json').read_text()) for s in SNAPS}
 result={'status':'SCORED','schema':'card020-score-v1','selected_endpoint':'step80k','selected_candidate_pass':gates['step80k']['PASS'],'step40k_role':'descriptive_only_not_eligible_for_promotion','compression_candidate':gates,'frame_replacement':frame,'equal9_recall':equal,'per_source':sources,'per_decile':decs,'continuity':{a:float(cont[a].mean()) for a in HEADS},'paired_ci95':cis,'quality_guard_details':details,'dev_curve':curves,'benchmark':{'inherited_from':'card016-projection-benchmark.json','measured_endpoint':'Card016 compact_l1 at20K','status':'Inherited architecture/precision measurement; not independently measured40K/80K timings','speedup':bench['speedup_vs_teacher']['compact_l1'],'parameter_fraction':bench['parameter_counts']['compact_l1']/bench['parameter_counts']['teacher']},'model_hashes':hashes,'provenance':{'scorer_sha':sha(__file__),'input_manifest_sha':sha(DATA/'manifest.json'),'runtime_manifest_sha':sha(ROOT/'card020-runtime-sha.json')},'limits':'Fixed40K/80K descriptive endpoints, no adaptive selection; single seed development evidence. Original nine-source equal weighting and conditional continuity panel. No reserve10K. Model-only inherited throughput, not end-to-end100M speed.','cpu_wall_s':time.monotonic()-t}
 write(DEST/'result.json',result);write(OC/'card020-score.json',result)
if __name__=='__main__':main()
