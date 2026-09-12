"""Independent train-radius formulas and prospective score-gate controls, no GPU/outcomes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.stats import rankdata
import card053_common as C
sys.path.insert(0,str(C.O))
import score_card053 as S

def main():
 start=time.monotonic();m=C.input_check();checks={};D=C.D;raw=np.load(D/'r-raw.npy');old=json.loads((D.parent/'card033-scale4m/manifest.json').read_text());rr=np.maximum(raw/old['radius']['train_p95'],1e-6);half=np.load(D/'r-half.npy');q=np.load(D/'r-quarter.npy');t=np.load(D/'r-taper.npy');p=np.load(D/'training-percentile.npy');w=np.load(D/'taper-weight.npy');ind=(rankdata(rr,method='average')-.5)/len(rr);u=np.clip((ind-.7)/.3,0,1);iw=(1-u)**2*(1+2*u)
 checks['parent_exact_half']=np.array_equal(half,np.load(D.parent/'card033-scale4m/r_actual.npy'));checks['original_raw_formula']=np.array_equal(half,np.sqrt(rr).astype('f4'));checks['independent_midrank']=np.array_equal(p,ind);checks['smoothstep_independent_factored']=np.allclose(w,iw,rtol=0,atol=1e-15);checks['quarter_exact']=np.array_equal(q,np.sqrt(half));checks['taper_formula']=np.max(np.abs(t-half.astype('f8')**iw))<1e-7;checks['dense_exact_control']=np.array_equal(t[p<=.7],half[p<=.7]);checks['attenuate_log_radius']=np.all(np.abs(np.log(t.astype('f8')))<=np.abs(np.log(half.astype('f8')))+2e-7);checks['finite_positive']=all(np.isfinite(a).all() and (a>0).all() for a in [half,q,t]);checks['distinct_radius_arrays']=len({C.sha(D/C.RAD[a]) for a in C.ARMS})==3
 ex=np.array([1,1,2,3,3,3.]);ep=(rankdata(ex)-.5)/len(ex);checks['ties_same_percentile']=ep[0]==ep[1] and ep[3]==ep[4]==ep[5];xx=np.array([0,.7,.85,1.]);uu=np.clip((xx-.7)/.3,0,1);ww=(1-uu)**2*(1+2*uu);checks['taper_endpoints']=np.allclose(ww,[1,1,.5,0],atol=1e-15)
 ids=np.load(D/'train-ids.npy');seal=Path('/data2/monet/eval-common-v2');checks['queries_excluded']=not np.isin(np.load(seal/'val_idx.npy'),ids).any();checks['reference_excluded']=not np.isin(np.load(seal/'ref_idx.npy'),ids).any()
 z={'val_source':np.repeat(np.array(['a','b','c']),100),'decile_zero_based':np.tile(np.arange(10),30),'enc_radius':np.arange(300.)+1};baseline=np.ones(300)*.6
 for a in S.ARMS:
  for b in S.BUDGETS:z[f'{a}_B{b}']=baseline.copy()
  z[a+'_continuity_per_query']=np.ones(30)*.8;z[a+'_map_radius']=z['enc_radius'].copy()
 ci={};good={'pooled_sparse_B250_gain':{'quarter':.006,'sparse_taper':.006},'CI975':{'quarter':[.001,.01],'sparse_taper':[.001,.01]}}
 r=S.aggregate(z,ci,good);checks['all_guards_and_selection']=r['GATE_PASS'] and r['selected_candidate']=='quarter'
 bad={**good,'CI975':{'quarter':[-.001,.01],'sparse_taper':[-.001,.01]}};checks['CI_failure_fails']=not S.aggregate(z,ci,bad)['GATE_PASS']
 bad={**good,'pooled_sparse_B250_gain':{'quarter':.004,'sparse_taper':.004}};checks['minimum_gain_fails']=not S.aggregate(z,ci,bad)['GATE_PASS']
 for a in ['quarter','sparse_taper']:z[a+'_B2000'][z['decile_zero_based']==9]-=.011
 checks['sparse_decile_guard_AND']=not S.aggregate(z,ci,good)['GATE_PASS']
 for a in ['quarter','sparse_taper']:z[a+'_B2000']=baseline.copy()
 z['parent_B250']+=.006;checks['parent_guard_AND']=not S.aggregate(z,ci,good)['GATE_PASS'];z['parent_B250']=baseline.copy()
 for a in ['quarter','sparse_taper']:z[a+'_continuity_per_query']-=.006
 checks['continuity_guard_AND']=not S.aggregate(z,ci,good)['GATE_PASS']
 r={'PASS':all(checks.values()),'checks':{k:bool(v) for k,v in checks.items()},'cpu_s':time.monotonic()-start,'input_manifest_sha':C.sha(D/'inputs-manifest.json'),'test_sha':C.sha(__file__),'scope':'All4M radius formulas and train/eval exclusion joins; synthetic gate fixtures. No GPU or heldout model outcomes.'};C.write(C.O/'card053-cpu-canary.json',r);print(json.dumps(r,indent=2));assert r['PASS']
if __name__=='__main__':main()
