"""Actual core device resume twins at both widths, on512 training rows with exact15 graph.
Small epochs exercise boundary resume cheaply. Full300K configuration is separately preflighted.
Observer adds no training/RNG change; precision, width, radii and bound identity are checked.
"""
import sys,gc,json,tempfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import numpy as np,torch
import card038_validate as V
from card038_fit import configure_pumap
from basemap.pumap.parametric_umap.core import ParametricUMAP
R=Path(__file__).resolve().parents[2];O=V.OC;D=O/'card038-canary-data';N=512;STEPS=18

def same(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def run(arm,base,observer=True,radii=None,resume=None,overrides=None):
 gc.collect();torch.cuda.empty_cache();torch.manual_seed(V.SEED);np.random.seed(V.SEED);torch.cuda.manual_seed_all(V.SEED)
 x=np.load(D/'X.npy');r=np.load(D/'radii.npy') if radii is None else radii
 identity={**V.expected_identity(arm,R),'scope':'512-row actual-core boundary canary','n_nodes':N,'dose':STEPS,'rankneg_window':N//4,'observe_exposure':observer,'canary_radii_sha':hashlib.sha256(r.tobytes()).hexdigest(),'canary_input_sha':V.full_sha(D/'X.npy'),'canary_graph_sha':V.full_sha(D/'edges.npz')};identity.update(overrides or {})
 if resume:
  ck=torch.load(resume,map_location='cpu',weights_only=False);V.validate_ckpt_payload(ck,arm,R,identity,n_nodes=N)
 p=ParametricUMAP.load(str(V.CHAMPION),device='cuda');configure_pumap(p,arm,STEPS,r,identity,{3,7,STEPS});p.rankneg_window=N//4;p._card038_observe=observer
 p.fit(x,precomputed_edges_path=str(D/'edges.npz'),random_state=V.SEED,verbose=False,warm_start_state=V.expected_init(arm),checkpoint_every_epochs=1,checkpoint_dir=str(base),resume_from=str(resume) if resume else None)
 assert p._pipeline_info['x_residency']=='device_fp16' and p._train_stats['positive_lr_optimizer_steps']==STEPS
 assert sum(t.numel() for t in p.model.parameters())==V.NPARAM[arm];assert p.model.proj_out.out_features==3
 ck=torch.load(Path(base)/f'ckpt-step{STEPS}.pt',map_location='cpu',weights_only=False);V.validate_ckpt_payload(ck,arm,R,identity,expect_step=STEPS,n_nodes=N)
 return ck

def main():
 assert V.runtime_manifest_check(R)[0];proof=json.loads((O/'card038-foundation.json').read_text());assert proof['PASS']
 for n,h in proof['canary_data'].items():assert V.full_sha(D/n)==h
 checks={};first_positive={};fields=['model','optimizer','scheduler','scaler','loader_gen','loader_perm','loader_pos_idx','loader_batch_no','loader_rank_of_node','loader_node_at_rank','rankneg_scale','torch_rng','cuda_rng']
 with tempfile.TemporaryDirectory(dir=str(V.SB),prefix='card038-gpu-canary-') as td:
  T=Path(td)
  for arm in V.ARMS:
   for name in ['full','off','mid','epoch','ones']:(T/arm/name).mkdir(parents=True)
   full=run(arm,T/arm/'full');first_positive[arm]=full['train_stats']['card038_attempted_probes'][0]['positive_pair_sha256'];off=run(arm,T/arm/'off',observer=False)
   checks[arm+'_observer_off_model_optimizer_RNG_bitwise']=all(same(full[k],off[k]) for k in fields)
   checks[arm+'_counts_match_attempts']=full['train_stats']['card038_attempted_positive']>=full['train_stats']['card038_successful_positive']>0
   step=T/arm/'full/ckpt-step3.pt';obj=torch.load(step,map_location='cpu',weights_only=False);assert 0<obj['global_step']<STEPS
   mid=run(arm,T/arm/'mid',resume=step);checks[arm+'_mid_resume_full_state']=all(same(full[k],mid[k]) for k in fields);checks[arm+'_mid_resume_exposure']=all(same(full['train_stats'][k],mid['train_stats'][k]) for k in full['train_stats'] if k.startswith('card038_'))
   epochs=list((T/arm/'full').glob('ckpt-epoch*.pt'));epochs.sort(key=lambda p:int(torch.load(p,map_location='cpu',weights_only=False)['global_step']));ep=epochs[0];eobj=torch.load(ep,map_location='cpu',weights_only=False);assert 0<eobj['global_step']<STEPS and eobj['epoch']<full['epoch']
   resumed=run(arm,T/arm/'epoch',resume=ep);checks[arm+'_epoch_resume_cross_boundary_full_state']=all(same(full[k],resumed[k]) for k in fields);checks[arm+'_epoch_resume_exposure']=all(same(full['train_stats'][k],resumed['train_stats'][k]) for k in full['train_stats'] if k.startswith('card038_'))
   ones=run(arm,T/arm/'ones',radii=np.ones(N,'f4'));checks[arm+'_half_radii_active']=not same(full['model'],ones['model'])
   def reject(changes):
    try:run(arm,T/arm/'mid',resume=step,overrides=changes)
    except AssertionError as e:return 'admission-identity mismatch' in str(e)
    return False
   checks[arm+'_wrong_width_reject']=reject({'width_hidden_dim':999});checks[arm+'_wrong_dose_reject']=reject({'dose':19});checks[arm+'_wrong_radius_reject']=reject({'canary_radii_sha':'wrong'})
 checks['widths_attempted_positive_probe_match']=len(set(first_positive.values()))==1
 result={'PASS':bool(all(checks.values())),'checks':checks,'runtime_manifest_sha':V.full_sha(V.runtime_manifest_path(R)),'scope':'Actual device core on512 full1536 training rows, exact15 graph, production batch and quarter-pool ranked noise. Both widths, active half radii, actual step/epoch state and read-only observer. Full300K throughput remains separately gated.'};(O/'card038-canary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['PASS']
if __name__=='__main__':main()
