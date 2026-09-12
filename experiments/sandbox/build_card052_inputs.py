"""Train-only PCA/global target preparation, no GPU and no heldout map outcomes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib,time,resource,datetime as dt
import numpy as np,torch
from scipy.linalg import eigh
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));from basemap.pumap.parametric_umap.core import ParametricUMAP
O=R.parent/'overseer-codex';D=Path('/data/latent-basemap/substrates/card052-global-pca');BASE=Path('/data/latent-basemap/substrates/card010-adaptive');PARENT=R.parent/'card015-train/model-actual_full3d.pt'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 start=time.monotonic();torch.set_num_threads(2);D.mkdir(exist_ok=True);assert not (D/'inputs-manifest.json').exists();prep=json.loads((O/'card052-preparation.json').read_text());assert prep['GPU_RELEASED'] is False and prep['protocol_sha']==sha(O/'card052-global-pca-tether.md')
 for src,name in [(BASE/'substrate.f16.npy','train.f16.npy'),(BASE/'edges-fixed15.npz','edges-fixed15.npz'),(BASE/'draw_ids.npy','train-ids.npy'),(R.parent/'card013-radii/r_actual.npy','radii.npy')]:
  if not (D/name).exists():os.link(src,D/name)
  assert sha(src)==sha(D/name)
 X=np.load(D/'train.f16.npy',mmap_mode='r');ids=np.load(D/'train-ids.npy');assert X.shape==(300000,1536) and len(np.unique(ids))==300000
 fit=np.random.default_rng(52052).choice(len(X),20000,replace=False);bank=np.random.default_rng(52055).choice(len(X),200000,replace=False);np.savez(D/'training-selections.npz',fit_local=fit,fit_ids=ids[fit],bank_local=bank,bank_ids=ids[bank])
 panel=np.load(O/'card013-scoring/closeout-persist.npz')['panel_local'];assert len(panel)==1800;perm=np.random.default_rng(52053).permutation(1800);np.savez(D/'global-pairs-before-outcomes.npz',panel_local=panel,pair_panel_indices=perm.reshape(-1,2));assert len(np.unique(perm))==1800
 A=np.asarray(X[fit],'f8');mean=A.mean(0);A-=mean;cov=A.T@A/(len(A)-1);evals,vec=eigh(cov,subset_by_index=[1533,1535],driver='evr');evals=evals[::-1];vec=vec[:,::-1].copy();del A
 # Canonicalize signs, then verify the full covariance eigen-equation independently.
 for j in range(3):
  if vec[np.argmax(np.abs(vec[:,j])),j]<0:vec[:,j]*=-1
 orth=float(np.max(np.abs(vec.T@vec-np.eye(3))));res=float(np.linalg.norm(cov@vec-vec*evals)/np.linalg.norm(cov@vec));assert orth<1e-10 and res<1e-8 and (evals>0).all()
 np.savez(D/'pca-fit.npz',mean=mean,components=vec,eigenvalues=evals,covariance=cov,fit_local=fit);xb=np.array(X[bank],dtype='f2');pca=(xb.astype('f8')-mean)@vec
 model=ParametricUMAP.load(str(PARENT),device='cpu').model.eval();pred=[]
 with torch.inference_mode():
  for lo in range(0,len(xb),4096):pred.append(model(torch.from_numpy(xb[lo:lo+4096].astype('f4'))).numpy())
 pred=np.concatenate(pred);assert pred.shape==(200000,3) and np.isfinite(pred).all();pm=pca.mean(0);ym=pred.astype('f8').mean(0);pc=pca-pm;yc=pred.astype('f8')-ym;alpha=float(np.linalg.norm(yc)/np.linalg.norm(pc));u,sv,vt=np.linalg.svd(pc.T@yc);rot=u@vt;targets=(alpha*pc@rot+ym).astype('f4');M=float(np.mean((pred.astype('f8')-targets.astype('f8'))**2));assert M>1e-8 and np.isfinite(M);weights={'ordinary':0.,'weak_PCA':.005/(3*M),'strong_PCA':.02/(3*M)}
 # Orthogonal alignment may include reflection, which preserves Euclidean geometry.
 assert np.allclose(rot.T@rot,np.eye(3),atol=1e-10);assert np.allclose(targets.mean(0),ym,atol=1e-4);assert np.isfinite(targets).all();assert np.max(np.abs(np.linalg.norm(xb.astype('f4'),axis=1)-1))<.002
 np.savez(D/'pca-bank.npz',replay_X=xb,replay_targets=targets,replay_ids=ids[bank]);np.savez(D/'bank-target-provenance.npz',parent_predictions=pred,pca_coordinates=pca,aligned_targets=targets,rotation=rot,pca_center=pm,parent_center=ym,alpha=alpha)
 warm=torch.load(PARENT,map_location='cpu',weights_only=False)['model_state_dict'];torch.save({'model_state':warm,'parent_sha':sha(PARENT)},D/'init.pt')
 out={'PASS':True,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'n_train':300000,'dim':1536,'n_components':3,'bank_rows':200000,'fit_rows':20000,'seeds':{'fit':52052,'bank':52055,'global_pairs':52053,'replay':52052},'weights':weights,'initial_CPU_FP32_bank_MSE':M,'initial_weighted_replay_loss':{a:w*3*M for a,w in weights.items()},'eigenvalues':evals.tolist(),'fit_total_variance':float(np.trace(cov)),'fit_PCA3_variance_fraction':float(evals.sum()/np.trace(cov)),'eigen_residual_relative':res,'orthogonality_maxabs':orth,'alignment_alpha':alpha,'alignment_determinant':float(np.linalg.det(rot)),'parent_sha':sha(PARENT),'init_state_sha256':state_sha(warm),'files':{p.name:sha(p) for p in D.iterdir() if p.is_file()},'protocol_sha':prep['protocol_sha'],'builder_sha':sha(__file__),'cpu_s':time.monotonic()-start,'max_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'GPU_s':0,'scope':'Fit/bank entirelyoriginal300Ktraininggraph, exactstoredFP16values. FixedPCA3targets aligned/scaled toparent; RMS/isometry do not imply semantictruth. CPUFP32initialpenalty setsweights, not gradientmatching; training replayusesexistingAMPpath.'};assert out['max_rss_MiB']<16384;write(D/'inputs-manifest.json',out);print('INPUTS READY',weights,'MSE',M,'sec',out['cpu_s'],flush=True)
def state_sha(sd):
 h=hashlib.sha256()
 for k in sorted(sd):h.update(k.encode());h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
 return h.hexdigest()
if __name__=='__main__':main()
