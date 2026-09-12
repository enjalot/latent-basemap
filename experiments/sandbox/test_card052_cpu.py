"""PCA/bank identity and prospective gate controls before GPU release."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,time,json,copy,hashlib
import numpy as np,torch
import card052_common as C
sys.path.insert(0,str(C.O));import score_card052 as S
checks=[];t=time.monotonic()
def ck(v,n):assert bool(v),n;checks.append(n)
m=C.input_check();X=np.load(C.D/'train.f16.npy',mmap_mode='r');sel=np.load(C.D/'training-selections.npz');bank=np.load(C.D/'pca-bank.npz');ids=np.load(C.D/'train-ids.npy');fit=np.load(C.D/'pca-fit.npz');target=np.load(C.D/'bank-target-provenance.npz');ck(np.array_equal(bank['replay_X'],X[sel['bank_local']]),'all200Kinput joins');ck(np.array_equal(bank['replay_ids'],ids[sel['bank_local']]),'all200Kglobal IDs');ck(len(np.unique(sel['fit_local']))==20000 and np.array_equal(sel['fit_ids'],ids[sel['fit_local']]),'PCAfit training IDs');ck(np.array_equal(np.random.default_rng(52052).choice(300000,20000,replace=False),sel['fit_local']),'PCAfit seed');ck(np.array_equal(np.random.default_rng(52055).choice(300000,200000,replace=False),sel['bank_local']),'bankseed')
panel=np.load(C.D/'global-pairs-before-outcomes.npz');ck(np.array_equal(panel['pair_panel_indices'].ravel(),np.random.default_rng(52053).permutation(1800)),'frozen disjoint900pairs');val=np.load('/data2/monet/eval-common-v2/val_idx.npy');ck(not np.isin(val,ids).any(),'heldout queries excluded fromalltraining')
mean=fit['mean'];comp=fit['components'];cov=fit['covariance'];eig=fit['eigenvalues'];ck(np.max(np.abs(comp.T@comp-np.eye(3)))<1e-10,'orthogonalPCA');ck(np.linalg.norm(cov@comp-comp*eig)/np.linalg.norm(cov@comp)<1e-8,'eigen equation');ck(np.max(np.abs(X[sel['fit_local']].astype('f8').mean(0)-mean))<1e-12,'PCAmean exactfit');calc=(bank['replay_X'].astype('f8')-mean)@comp;ck(np.max(np.abs(calc-target['pca_coordinates']))<1e-12,'allbankPCA targets');expect=(float(target['alpha'])*(calc-target['pca_center'])@target['rotation']+target['parent_center']).astype('f4');ck(np.array_equal(expect,bank['replay_targets']),'allbank alignment identity')
loss=((torch.tensor(target['parent_predictions'])-torch.tensor(bank['replay_targets'])).double().square().sum(1).mean()).item();cpu_loss=float(np.square(target['parent_predictions'].astype('f8')-bank['replay_targets'].astype('f8')).sum(1).mean());ck(abs(cpu_loss-m['initial_CPU_FP32_replay_loss'])<1e-12,'correct summed-coordinate denominator');ck(abs(m['weights']['weak_PCA']*cpu_loss-.005)<1e-12 and abs(m['weights']['strong_PCA']*cpu_loss-.02)<1e-12,'initialpenalties .005 .02')
# Goodglobal score must not override an old sparse or parent guard; weaker wins ties.
g=np.repeat(np.arange(9).astype(str),20);n=len(g);z={'val_source':g,'decile_zero_based':np.tile(np.arange(10),18),'enc_radius':np.linspace(.1,2,n)}
for a in S.ARMS:
 z[a+'_map_radius']=np.linspace(.1,2,n);z[a+'_continuity_per_query']=np.full(18,.9)
 for b in S.BUDGETS:z[a+'_B'+str(b)]=np.full(n,.8)
global_m={'disjoint_pair_CPD':{'parent':.1,'ordinary':.1,'weak_PCA':.15,'strong_PCA':.16},'candidate_minus_ordinary_CI975':{'weak_PCA':[.02,.07],'strong_PCA':[.02,.08]}}
r=S.aggregate(z,{},global_m);ck(r['GATE_PASS'] and r['selected_candidate']=='weak_PCA','weaker wins bothpass')
v=copy.deepcopy(z);v['weak_PCA_B250'][v['decile_zero_based']==9]=.78;r=S.aggregate(v,{},global_m);ck(not all(r['gates']['weak_PCA'].values()) and r['selected_candidate']=='strong_PCA','sparsestguard rejectsweak')
v=copy.deepcopy(z);v['parent_B250']+=.006;r=S.aggregate(v,{},global_m);ck(not r['GATE_PASS'],'originalparent protects againstdegradedordinary')
v=copy.deepcopy(global_m);v['candidate_minus_ordinary_CI975']['weak_PCA']=[-.01,.08];r=S.aggregate(z,{},v);ck(not r['gates']['weak_PCA']['CPD_CI975_positive'],'globalCI crossingzero rejects')
v=copy.deepcopy(global_m);v['disjoint_pair_CPD']['weak_PCA']=.129;r=S.aggregate(z,{},v);ck(not r['gates']['weak_PCA']['CPD_gain_ge03'],'globalpractical threshold')
for a in C.ARMS:
 ident=C.identity(a);ck(ident['replay_weight']==m['weights'][a] and ident['lr']==.0001 and ident['input_dim']==1536,'admittedconfig '+a)
C.write(C.O/'card052-cpu-canary.json',{'PASS':True,'checks':checks,'n_checks':len(checks),'cpu_s':time.monotonic()-t,'GPU_s':0,'scorer_sha':C.sha(C.O/'score_card052.py'),'auditor_sha':C.sha(C.O/'audit_card052.py'),'runtime_sha':C.source_check(),'input_manifest_sha':C.sha(C.D/'inputs-manifest.json'),'scope':'CompletePCA/bank join/formula/seed/denominator audit and prospectivegate controls; actualGPUresumetwins stillrequired.'});print('CPU PASS',len(checks),flush=True)
