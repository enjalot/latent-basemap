import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,pathlib,json,hashlib,numpy as np,torch,types,datetime
root=pathlib.Path('/data/latent-basemap/sandbox/card015-code');sys.path.insert(0,str(root));from basemap.pumap.parametric_umap.core import ParametricUMAP
sub=pathlib.Path('/data/latent-basemap/substrates/card010-adaptive'); oc=pathlib.Path('/data/latent-basemap/sandbox/overseer-codex'); rad=sub.parent.parent/'sandbox/card013-radii'
rad=pathlib.Path('/data/latent-basemap/sandbox/card013-radii')
r=np.load(rad/'r_actual.npy'); sh=np.load(rad/'r_shuffled.npy');ds=np.load(sub/'knn_dist.npy',mmap_mode='r')[:,1:16].astype('float64'); raw=np.sqrt((ds*ds).mean(1)); expected=np.maximum(raw/np.percentile(raw,95),1e-6).astype('float32');src=np.load(sub/'draw_source.npy',allow_pickle=True).astype(str)
checks={'radii_formula_exact':np.array_equal(r,expected),'all_source_distributions_preserved':all(np.array_equal(np.sort(r[src==g]),np.sort(sh[src==g])) for g in np.unique(src)), 'zero_floor_rows':not np.any(raw/np.percentile(raw,95)<1e-6)}
p=types.SimpleNamespace(_kernel_exp=1.,a=1.7,b=.8,low_dim_kernel='umap',kernel_alpha=1.)
for s in [1.,.2,1.4]:
 for y in [0.,1.]:
  x=torch.tensor([[.3,-.7,.2]],requires_grad=True); dst=torch.tensor([[-.2,.4,-.6]]); scale=torch.tensor([s]);q,_=ParametricUMAP._low_dim_qs(p,x,dst,scale); loss=torch.nn.functional.binary_cross_entropy(q,torch.tensor([y]));g=torch.autograd.grad(loss,x)[0].numpy()[0]
  delta=x.detach().numpy()[0].astype('float64')-dst.numpy()[0]; d2=np.sum(delta**2); a=p.a;b=p.b; t=d2/s; qq=1/(1+a*t**b); analytic=(qq-y)/(qq*(1-qq))*(-2*a*b*t**(b-1)*delta/s/(1+a*t**b)**2)
  checks[f'q_scalar_s{s}_y{y}']=abs(q.item()-qq)<1e-6;checks[f'BCE_gradient_scalar_s{s}_y{y}']=np.allclose(g,analytic,rtol=2e-6,atol=1e-6)
  swap,_=ParametricUMAP._low_dim_qs(p,dst,x.detach(),scale); checks[f'endpoint_swap_s{s}_y{y}']=torch.equal(q.detach(),swap)
h=hashlib.sha256()
with open(sub/'edges-fixed15.npz','rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
checks['edge_hash_matches_admission']=h.hexdigest()[:16]=='d214a839b07113df'
init2=torch.load(sub/'init-card010.pt',map_location='cpu',weights_only=False)['model_state']
init3=torch.load(pathlib.Path('/data/latent-basemap/sandbox/card015-init/init-card015-3d.pt'),map_location='cpu',weights_only=False)['model_state']
checks['shared_hidden_exact']=all(torch.equal(init2[k],init3[k]) for k in init2 if not k.startswith('proj_out.'))
checks['shared_first_two_rows']=all(torch.equal(init2[k],init3[k][:2]) for k in ['proj_out.weight','proj_out.bias'])
model=ParametricUMAP.load('/data/latent-basemap/sandbox/dino-arrival-t0/champion-bs16k/model.pt',device='cpu');model.model=None;model.n_components=3;model._init_model(1536);model.model.load_state_dict(init3)
x=torch.tensor(np.asarray(np.load(sub/'substrate.f16.npy',mmap_mode='r')[:8],np.float32),requires_grad=True); y=model.model(x);y.square().mean().backward()
checks['forward_3d_finite']=y.shape==(8,3) and bool(torch.isfinite(y).all())
checks['input_gradient_finite_nonzero']=bool(torch.isfinite(x.grad).all() and x.grad.abs().sum()>0)
out={'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PASS':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'edge_sha256':h.hexdigest(),'scope':'CPU full-radius formula/source permutation audit and independent scalar q/BCE gradient+swap calculations. Does not validate sampler endpoint identity or resume.'}
(oc/'card015-independent-cpu-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['PASS']
