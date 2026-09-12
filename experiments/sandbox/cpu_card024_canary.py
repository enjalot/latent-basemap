"""Actual production NEG/NCE helper and zero-distance radial tested against independent scalar math."""
import os,sys,json,math
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import torch,numpy as np
from basemap.pumap.parametric_umap.core import ParametricUMAP,contrastive_logit_loss
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import DeviceEdgeSampler
import card024_validate as V

def main():
 torch.set_num_threads(2);checks={};p=ParametricUMAP(device='cpu',a=V.A,b=V.B,low_dim_kernel='umap')
 def radial_loss(distance,beta):
  x=torch.tensor([[distance,0.]],dtype=torch.float32,requires_grad=True);_,rad=p._low_dim_qs(x,torch.zeros_like(x));loss=contrastive_logit_loss(rad,torch.ones(1),beta,V.A);g=torch.autograd.grad(loss,x)[0];return float(loss.detach()),g
 for d in [0.,1e-5,.05,1.,100.,1e5]:
  val,g=radial_loss(d,torch.tensor(.3));expected=math.log1p(math.exp(.3)*(1+V.A*(d*d)**V.B));checks[f'scalar_reference_d{d}']=abs(val-expected)<3e-6;checks[f'finite_grad_d{d}']=bool(torch.isfinite(g).all());checks[f'nonzero_or_zero_grad_d{d}']=bool(g.abs().sum()>0) if d else bool((g==0).all())
 checks['far_tail_not_probability_clamped']=radial_loss(1e5,torch.tensor(0.))[0]>-math.log(1e-7)+1
 radial=torch.linspace(.01,10,64,dtype=torch.float64,requires_grad=True);target=torch.zeros(64,dtype=torch.float64);target[:7]=1
 fixed=contrastive_logit_loss(radial,target,torch.tensor(0.,dtype=torch.float64),V.A);gfix=torch.autograd.grad(fixed,radial)[0];beta=torch.zeros((),dtype=torch.float64,requires_grad=True);learned=contrastive_logit_loss(radial,target,beta,V.A);gl,gb=torch.autograd.grad(learned,[radial,beta]);checks['beta0_exact_loss_model_gradient']=torch.equal(fixed,learned) and torch.equal(gfix,gl)
 h=1e-6;fd=(contrastive_logit_loss(radial,target,torch.tensor(h,dtype=torch.float64),V.A)-contrastive_logit_loss(radial,target,torch.tensor(-h,dtype=torch.float64),V.A))/(2*h);checks['beta_finite_difference']=abs(float((gb-fd).detach()))<1e-9
 q=1/(1+V.A*radial);prob=q/(q+1);direct=-(target*prob.log()+(1-target)*torch.log1p(-prob)).mean();checks['NEG_probability_equivalent']=abs(float((direct-fixed).detach()))<1e-12
 # Actual sampler versus independent draws using its cloned initial generator.
 sampler=DeviceEdgeSampler(None,np.arange(7),np.roll(np.arange(7),-1),None,7,batch_size=10,pos_ratio=.1,random_state=24,device='cpu')
 gen=torch.Generator().set_state(sampler.gen.get_state());ss=torch.randint(0,7,(4200,),generator=gen);off=torch.randint(1,7,(4200,),generator=gen);dd=(ss+off)%7;a,b=sampler._sample_negatives(4200);checks['actual_uniform_nonself_draw_formula']=torch.equal(ss,a) and torch.equal(dd,b);checks['nonself_observed']=bool((a!=b).all());checks['all42_ordered_pairs_covered']=len(torch.unique(a*7+b))==42
 allpairs={(i,(i+j)%7) for i in range(7) for j in range(1,7)};checks['toy_exact_uniform_support']=allpairs=={(i,j) for i in range(7) for j in range(7) if i!=j}
 z=np.load(V.GRAPH);checks['each_graph_row15']=np.array_equal(np.bincount(z['sources'],minlength=V.N),np.full(V.N,15));checks['graph_nonself']=not np.any(z['sources']==z['targets']);V.check_init();checks['full_actual_init_identity']=True
 r={'PASS':bool(all(checks.values())),'n_checks':len(checks),'checks':{k:bool(v) for k,v in checks.items()},'noise_recipe':V.noise_recipe(),'scope':'Actual production helper/radial, independent scalar/gradient references incl zero/far tail, actual sampler draw law, full graph degree/self and actual full init hashes. GPU path/resume validation still mandatory.'};(V.OC/'card024-cpu-canary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));assert r['PASS']
if __name__=='__main__':main()
