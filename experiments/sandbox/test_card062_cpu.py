"""CPU operator checks using only synthetic and exposed development values."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import numpy as np,torch
import card062_common as C
checks=[]
def scalar(t,z,w,c):
 t=t.copy();d=np.zeros(t.shape,dtype=np.longdouble)
 for i in range(len(t)):
  if np.max(z[i])<=0:
   v=z[i].astype(np.longdouble)@w.astype(np.longdouble).T;n=np.sqrt((v*v).sum())
   if n>0:d[i]=c*v/n;t[i]=(t[i].astype(np.longdouble)+d[i]).astype('f4')
 return t,d
for seed in [61,62,63]:
 rng=np.random.default_rng(seed);z=rng.normal(size=(513,64)).astype('f4');z[:300]=-np.abs(z[:300]);z[300]=0;t=rng.normal(size=(513,3)).astype('f4');w=rng.normal(size=(3,64)).astype('f4');w[:,0]=0;t[400,0]=-0.
 expected,de=scalar(t,z,w,.001);y,d,b=C.kernel(torch.from_numpy(t),torch.from_numpy(z),torch.from_numpy(w),.001,collect=True)
 assert np.array_equal(y.numpy().view('u4'),expected.view('u4'));assert np.allclose(d.numpy(),de,rtol=1e-12,atol=1e-15);assert np.array_equal(y.numpy()[~b].view('u4'),t[~b].view('u4'));checks+=['scalar FP32 '+str(seed),'offset FP64 '+str(seed),'active bytes '+str(seed)]
 for size in [1,17,256,513]:
  yy=torch.cat([C.kernel(torch.from_numpy(t[i:i+size]),torch.from_numpy(z[i:i+size]),torch.from_numpy(w),.001)[0] for i in range(0,513,size)]);assert torch.equal(yy,y);checks.append('kernel chunk '+str(seed)+'/'+str(size))
for name in C.HEADS:
 z=np.load(C.O/f'card057-projection/{name}.npz');c=C.read(C.O/'card059-score.json')['heads'][name]['constant_norm'];expected=np.load(C.O/f'card059-scoring/{name}-coords.npz')
 for pop in ['reference','target']:
  rows=z[pop+'_inactive_rows'];t=z[pop+'_teacher'][rows];pre=z[pop+'_inactive_preactivation'];yy,dd,mask=C.kernel(torch.from_numpy(t),torch.from_numpy(pre),torch.from_numpy(z['output_weight']),c,collect=True)
  assert np.array_equal(yy.numpy(),expected[pop+'_direction_constant_norm'][rows]);ss,sd=scalar(t,pre,z['output_weight'],c);assert np.array_equal(yy.numpy(),ss);checks += [name+pop+'059 output exact',name+pop+' longdouble scalar exact']
C.write(C.O/'card062-cpu-canary.json',{'PASS':True,'n_checks':len(checks),'checks':checks,'scope':'Kernel formulas, grouping and original059 outputs; no new query or GPU outcome.'});print('PASS',len(checks))
