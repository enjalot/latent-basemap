"""Pinned CPU reference and atomic directed-membership bundle; no feature bank or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib,inspect,time,resource
import numpy as np
R=Path(__file__).resolve().parents[2];SB=R.parent;O=SB/'overseer-codex';D=Path('/data/latent-basemap/substrates/card086-directed-membership')
IDX=SB/'dino-arrival-t0/knn_indices.npy';DIST=SB/'dino-arrival-t0/knn_dists.npy';G=Path('/data/latent-basemap/substrates/card073-finishing/edges-fixed15.npz')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def bound():assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*1024**2,'CPU RSS8GiB STOP'
def reference():
 import umap,umap.umap_ as u,numba
 numba.set_num_threads(2)
 return u,{'umap_version':umap.__version__,'numba_version':numba.__version__,'module_path':u.__file__,'module_sha':sha(u.__file__),'smooth_function_sha':hashlib.sha256(inspect.getsource(u.smooth_knn_dist).encode()).hexdigest(),'membership_function_sha':hashlib.sha256(inspect.getsource(u.compute_membership_strengths).encode()).hexdigest(),'k':16,'columns':16,'self_column':0,'n_iter':64,'local_connectivity':1.,'bandwidth':1.,'smooth_tolerance':float(u.SMOOTH_K_TOLERANCE),'sigma_floor_scale':float(u.MIN_K_DIST_SCALE),'distance_dtype':'float32 stored cosine','calibration':'one full-support reference call; global-mean zero-rho floor never recalibrated by chunks','weights_dtype':'float32','membership':'directed only; retain zero-weight endpoints; no symmetrization'}
def weights(dist,rho,sigma):
 z=dist.astype('f8')-rho[:,None].astype('f8');scale=np.where(sigma==0,1.,sigma).astype('f8')[:,None]
 with np.errstate(over='ignore',under='ignore'):v=np.exp(-np.maximum(z,0)/scale)
 v[(z<=0)|(sigma[:,None]==0)]=1.
 return v.astype('f4')
def scalar(row):
 # Independent scalar float64 bisection, reference float32 psum differs slightly.
 nz=[float(x) for x in row if x>0];rho=nz[0] if nz else 0.;lo=0.;hi=float('inf');mid=1.
 for _ in range(64):
  total=sum(1. if d<=rho else __import__('math').exp(-(float(d)-rho)/mid) for d in row[1:])
  if abs(total-4.)<1e-5:break
  if total>4.:hi=mid;mid=(lo+hi)/2
  else:lo=mid;mid=mid*2 if not np.isfinite(hi) else (lo+hi)/2
 return rho,mid

def controls(u):
 x=np.array([[0]+[.01*j for j in range(1,16)],[0]*16,[0,0,0]+[.3]*13,[0]+[1.]*15],dtype='f4');sig,rho=u.smooth_knn_dist(x,16.,n_iter=64,local_connectivity=1.,bandwidth=1.)
 checks={};errors=[]
 for i,row in enumerate(x):
  rr,ss=scalar(row);ss=max(ss,.001*float(np.mean(row if rr>0 else x)))
  errors.append(abs(float(sig[i])-ss));assert abs(float(rho[i])-rr)<1e-7 and abs(float(sig[i])-ss)<2e-5
 checks['independent_scalar_ties_zero_floor']=True
 ids=np.stack([np.array([i]+[(i+j)%100 for j in range(1,16)]) for i in range(4)]).astype('i8')
 rows,cols,v,_=u.compute_membership_strengths(ids,x,sig,rho,False)
 own=weights(x[:,1:],rho,sig);ref=v.reshape(4,16);assert np.allclose(own,ref[:,1:],atol=2e-6,rtol=2e-6) and (ref[:,0]==0).all();checks['reference_membership_self_zero']=True
 # Offsets affect endpoint IDs, not already-calibrated membership arithmetic.
 for step in [1,2,3]:
  out=np.concatenate([weights(x[j:j+step,1:],rho[j:j+step],sig[j:j+step]) for j in range(0,4,step)])
  assert np.array_equal(out,own)
 checks['chunk_invariance_after_global_calibration']=True
 offset=12345;src=np.repeat(np.arange(offset,offset+4,dtype='i4'),15);assert src[0]==offset and src[-1]==offset+3;assert np.array_equal(weights(x[:,1:],rho,sig),own);checks['global_row_offset_preserved']=True
 return {'PASS':True,'checks':checks,'scalar_sigma_max_error':max(errors)}
def diagnostics(w,n,dst,source):
 w=w.reshape(-1);wf=w.astype('f8');total=float(wf.sum());assert np.isfinite(wf).all() and (wf>=0).all() and total>0
 cdf=np.cumsum(wf);cdf/=total;interval=np.diff(cdf,prepend=0.)
 assert np.isfinite(cdf).all() and (interval>=0).all() and abs(cdf[-1]-1)<1e-12
 lost=(wf>0)&(interval==0);source_mass=wf.reshape(n,15).sum(axis=1)/total;dest_mass=np.bincount(dst,weights=wf,minlength=n)/total
 out={'total_weight':total,'zero_weights':int((wf==0).sum()),'positive_weights':int((wf>0).sum()),'tiny_below_2pow24_count':int(((wf>0)&(wf<2**-24)).sum()),'tiny_below_2pow24_probability':float(wf[wf<2**-24].sum()/total),'cdf_zero_width_positive_count':int(lost.sum()),'cdf_zero_width_positive_probability':float(wf[lost].sum()/total),'cdf_l1_interval_error':float(np.abs(interval-wf/total).sum()),'cdf_terminal':float(cdf[-1]),'weight_quantiles':np.quantile(wf,[0,.01,.1,.5,.9,.99,1]).tolist(),'row_sum_quantiles':np.quantile(source_mass*total,[0,.01,.5,.99,1]).tolist(),'source_probability_quantiles':np.quantile(source_mass,[0,.01,.5,.99,1]).tolist(),'destination_probability_quantiles':np.quantile(dest_mass,[0,.01,.5,.99,1]).tolist(),'source_ESS':float(1/np.square(source_mass).sum()),'destination_ESS':float(1/np.square(dest_mass).sum()),'by_source':{str(s):{'positive_source_mass':float(source_mass[source==s].sum()),'positive_destination_mass':float(dest_mass[source==s].sum())} for s in np.unique(source)},'cpu_cdf_only':'Production CUDA cumsum/searchsorted must independently report numerical lost mass before admission.'}
 bound();return out

def validate_bundle(path=D):
 path=Path(path);m=json.loads((path/'manifest.json').read_text());assert m['PASS'] is True,'bundle lacks PASS'
 assert all(sha(path/f)==h for f,h in m['files'].items()),'card086 data artifact hash mismatch'
 assert all(sha(f)==h for f,h in m['inputs'].items()),'card086 data input hash mismatch'
 assert sha(m['reference']['module_path'])==m['reference']['module_sha'],'card086 reference source mismatch'
 assert sha(__file__)==m['builder_sha'],'card086 builder source mismatch';return m

def main():
 start=time.monotonic();assert not D.exists(),'existing bundle must never be overwritten'
 stage=D.with_name(D.name+'.building');stage.mkdir(parents=True,exist_ok=False)
 u,ref=reference();control=controls(u)
 inputs={str(p):sha(p) for p in [IDX,DIST,G,O/'card018-data-manifest.json',O/'card086-directed-membership.md']}
 old=json.loads((O/'card018-data-manifest.json').read_text());assert inputs[str(IDX)]==old['inputs'][str(IDX)] and inputs[str(DIST)]==old['inputs'][str(DIST)]
 idx=np.load(IDX,mmap_mode='r');dist=np.load(DIST,mmap_mode='r');n=len(idx);assert idx.shape==dist.shape==(2000000,15)
 graph=np.load(G);src=graph['sources'];dst=graph['targets'];assert np.array_equal(dst,idx.ravel()) and np.array_equal(src,np.repeat(np.arange(n,dtype='i4'),15));assert np.isfinite(dist).all() and (dist>=0).all() and (np.diff(dist,axis=1)>=0).all();bound()
 x=np.empty((n,16),dtype='f4');x[:,0]=0;x[:,1:]=dist
 sigma,rho=u.smooth_knn_dist(x,16.,n_iter=64,local_connectivity=1.,bandwidth=1.);assert np.isfinite(sigma).all() and (sigma>=0).all();del x
 np.save(stage/'rho.npy',rho);np.save(stage/'sigma.npy',sigma)
 w=np.lib.format.open_memmap(stage/'membership-weights.npy',mode='w+',dtype='f4',shape=(n,15))
 for lo in range(0,n,32768):
  hi=min(n,lo+32768);w[lo:hi]=weights(dist[lo:hi],rho[lo:hi],sigma[lo:hi]);bound()
 w.flush();source=np.load('/data/latent-basemap/substrates/card018-scale2m/draw_source.npy')
 diag={}
 for arm,aw in [('membership',w.ravel()),('all_one',np.ones(n*15,dtype='f4'))]:
  np.savez(stage/f'{arm}-edges.npz',sources=src,targets=dst,weights=aw,n_nodes=np.int64(n));diag[arm]=diagnostics(aw,n,dst,source);bound()
 diag['rho_sigma']={'rho_zero':int((rho==0).sum()),'sigma_zero':int((sigma==0).sum()),'rho_quantiles':np.quantile(rho,[0,.01,.5,.99,1]).tolist(),'sigma_quantiles':np.quantile(sigma,[0,.01,.5,.99,1]).tolist()}
 write(stage/'diagnostics.json',diag);write(stage/'reference-controls.json',control)
 manifest={'PASS':True,'status':'CPU_BUILT_DEVICE_CDF_AUDIT_PENDING','n':n,'dim':1536,'edges':n*15,'arms':['all_one','membership'],'binary_labels':True,'weighted_sampling':True,'reference':ref,'inputs':inputs,'builder_sha':sha(__file__),'endpoint_sources_values_sha':hashlib.sha256(src.tobytes()).hexdigest(),'endpoint_targets_values_sha':hashlib.sha256(dst.tobytes()).hexdigest(),'files':{p.name:sha(p) for p in stage.iterdir() if p.is_file()},'wall_s':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'threads':2}
 write(stage/'manifest.json',manifest);stage.rename(D);validate_bundle();write(O/'card086-data-ready.json',{'CPU_PASS':True,'GPU_READY':False,'path':str(D),'manifest_sha':sha(D/'manifest.json'),'wall_s':manifest['wall_s'],'peak_rss_mib':manifest['peak_rss_mib'],'reference_controls':control,'diagnostics':diag});print('CPU DATA PASS',manifest['wall_s'],manifest['peak_rss_mib'],flush=True)
if __name__=='__main__':main()
