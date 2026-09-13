"""CPU fresh-head scale calibration; no learned target or model-quality selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,time,resource
import numpy as np,torch
import card065_common as C
sys.path.insert(0,str(C.R));from basemap.pumap.parametric_umap.core import ParametricUMAP
def main():
 start=time.monotonic();torch.set_num_threads(2);C.input_check();assert not (C.D/'scale-manifest.json').exists();init=torch.load(C.D/'fresh-init.pt',map_location='cpu',weights_only=False);assert init['init_state_sha256']=='5544a31160054bcc'
 m=ParametricUMAP.load(str(C.CHAMP),device='cpu').model.float().eval().requires_grad_(False);m.load_state_dict(init['model_state'],strict=True);assert C.state_sha(m.state_dict()).startswith('5544a31160054bcc')
 x=np.load(C.D/'train.f16.npy',mmap_mode='r');sel=np.load(C.D/'supervised-selection.npz');train=sel['train_rows'];out=[]
 with torch.inference_mode():
  for lo in range(0,len(train),4096):out.append(m(torch.from_numpy(np.array(x[train[lo:lo+4096]],dtype='f4'))).numpy())
 y=np.concatenate(out).astype('f8');center=y.mean(0);rms=float(np.sqrt(np.mean(np.sum((y-center)**2,axis=1))));assert np.isfinite(rms) and rms>1e-8
 pca=np.load(C.D/'pca-unscaled.f64.npy');pm=pca[train].mean(0);pr=float(np.sqrt(np.mean(np.sum((pca[train]-pm)**2,axis=1))));target=((pca-pm)*rms/pr).astype('f4');assert np.isfinite(target).all();np.save(C.D/'pca-targets.f32.npy',target);np.save(C.D/'random-train-predictions.f32.npy',y.astype('f4'))
 C.write(C.D/'scale-manifest.json',{'status':'TRAIN_ONLY_SHARED_RMS_READY','original_init_state_sha':C.state_sha(m.state_dict()),'shared_RMS':rms,'random_center':center.tolist(),'pca_center':pm.tolist(),'pca_unscaled_RMS':pr,'target_sha':C.sha(C.D/'pca-targets.f32.npy'),'random_predictions_sha':C.sha(C.D/'random-train-predictions.f32.npy'),'input_manifest_sha':C.sha(C.D/'inputs-manifest.json'),'loaded_modules':C.loaded_modules(),'source_sha':C.sha(__file__),'cpu_s':time.monotonic()-start,'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'scope':'Only original300K training graph used; no supervised or graph training and no quality outcomes'});assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<7500
 print('sharedRMS',rms,'seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
