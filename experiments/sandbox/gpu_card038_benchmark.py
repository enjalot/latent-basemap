"""Actual strictFP32 warm projection + host-copy benchmark. Disk input excluded and disclosed."""
import sys,time,json,copy,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));from _paths import ensure_paths;ensure_paths()
import numpy as np,torch
import card038_validate as V
from basemap.pumap.parametric_umap.core import ParametricUMAP
R=Path(__file__).resolve().parents[2];O=V.OC;SEAL=Path('/data2/monet/eval-common-v2');BATCH=8192
@torch.inference_mode()
def forward(m,x):
 y=[]
 for start in range(0,len(x),BATCH):y.append(m(x[start:start+BATCH]).cpu().numpy())
 out=np.concatenate(y);assert out.shape==(len(x),3) and np.isfinite(out).all();return out

def main():
 assert V.runtime_manifest_check(R)[0];receipts={a:V.strict_validate_arm(a,R) for a in V.ARMS};torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
 x=np.array(np.load(SEAL/'ref_hd.f16.npy',mmap_mode='r'),'f4',copy=True);assert x.shape==(250000,1536);x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12);xt=torch.from_numpy(x).cuda();models={a:ParametricUMAP.load(str(V.TD_DEFAULT/f'model-{a}.pt'),device='cuda').model.float().eval() for a in V.ARMS};checks={};fidelity={}
 for a,m in models.items():
  gpu=forward(m,xt[:8192])[:1024];cpu=copy.deepcopy(m).cpu().eval()
  with torch.inference_mode():f32=cpu(torch.from_numpy(x[:1024])).numpy();f64=cpu.double()(torch.from_numpy(x[:1024]).double()).numpy()
  bound=2e-4+2e-5*np.abs(f64);checks[a+'_cpu_GPU_fidelity']=bool((np.abs(gpu-f64)<=bound).all());checks[a+'_cpu32_64_fidelity']=bool((np.abs(f32-f64)<=bound).all());fidelity[a]={'GPU_vs_CPU64_max':float(np.abs(gpu-f64).max()),'CPU32_vs_CPU64_max':float(np.abs(f32-f64).max()),'bound':'componentwise2e-4+2e-5*abs(CPU64)'};assert checks[a+'_cpu_GPU_fidelity'] and checks[a+'_cpu32_64_fidelity'];del cpu
  forward(m,xt) # one full warmup, discarded
 times={a:[] for a in V.ARMS};order=[]
 for rep in range(5):
  names=V.ARMS if rep%2==0 else V.ARMS[::-1]
  for a in names:
   torch.cuda.synchronize();start=time.perf_counter();y=forward(models[a],xt);torch.cuda.synchronize();elapsed=time.perf_counter()-start;assert elapsed>0;times[a].append(elapsed);order.append({'repeat':rep,'arm':a,'seconds':elapsed});del y
 arms={a:{'pass_seconds':times[a],'rows_per_s_median':float(np.median(250000/np.array(times[a]))),'model_sha256':V.full_sha(V.TD_DEFAULT/f'model-{a}.pt'),'parameters':sum(t.numel() for t in models[a].parameters()),'fp32_parameter_bytes':sum(t.numel()*4 for t in models[a].parameters())} for a in V.ARMS};speed=arms['compact1024']['rows_per_s_median']/arms['wide2048']['rows_per_s_median'];free,total=torch.cuda.mem_get_info();checks['global_VRAM_lt30']=(total-free)/2**30<30
 result={'PASS':bool(all(checks.values())),'checks':checks,'arms':arms,'speedup_compact_over_wide':speed,'pass_order':order,'fidelity':fidelity,'precision':'float32','TF32':False,'batch_size':BATCH,'n_reference':250000,'reference_input_sha':V.full_sha(SEAL/'ref_hd.f16.npy'),'reference_ids_sha':V.full_sha(SEAL/'ref_idx.npy'),'process_peak_gib':torch.cuda.max_memory_allocated()/2**30,'global_used_gib':(total-free)/2**30,'runtime_manifest_sha':V.full_sha(V.runtime_manifest_path(R)),'scope':'Warm resident GPU input, FP32 model forward plus every output copied to CPU; normalization, model load and disk read excluded. PASS is engineering validity; speed<2 is honest nonpromotion, not invalid benchmarking.'};(O/'card038-benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['PASS']
if __name__=='__main__':main()
