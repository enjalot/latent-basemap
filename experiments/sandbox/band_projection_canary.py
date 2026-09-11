"""CPU reinference and local input sensitivity of the fixed band case panel.

Perturbations are mathematical feature probes, not newly observed images.
Use frozen pre-pilot core so concurrent training implementation cannot affect this audit.
"""
import sys,json,time
from pathlib import Path
import numpy as np
import torch
from band_scan import ROOT,HEADS,SB,POOL_N,sha
sys.path.insert(0,str(SB/'overseer-codex/card009-code-e173e3b'))
from basemap.pumap.parametric_umap.core import ParametricUMAP

def raw_features(ids):
    ids=np.asarray(ids);out=np.empty((len(ids),1536),np.float32)
    for lo,hi,name in [(0,POOL_N,'pool-20m'),(POOL_N,103816750,'pool-complement-88m')]:
        m=np.load(Path('/data2/monet')/name/'dino1536.f16.npy',mmap_mode='r');ix=np.flatnonzero((ids>=lo)&(ids<hi));out[ix]=m[ids[ix]-lo]
    return out

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);started=time.time()
    cases=json.loads((ROOT/'forensics-result.json').read_text())['cases'];identity=np.load(ROOT/'survey-identities.npz');gids=identity['full_ids']
    detail=np.load(ROOT/'detail-neighbors.npz');neighbors=dict(zip(detail['query_survey_indices'],detail['neighbor_survey_indices']))
    # Query plus first endpoint A/B point in every case, before sensitivity is measured.
    panel=[]
    for n,c in enumerate(cases):
        for role,q in [('center',c['survey_index']),('endpoint_A',c['endpoint_indices'][0][0]),('endpoint_B',c['endpoint_indices'][1][0])]:
            panel.append({'case':n+1,'role':role,'survey_index':int(q),'full_id':int(gids[q]),'neighbor_full_id':int(gids[neighbors[q][0]])})
    X=raw_features([r['full_id'] for r in panel]);Y=raw_features([r['neighbor_full_id'] for r in panel])
    rng=np.random.default_rng(9111725);random=rng.normal(size=X.shape).astype(np.float32);random/=np.linalg.norm(random,axis=1,keepdims=True)
    distance=np.linalg.norm(Y-X,axis=1,keepdims=True);direction=Y-X
    variants=[X,X+.05*direction,X-.05*direction,X+.05*distance*random,X-.05*distance*random]
    pca=np.load('/data2/monet/random-dino-6m/pca768-model.npz');mean=torch.tensor(pca['mean']);comp=torch.tensor(pca['components'])
    result={'status':'exploratory fixed-panel CPU audit','panel':panel,'heads':{},'input_probe':'+/-5% of nearest REAL reference-neighbor raw-feature distance; toward-neighbor and matched-length random directions; PCA heads apply original saved PCA+L2. Full heads consume raw fp16->fp32 as their saved projection did. Perturbed vectors need not correspond to real images.', 'source_sha256':sha(Path(__file__))}
    for h,path in HEADS.items():
        manifest=json.loads((path/'manifest.json').read_text());ckpt=Path(manifest['checkpoint']);model=ParametricUMAP.load(str(ckpt),device='cpu').model.eval()
        def process(x):
            tx=torch.from_numpy(x)
            return torch.nn.functional.normalize((tx-mean)@comp,dim=1) if 'pca' in h else tx
        with torch.inference_mode():
            inp=process(X);z=model(inp).numpy();z1=np.concatenate([model(inp[i:i+1]).numpy() for i in range(len(inp))])
            got=np.stack([model(process(v)).numpy() for v in variants])
        saved=np.load(path/'coords.f32.npy',mmap_mode='r')[[r['full_id'] for r in panel]]
        error=float(np.max(np.abs(z-saved)));batch_error=float(np.max(np.abs(z-z1)))
        assert np.isfinite(z).all() and np.isfinite(z1).all()
        # Report the original 1e-4 diagnostic threshold honestly; do not abort later heads on a tiny cross-device deviation.
        geom=np.load(ROOT/(h+'-geometry.npz'));r32=geom['r32'][[r['survey_index'] for r in panel]]
        shifts=np.linalg.norm(got[1:]-got[0],axis=2);relative=shifts/np.maximum(r32,1e-12)
        result['heads'][h]={'checkpoint_sha256':sha(ckpt),'stored_max_abs_error':error,'original_1e4_threshold_pass':bool(error<1e-4),'max_saved_vector_error_over_r32':float(np.max(np.linalg.norm(z-saved,axis=1)/np.maximum(r32,1e-12))),'batch_1_vs_36_max_abs_error':batch_error,'raw_feature_norm_min':float(np.linalg.norm(X,axis=1).min()),'raw_feature_norm_max':float(np.linalg.norm(X,axis=1).max()),'probe_shift_over_r32':relative.T.tolist(),'median_by_role':{role:float(np.median(relative[:,[i for i,r in enumerate(panel) if r['role']==role]])) for role in ['center','endpoint_A','endpoint_B']}}
        print(h,error,batch_error,flush=True)
    result['wall_seconds']=time.time()-started
    (ROOT/'projection-canary.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',flush=True)

if __name__=='__main__':main()
