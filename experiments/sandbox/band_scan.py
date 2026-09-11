"""CPU survey of candidate line-like bridges in matched full-corpus projections.

Geometric triage only: a flagged ridge is not a proven artifact or semantic bridge.
No training, GPU access, feature-space clustering or changes to existing assets.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path('/data/latent-basemap/sandbox/overseer-codex/band-investigation-20260911')
SB = Path('/data/latent-basemap/sandbox')
N = 103816750
POOL_N = 19344847
HEADS = {
    'dino2m': SB/'fullcorpus-dino-2d',
    'dino6m': SB/'fullcorpus-dino-6m-2d',
    'dino6m_pca': SB/'fullcorpus-dino-6m-pca768-2d',
    'dino12m_pca': Path('/data/latent-scope-3d/projections/fullcorpus-dino-12m-pca768-2d-20260910a'),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def member(ids, full_pos):
    x = np.load(full_pos, mmap_mode='r'); pos = np.searchsorted(x, ids)
    return (pos < len(x)) & (x[np.minimum(pos, len(x)-1)] == ids)


def summarize(mask, groups):
    return {str(g): {'n': int((groups == g).sum()), 'fraction': float(mask[groups == g].mean())}
            for g in np.unique(groups)}


def geometry(z):
    n = len(z); tree = cKDTree(z)
    lin32 = np.empty(n); lin96 = np.empty(n); r32 = np.empty(n); balance = np.empty(n)
    axes = np.empty((n, 2)); nearest = np.empty((n, 32), np.int32)
    for s in range(0, n, 2048):
        e = min(s+2048, n); ds, ix = tree.query(z[s:e], k=97, workers=2)
        for j, i in enumerate(range(s,e)):
            keep = ix[j] != i
            # Candidate duplicates in coordinates are valid rows; only identity self removed.
            ids = ix[j][keep][:96]; dd = ds[j][keep][:96]
            assert len(ids) == 96
            nearest[i] = ids[:32]; r32[i] = dd[31]
            for k in (32, 96):
                v = z[ids[:k]].astype(float); v -= v.mean(0)
                w, u = np.linalg.eigh(v.T@v/k)
                linear = 1-w[0]/max(w[1], 1e-30)
                if k == 32:
                    lin32[i] = linear; axes[i] = u[:,1]
                    t = (z[ids[:k]]-z[i])@axes[i]
                    balance[i] = min(float((t>0).mean()), float((t<0).mean()))
                else: lin96[i] = linear
    contrast = np.zeros(n); support = np.zeros(n, bool)
    probe_ids = np.full((n, 2), -1, np.int32)
    eligible = np.flatnonzero((lin32>=.85) & (balance>=.2) & (r32>0))
    for s in range(0, len(eligible), 2048):
        q = eligible[s:s+2048]
        probes = z[q,None,:] + np.array([-4.,4.])[None,:,None]*r32[q,None,None]*axes[q,None,:]
        dd, ii = tree.query(probes.reshape(-1,2), k=1, workers=2)
        dd=dd.reshape(-1,2); ii=ii.reshape(-1,2)
        probe_ids[q]=ii
        support[q]=(dd<=.5*r32[q,None]).all(1)
        contrast[q]=np.min(r32[q,None]/np.maximum(r32[ii],1e-20),axis=1)
    flag=(lin32>=.90)&(lin96>=.80)&(balance>=.2)&support&(contrast>=1.3)
    return dict(linearity32=lin32,linearity96=lin96,r32=r32,balance=balance,
                axis=axes,contrast=contrast,probe_support=support,probe_ids=probe_ids,
                ridge_candidate=flag,neighbors32=nearest)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    seed=9111705; rng=np.random.default_rng(seed)
    ids=np.sort(rng.choice(N,200000,replace=False)); assert len(np.unique(ids))==len(ids)
    src=np.empty(len(ids),dtype='U40')
    shards=json.loads(Path('/data2/monet/pool-complement-88m/full_shards.json').read_text())['shards']
    source_names=np.array([('/'.join(x.split('/')[1:3]) if x.split('/')[1]=='synthetic' else x.split('/')[1]) for x in shards])
    for lo,hi,p in [(0,POOL_N,Path('/data2/monet/pool-20m')),
                    (POOL_N,N,Path('/data2/monet/pool-complement-88m'))]:
        sel=(ids>=lo)&(ids<hi); a=np.load(p/'prov_shard_idx.npy',mmap_mode='r')
        src[sel]=source_names[a[ids[sel]-lo]]
    in6=member(ids,Path('/data2/monet/random-dino-6m/full_pos.npy'))
    in12=member(ids,Path('/data2/monet/random-dino-12m/full_pos.npy'))
    pos6=np.load('/data2/monet/random-dino-6m/full_pos.npy',mmap_mode='r')
    oldmask=np.load('/data2/monet/random-dino-6m/member_mask.npy',mmap_mode='r')
    in2=member_array(ids,np.asarray(pos6[oldmask]))
    contract={'status':'pre-analysis frozen descriptive detector','seed':seed,'n':len(ids),
              'population':N,'sample':'uniform full-corpus row IDs, shared across heads',
              'source_identity':'provenance shard -> full_shards.json path; synthetic/<generator> retained',
              'definition':'linearity=1-small/large covariance eigenvalue on own map kNN; candidate requires k32>=.90,k96>=.80, >=20% neighbors each axis side, probes +/-4*r32 within .5*r32 of a sample, both endpoint r32 <= center r32/1.3',
              'caveat':'Custom exploratory line/bridge detector, not a standard quality metric or semantic classification. Training membership/dose/graph/precision differ across heads; comparisons are operational.',
              'source_code_sha256':sha(Path(__file__)),'heads':{h:str(p) for h,p in HEADS.items()}}
    (ROOT/'survey-contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    np.savez(ROOT/'survey-identities.npz',full_ids=ids,source=src,in2=in2,in6=in6,in12=in12)
    result={'contract':contract,'heads':{}}
    for name,p in HEADS.items():
        t=time.time(); manifest=json.loads((p/'manifest.json').read_text())
        assert manifest['n_rows']==N and manifest['n_pool']==POOL_N
        full=np.load(p/'coords.f32.npy',mmap_mode='r'); assert full.shape==(N,2)
        z=np.array(full[ids],np.float32); assert np.isfinite(z).all()
        if (ROOT/(name+'-geometry.npz')).exists():
            raise RuntimeError('Preserve existing results; use a new output directory for a changed analysis')
        g=geometry(z);np.savez_compressed(ROOT/(name+'-geometry.npz'),coords=z,**g)
        membership=in2 if name=='dino2m' else in12 if name=='dino12m_pca' else in6
        row={'ridge_fraction':float(g['ridge_candidate'].mean()),'ridge_count':int(g['ridge_candidate'].sum()),
             'linearity32_ge_09_fraction':float((g['linearity32']>=.9).mean()),
             'linearity96_ge_09_fraction':float((g['linearity96']>=.9).mean()),
             'by_source':summarize(g['ridge_candidate'],src),
             'by_membership':summarize(g['ridge_candidate'],np.where(membership,'train','unseen')),
             'outside_all_12m_fraction':float(g['ridge_candidate'][~in12].mean()),
             'checkpoint':manifest['checkpoint'],'manifest_sha256':sha(p/'manifest.json'),
             'coordinates_identity':{'path':str(p/'coords.f32.npy'),'bytes':full.nbytes,'mtime_ns':(p/'coords.f32.npy').stat().st_mtime_ns},
             'wall_seconds':time.time()-t}
        result['heads'][name]=row;(ROOT/'survey-result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name,json.dumps({k:v for k,v in row.items() if k not in ['by_source','coordinates_identity']}),flush=True)
    # Matched-ID transition, not alignment-dependent movement.
    a=np.load(ROOT/'dino6m_pca-geometry.npz')['ridge_candidate'];b=np.load(ROOT/'dino12m_pca-geometry.npz')['ridge_candidate']
    result['6pca_to_12pca']={'both':int((a&b).sum()),'6only':int((a&~b).sum()),'12only':int((~a&b).sum()),
                            'neither':int((~a&~b).sum()),'same_population':True}
    (ROOT/'survey-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',flush=True)


def member_array(ids,x):
    pos=np.searchsorted(x,ids)
    return (pos<len(x))&(x[np.minimum(pos,len(x)-1)]==ids)


if __name__=='__main__':main()
