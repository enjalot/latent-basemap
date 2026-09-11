"""Matched-point encoder-neighbor, dimensionality and qualitative band diagnostics.

CPU only; exact neighbors against a fixed 200K uniform corpus reference.
Findings are exploratory and reference-size-specific, not global semantic labels.
"""
import json
import time
from pathlib import Path

import numpy as np
import faiss
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from band_scan import ROOT, SB, POOL_N, sha


def features(ids):
    ids=np.asarray(ids); out=np.empty((len(ids),1536),np.float32)
    for lo,hi,p in [(0,POOL_N,'pool-20m'),(POOL_N,103816750,'pool-complement-88m')]:
        mm=np.load(Path('/data2/monet')/p/'dino1536.f16.npy',mmap_mode='r');sel=np.flatnonzero((ids>=lo)&(ids<hi))
        for s in range(0,len(sel),2048):
            ix=sel[s:s+2048];out[ix]=mm[ids[ix]-lo]
    assert np.isfinite(out).all()
    out/=np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-20)
    return out


def mean_pair_scale(z):
    delta=z[:,None,:]-z[None,:,:]
    return float(np.sqrt(np.mean(np.sum(delta*delta,axis=-1))))


def main():
    faiss.omp_set_num_threads(3); started=time.time()
    identity=np.load(ROOT/'survey-identities.npz'); gids=identity['full_ids'];sources=identity['source']
    cases=json.loads((ROOT/'candidate-selection.json').read_text())
    geoms={h:np.load(ROOT/(h+'-geometry.npz')) for h in ['dino2m','dino6m','dino6m_pca','dino12m_pca']}
    trees={h:cKDTree(a['coords']) for h,a in geoms.items()}
    # Three disjoint spatial samples: center, and the two denser endpoint probes.
    allq=set()
    for c in cases:
        h=c['head'];i=c['survey_index'];a=geoms[h];z=a['coords'];ends=a['probe_ids'][i]
        center=a['neighbors32'][i][:16].astype(int).tolist()
        center=[i]+[j for j in center if j!=i][:15]
        protos=[]
        for j in ends:
            _,ix=trees[h].query(z[j],k=64)
            ix=[int(v) for v in ix if int(v) not in center and (not protos or int(v) not in protos[0])][:16]
            assert len(ix)==16;protos.append(ix)
        c['center_indices']=center;c['endpoint_indices']=protos
        allq.update(center);allq.update(protos[0]);allq.update(protos[1])
    qids=np.array(sorted(allq));assert len(qids)<1000
    X=features(gids);index=faiss.IndexFlatIP(1536);index.add(X)
    sim,nn=index.search(X[qids],65)
    # identity self exclusion, no assumed rank under equal similarities.
    neighbors={int(q):ix[ix!=q][:64] for q,ix in zip(qids,nn)}
    np.savez_compressed(ROOT/'detail-neighbors.npz',query_survey_indices=qids,
                        query_full_ids=gids[qids],neighbor_survey_indices=np.stack([neighbors[int(i)] for i in qids]),
                        reference_full_ids=gids)
    z3={}
    for h,p in [('dino6m_pca','fullcorpus-dino-6m-pca768-3d'),('dino12m_pca','fullcorpus-dino-12m-pca768-3d')]:
        m=np.load(SB/p/'coords.f32.npy',mmap_mode='r');assert m.shape==(103816750,3);z3[h]=np.array(m[gids])
    out=[]
    for c in cases:
        h=c['head'];i=c['survey_index'];band=c['center_indices'];A,B=map(np.array,c['endpoint_indices']);a=geoms[h];z=a['coords']
        # Space-defined endpoint prototypes; semantic labels require human inspection.
        protoA=X[A].mean(0);protoA/=np.linalg.norm(protoA);protoB=X[B].mean(0);protoB/=np.linalg.norm(protoB)
        ddAB=1-X[A]@X[B].T;ddAA=1-X[A]@X[A].T;ddBB=1-X[B]@X[B].T
        within=float((ddAA.sum()+ddBB.sum())/(2*16*15))
        core_sep=float(ddAB.mean()/max(within,1e-12))
        margin=(X[band]@protoB-X[band]@protoA)
        # Equal-prior soft affinity; temperature is fixed diagnostic .05, not a calibrated probability.
        logits=margin/.05;prob=1/(1+np.exp(-np.clip(logits,-40,40)))
        entropy=-(prob*np.log2(np.maximum(prob,1e-12))+(1-prob)*np.log2(np.maximum(1-prob,1e-12)))
        t=(z[band]-z[i])@a['axis'][i]
        per_head={}
        for hh,aa in geoms.items():
            zz=aa['coords'];_,mnn=trees[hh].query(zz[band],k=2001,workers=2)
            rec=[];offset=[];ns=[];budgets={b:[] for b in [15,50,250,2000]}
            for q,idx in zip(band,mnn):
                hd=neighbors[q][:15];lo=[int(v) for v in idx if v!=q][:250]
                rec.append(len(set(lo)&set(map(int,hd)))/15)
                for budget in budgets:
                    clean=[int(v) for v in idx if v!=q][:budget]
                    assert len(clean)==budget
                    budgets[budget].append(len(set(clean)&set(map(int,hd)))/15)
                scale=mean_pair_scale(zz[hd]);offset.append(float(np.linalg.norm(zz[q]-zz[hd].mean(0))/max(scale,1e-12)))
                ns.append(scale)
            per_head[hh]={'B250_k15':float(np.mean(rec)),'neighbor_centroid_offset_over_neighbor_pair_rms':float(np.mean(offset)),
                          'linearity32_median':float(np.median(aa['linearity32'][band])),
                          'ridge_candidate_fraction':float(aa['ridge_candidate'][band].mean()),
                          'query_B250':rec,'query_offset':offset,'budget_recall':{str(b):float(np.mean(v)) for b,v in budgets.items()}}
        for hh,zz in z3.items():
            _,mnn=cKDTree(zz).query(zz[band],k=2001,workers=2);rec=[]
            for q,idx in zip(band,mnn):rec.append(len(set([int(v) for v in idx if v!=q][:250])&set(map(int,neighbors[q][:15])))/15)
            per_head[hh+'_3d']={'B250_k15':float(np.mean(rec)),'query_B250':rec}
        c=dict(c);c['full_ids_by_part']={'center':gids[band].tolist(),'endpoint_A':gids[A].tolist(),'endpoint_B':gids[B].tolist()}
        c['encoder_neighbors_of_query']=gids[neighbors[i][:12]].tolist()
        c['encoder_core_between_over_within_cosine_distance']=core_sep
        c['encoder_core_mean_cross_cosine']=float(1-ddAB.mean())
        c['endpoint_soft_affinity_entropy_mean']=float(entropy.mean())
        c['endpoint_soft_affinity_margin']=margin.tolist()
        c['axis_affinity_spearman']=float(spearmanr(t,margin).statistic)
        c['comparisons']=per_head
        out.append(c)
    record={'status':'exploratory CPU diagnostics; not a causal or semantic classification',
            'reference':'same 200K uniform full-corpus rows; exact normalized full-D DINO inner product; self excluded',
            'source_sha256':sha(Path(__file__)),'identity_sha256':sha(ROOT/'survey-identities.npz'),
            'prototype_caveat':'A/B are nearby map-probe groups, not certified distinct semantic clusters. Affinity entropy uses arbitrary fixed temperature .05, not calibrated uncertainty.',
            'cases':out,'wall_seconds':time.time()-started}
    (ROOT/'forensics-result.json').write_text(json.dumps(record,indent=2)+'\n')
    print('DONE',len(out),'cases',len(qids),'queries',time.time()-started,flush=True)


if __name__=='__main__':main()
