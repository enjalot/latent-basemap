"""Extract real corridor rows and compare encoder support with matched ordinary rows.

Exploratory CPU-only follow-up; protocol frozen in support-v2/design.json.
Map-selected seeds expanded in encoder space are not independent semantic clusters.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
import time
from pathlib import Path
import numpy as np
import faiss
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from band_scan import ROOT, HEADS, N, POOL_N, member, member_array, sha
from band_forensics import features

OUT = ROOT / 'support-v2'
CASES = (3, 4, 10)
SEED = 912003


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sources(ids):
    shards = json.loads(Path('/data2/monet/pool-complement-88m/full_shards.json').read_text())['shards']
    names = np.array([('/'.join(x.split('/')[1:3]) if x.split('/')[1] == 'synthetic' else x.split('/')[1]) for x in shards])
    result = np.empty(len(ids), dtype='U40')
    for offset, end, folder in [(0, POOL_N, 'pool-20m'), (POOL_N, N, 'pool-complement-88m')]:
        ii = np.flatnonzero((ids >= offset) & (ids < end))
        shard = np.load(Path('/data2/monet') / folder / 'prov_shard_idx.npy', mmap_mode='r')
        result[ii] = names[shard[ids[ii] - offset]]
    return result


def extract():
    survey = np.load(ROOT / 'survey-identities.npz'); gids = survey['full_ids']
    cases = json.loads((ROOT / 'forensics-result.json').read_text())['cases']
    rng = np.random.default_rng(SEED)
    pos6 = np.load('/data2/monet/random-dino-6m/full_pos.npy', mmap_mode='r')
    pos2 = np.asarray(pos6[np.load('/data2/monet/random-dino-6m/member_mask.npy', mmap_mode='r')])
    records = []; arrays = {}
    for number in CASES:
        c = cases[number-1]; h = c['head']; geo = np.load(ROOT / (h+'-geometry.npz'))
        i = c['survey_index']; center = geo['coords'][i]; axis = geo['axis'][i]; radius = geo['r32'][i]
        perp = np.array([-axis[1], axis[0]]); mm = np.load(HEADS[h] / 'coords.f32.npy', mmap_mode='r')
        chunks = []
        for start in range(0, N, 500000):
            delta = np.asarray(mm[start:start+500000]) - center
            along = delta @ axis; across = delta @ perp
            ix = np.flatnonzero((abs(along) <= 3*radius) & (abs(across) <= .5*radius))
            chunks.append(ix + start)
        all_ids = np.concatenate(chunks); np.save(OUT/f'case-{number}-corridor-ids.npy', all_ids)
        eligible = all_ids[~np.isin(all_ids, gids)]
        assert len(eligible) >= 48
        band = np.sort(rng.choice(eligible, 48, replace=False)); bsrc = sources(band)
        member_key = 'in2' if h == 'dino2m' else 'in12' if h == 'dino12m_pca' else 'in6'
        bm = member_array(band, pos2) if member_key == 'in2' else member(band, Path('/data2/monet')/('random-dino-12m' if member_key == 'in12' else 'random-dino-6m')/'full_pos.npy')
        tree = cKDTree(geo['coords']); bxy = np.asarray(mm[band]); br = tree.query(bxy, k=32, workers=2)[0][:, -1]
        ordinary = (geo['linearity32'] < .8) & (np.linalg.norm(geo['coords']-center, axis=1) > 8*radius)
        used = set(); control = []; density_ratio = []
        for src, trained, r in zip(bsrc, bm, br):
            candidates = np.flatnonzero(ordinary & (survey['source'] == src) & (survey[member_key] == trained))
            candidates = np.array([j for j in candidates if int(j) not in used], dtype=int)
            assert len(candidates), (number, src, bool(trained))
            j = int(candidates[np.argmin(abs(np.log(np.maximum(geo['r32'][candidates],1e-12)/max(r,1e-12))))])
            used.add(j); control.append(gids[j]); density_ratio.append(float(geo['r32'][j]/r))
        control = np.array(control); assert len(np.unique(control)) == 48
        arrays[f'case{number}_band'] = band; arrays[f'case{number}_control'] = control
        arrays[f'case{number}_axis_position'] = (bxy-center) @ axis / radius
        records.append({'case':number,'selecting_head':h,'corridor_count':len(all_ids),'eligible_non_survey':len(eligible),'n_pairs':48,'source_counts':{str(s):int((bsrc==s).sum()) for s in np.unique(bsrc)},'training_member_count':int(bm.sum()),'control_to_band_r32_ratio':density_ratio,'center':center.tolist(),'axis':axis.tolist(),'r32':float(radius)})
        print('EXTRACT', number, len(all_ids), 'controls median r32 ratio', np.median(density_ratio), flush=True)
    np.savez(OUT/'cohorts.npz', **arrays); write('extraction.json', {'cases':records,'code_sha256':sha(Path(__file__))})


def neighbors(index, Xq, qids, rids, k):
    sim, idx = index.search(np.ascontiguousarray(Xq), k+1)
    clean = [row[rids[row] != q][:k] for row,q in zip(idx,qids)]
    assert all(len(row)==k for row in clean)
    return np.stack(clean)


def score():
    faiss.omp_set_num_threads(2)
    cases = json.loads((ROOT/'forensics-result.json').read_text())['cases']; cohort = np.load(OUT/'cohorts.npz')
    survey = np.load(ROOT/'survey-identities.npz'); rids_all = survey['full_ids']
    query_ids = np.concatenate([cohort[f'case{n}_{part}'] for n in CASES for part in ('band','control')])
    Xq = features(query_ids); Xref = features(rids_all)
    # HD medoid of each original probe group; expansions below use encoder space only.
    seed_ids=[]
    for n in CASES:
        for key in ('endpoint_A','endpoint_B'):
            ids=np.array(cases[n-1]['full_ids_by_part'][key]); x=features(ids)
            seed_ids.append(int(ids[np.argmax((x@x.T).mean(1))]))
    Xseed=features(np.array(seed_ids)); results={}; saved={'query_ids':query_ids,'endpoint_seed_ids':np.array(seed_ids),'reference_ids_200K':rids_all}
    small=np.sort(np.random.default_rng(SEED+1).choice(len(rids_all),50000,replace=False))
    paths={**{h:p/'coords.f32.npy' for h,p in HEADS.items()},'dino6m_pca_3d':Path('/data/latent-basemap/sandbox/fullcorpus-dino-6m-pca768-3d/coords.f32.npy'),'dino12m_pca_3d':Path('/data/latent-basemap/sandbox/fullcorpus-dino-12m-pca768-3d/coords.f32.npy')}
    for size,sel in [(50000,small),(200000,np.arange(len(rids_all)))]:
        rids=rids_all[sel]; xr=Xref[sel]; index=faiss.IndexFlatIP(1536); index.add(xr)
        hd=neighbors(index,Xq,query_ids,rids,60); end=neighbors(index,Xseed,np.array(seed_ids),rids,256)
        d60=np.sqrt(np.maximum(0,2-2*np.sum(Xq*xr[hd[:,-1]],axis=1)))
        saved[f'hd60_{size}']=rids[hd]; saved[f'endpoint_support_{size}']=rids[end]; saved[f'reference_ids_{size}']=rids
        rr={'cases':{},'heads':{}}
        for ci,n in enumerate(CASES):
            qslice=slice(ci*96,ci*96+48); A,B=end[2*ci:2*ci+2]
            ah=np.isin(hd[qslice,:15],A).mean(1); bh=np.isin(hd[qslice,:15],B).mean(1)
            am=np.isin(hd[qslice],A); bm=np.isin(hd[qslice],B)
            support_partition={'A_only':float((am&~bm).mean()),'B_only':float((bm&~am).mean()),'both':float((am&bm).mean()),'neither':float((~am&~bm).mean())}
            # Both endpoints can be equally irrelevant. Distances distinguish that
            # from actual neighborhood support; affinity alone cannot do so.
            endpoint_distances={}
            for label,group in [('A',A),('B',B)]:
                dd=np.sqrt(np.maximum(0,2-2*Xq[qslice]@xr[group].T))
                scaled=np.sort(dd,axis=1)[:,:5].mean(1)/np.maximum(d60[qslice],1e-12)
                endpoint_distances[label]=float(scaled.mean())
                saved[f'case{n}_endpoint_{label}_distance_ratio_{size}']=scaled
            pa=xr[A].mean(0);pa/=np.linalg.norm(pa);pb=xr[B].mean(0);pb/=np.linalg.norm(pb)
            margin=Xq[qslice]@(pb-pa); rho=spearmanr(cohort[f'case{n}_axis_position'],margin).statistic
            rr['cases'][str(n)]={'endpoint_support_jaccard':float(len(set(A)&set(B))/len(set(A)|set(B))),'HD60_support_partition':support_partition,'nearest5_endpoint_distance_over_query_d60':endpoint_distances,'mean_HD15_fraction_endpoint_A':float(ah.mean()),'mean_HD15_fraction_endpoint_B':float(bh.mean()),'fraction_with_HD15_neighbors_in_both':float(((ah>0)&(bh>0)).mean()),'axis_affinity_spearman':float(rho) if np.isfinite(rho) else None}
            saved[f'case{n}_margin_{size}']=margin; saved[f'case{n}_support_A_{size}']=ah; saved[f'case{n}_support_B_{size}']=bh
        for h,path in paths.items():
            mm=np.load(path,mmap_mode='r'); zref=np.asarray(mm[rids]); zq=np.asarray(mm[query_ids]); tree=cKDTree(zref)
            _,nn=tree.query(zq,k=251,workers=2)
            nn=np.stack([row[rids[row]!=q][:250] for row,q in zip(nn,query_ids)])
            map15=nn[:,:15]; outside=~(map15[:,:,None]==hd[:,None,:]).any(2)
            edge_dist=np.sqrt(np.maximum(0,2-2*np.einsum('qd,qkd->qk',Xq,xr[map15])))
            ratio=edge_dist/np.maximum(d60[:,None],1e-12)
            metrics={'outside_hd60':outside.mean(1),'unsupported_ratio_gt125':(outside&(ratio>1.25)).mean(1),'encoder_distance_ratio_p90':np.quantile(ratio,.9,axis=1)}
            for b in (15,63,250):metrics[f'B{b}']=(nn[:,:b,None]==hd[:,None,:15]).any(1).mean(1)
            rh={}
            for ci,n in enumerate(CASES):
                rh[str(n)]={}
                for metric,v in metrics.items():
                    band=v[ci*96:ci*96+48]; control=v[ci*96+48:ci*96+96]
                    rh[str(n)][metric]={'band_mean':float(band.mean()),'control_mean':float(control.mean()),'paired_delta_mean':float((band-control).mean())}
            rr['heads'][h]=rh
            for metric,v in metrics.items():saved[f'{h}_{size}_{metric}']=v
            saved[f'{h}_{size}_map15']=rids[map15]
            print('SCORE',size,h,flush=True)
        results[str(size)]=rr; del index,xr
    np.savez_compressed(OUT/'per-query.npz',**saved)
    write('results.json',{'status':'exploratory matched controls; no population or causal inference','n_pairs_per_case':48,'references':results,'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'design.json'),'cohort_sha256':sha(OUT/'cohorts.npz')})


if __name__=='__main__':
    started=time.time()
    if not (OUT/'cohorts.npz').exists(): extract()
    score()
    print('DONE seconds',time.time()-started,flush=True)
