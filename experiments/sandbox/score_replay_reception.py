"""Cards 006/007 reception scorer (CPU, off-flock) — deployment criteria 2 & 3, frozen thresholds.
Per-query recall on the original sealed instrument (eval_jina_pair.per_query) for heads:
frozen (T0/S0), original anchored (card005/004), IN replay, OUT replay. Reports, per group and
equal-cohort aggregate, at B250/B2000:
  Criterion 2 (old-quality retention): equal-cohort mean recall LOSS vs frozen <= .005 and worst
    cohort loss <= .01 at BOTH budgets (Vietnamese called out for Jina).
  Criterion 3 (arriving quality): arriving aggregate no worse than ORIGINAL anchored by .005 at
    either budget, and positive B250 gain over frozen with paired-bootstrap CI > 0 (Jina also
    requires original Chinese gain >= .03).
Evaluates fixed criteria using unrounded values and persists query IDs, recall,
cross-group hits/denominators, weighted summaries and paired uncertainty. Env CARD=card006|card007. Usage: score_replay_reception.py
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import eval_jina_pair as EP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
CARD = os.environ.get("CARD", "card006")

CFG = {
 "card006": {"seal": Path("/data2/monet/eval-common-v2"), "grp_file": "val_source.npy",
             "outd": SB / "dino-arrival-t0/replay-updates",
             "heads": {"frozen": SB / "dino-arrival-t0/champion-bs16k/model.pt",
                       "anchored": SB / "dino-arrival-t0/updates/model-anchored.pt",
                       "unanchored": SB / "dino-arrival-t0/updates/model-unanchored.pt"},
             "arriving": ["synthetic-flux-klein", "synthetic-flux-schnell", "synthetic-z-image"],
             "old": ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"],
             "exclude": ["diffusion-aesthetic-4k"], "chinese": None},
 "card007": {"seal": Path("/data2/monet/eval-common-multilingual"), "grp_file": "val_cohort.npy",
             "outd": SB / "jina-ladder-2m-s0/replay-updates",
             "heads": {"frozen": SB / "jina-ladder-2m-s0/champion-bs16k/model.pt",
                       "anchored": SB / "jina-ladder-2m-s0/updates/model-anchored.pt",
                       "unanchored": SB / "jina-ladder-2m-s0/updates/model-unanchored.pt"},
             "arriving": ["cmn_Hani", "ml-cmn_Hani"], "old": None, "exclude": [], "chinese": ["cmn_Hani", "ml-cmn_Hani"]},
}[CARD]


def paired_ci(delta, seed=0):
    d = np.asarray(delta, np.float64)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError('Invalid paired query deltas')
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return {'mean': float(d.mean()), 'ci95': np.percentile(bs, [2.5, 97.5]).tolist()}


def quality_gates(pq, grp, arriving, old, chinese=False):
    if not arriving or not old or not set(arriving + old) <= set(grp):
        raise ValueError('Missing required arriving/old cohorts')
    for head in ('frozen', 'anchored', 'in', 'out'):
        if head not in pq: raise ValueError(f'Missing required head {head}')
        for B in EP.BUDGETS:
            x = pq[head][B]
            if x.shape != grp.shape or not np.isfinite(x).all():
                raise ValueError(f'Invalid metric array {head}/{B}')
    counts = [int((grp == g).sum()) for g in arriving]
    if len(set(counts)) != 1: raise ValueError('Arriving bootstrap assumes balanced cohorts')
    crit2, crit3 = {}, {}
    for tag in ('in', 'out'):
        c2, c3 = {}, {}
        for B in EP.BUDGETS:
            losses = {g: float((pq['frozen'][B] - pq[tag][B])[grp == g].mean()) for g in old}
            c2[f'B{B}'] = {'mean_loss': float(np.mean(list(losses.values()))),
                           'worst_loss': max(losses.values()), 'worst_group': max(losses, key=losses.get),
                           'per_group_loss': losses}
            am = np.isin(grp, arriving)
            v, anc = float(pq[tag][B][am].mean()), float(pq['anchored'][B][am].mean())
            c3[f'B{B}'] = {'arriving_agg': v, 'anchored_agg': anc,
                           'no_worse_than_anchored_by.005': bool(v >= anc - .005)}
        c2['pass'] = bool(all(c2[f'B{B}']['mean_loss'] <= .005 and c2[f'B{B}']['worst_loss'] <= .01 for B in EP.BUDGETS))
        gain = paired_ci((pq[tag][250] - pq['frozen'][250])[am])
        c3.update(arriving_B250_gain=gain['mean'], arriving_B250_ci95=gain['ci95'],
                  positive_gain_ci_above_0=bool(gain['mean'] > 0 and gain['ci95'][0] > 0))
        if chinese: c3['chinese_gain_ge.03'] = bool(gain['mean'] >= .03)
        c3['pass'] = bool(c3['positive_gain_ci_above_0'] and all(c3[f'B{B}']['no_worse_than_anchored_by.005'] for B in EP.BUDGETS)
                          and (not chinese or c3['chinese_gain_ge.03']))
        crit2[tag], crit3[tag] = c2, c3
    return crit2, crit3


def main():
    import hashlib, torch, faiss
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    torch.set_num_threads(threads); faiss.omp_set_num_threads(threads)
    seal = CFG['seal']
    ref_hd = np.load(seal/'ref_hd.f16.npy', mmap_mode='r')
    val_hd = np.load(seal/'val_hd.f16.npy', mmap_mode='r')
    truth = np.load(seal/'truth_val.npy')
    grp = np.load(seal/CFG['grp_file'], allow_pickle=True).astype(str)
    val_ids, ref_ids = np.load(seal/'val_idx.npy'), np.load(seal/'ref_idx.npy')
    groups = sorted(set(grp.tolist()))
    arriving = [g for g in groups if g in set(CFG['arriving'])]
    old = CFG['old'] or [g for g in groups if g not in CFG['arriving'] and g not in CFG['exclude']]
    if not arriving or not old: raise ValueError('Required cohort absent')
    heads = dict(CFG['heads'])
    heads['fresh'] = SB/('dino-arrival-final/champion-bs16k/model.pt' if CARD == 'card006' else 'jina-ladder-2m-proportional/champion-bs16k/model.pt')
    heads.update({t: CFG['outd']/f'model-{t}.pt' for t in ('in','out')})
    for p in heads.values():
        if not p.exists(): raise FileNotFoundError(p)
    # Reuse already-reviewed fixed-seal total-recall arrays. New cross-group
    # diagnostics use cached model coordinates, not another total-recall pass.
    prior = np.load(OC/('card005-perq.npz' if CARD == 'card006' else 'card004-perq.npz'), allow_pickle=True)
    assert np.array_equal(prior['val_idx'], val_ids)
    assert np.array_equal(prior['val_source' if CARD == 'card006' else 'val_cohort'], grp)
    alias = {'frozen': 'frozen_t0' if CARD == 'card006' else 'frozen_s0', 'fresh':'fresh_final', 'anchored':'anchored','unanchored':'unanchored'}
    pq = {h: {B: prior[f'{a}_B{B}'] for B in EP.BUDGETS} for h,a in alias.items()}
    existing = OC/f'{CARD}-reception-perq.npz'
    if existing.exists():
        z = np.load(existing, allow_pickle=True)
        if not np.array_equal(z['val_group'], grp): raise ValueError('Existing query order changed')
        if 'val_idx' in z and not np.array_equal(z['val_idx'], val_ids): raise ValueError('Query ID mismatch')
        for h in ('in','out'):
            if all(f'{h}_B{B}' in z for B in EP.BUDGETS): pq[h] = {B:z[f'{h}_B{B}'] for B in EP.BUDGETS}
    if CARD == 'card006':
        from card005_fix import POOL
        weights = POOL
        pool_src = np.load('/data2/monet/pool-20m/source.npy', allow_pickle=True)
        ref_grp = pool_src[ref_ids].astype(str)
        del pool_src
    else:
        weights = EP.NAT
        ref_grp = np.load(seal/'ref_cohort.npy', allow_pickle=True).astype(str)
    aq = np.flatnonzero(np.isin(grp, arriving))
    cross = [truth[i][~np.isin(ref_grp[truth[i]], arriving)] for i in aq]
    eligible = np.array([len(t)>0 for t in cross]); aq = aq[eligible]; cross = [t for t in cross if len(t)]
    den = np.asarray(list(map(len,cross)),np.int32)
    cross_arrays = {'val_idx':val_ids[aq], 'truth_cross_count':den}
    cache = OC/'projection-cache';cache.mkdir(exist_ok=True)
    provenance, cross_report = {}, {}
    for h,p in heads.items():
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        instrument = [(str(f),f.stat().st_size,f.stat().st_mtime_ns) for f in (seal/'ref_hd.f16.npy',seal/'val_hd.f16.npy',seal/'ref_idx.npy',seal/'val_idx.npy')]
        key=hashlib.sha256(json.dumps([sha,instrument,'L2-f32-v1']).encode()).hexdigest()[:20]
        cp=cache/f'{CARD}-{h}-{key}.npz'
        if cp.exists():
            z=np.load(cp);rc,vc=z['ref'],z['val']
        else:
            mo=ParametricUMAP.load(str(p),device='cpu');mo.model.eval()
            def project(x):
                with torch.no_grad():return np.concatenate([mo.model(torch.from_numpy(EP._norm(np.asarray(x[i:i+25000],np.float32)))).numpy() for i in range(0,len(x),25000)])
            rc,vc=project(ref_hd),project(val_hd)
            np.savez(cp,ref=rc,val=vc)
            del mo
        if not np.isfinite(rc).all() or not np.isfinite(vc).all():raise ValueError('Nonfinite projection')
        ix=faiss.IndexFlatL2(rc.shape[1]);ix.add(np.ascontiguousarray(rc))
        if h not in pq:
            _,nn=ix.search(np.ascontiguousarray(vc),max(EP.BUDGETS))
            pq[h]={B:np.array([np.isin(truth[i],nn[i,:B]).sum()/truth.shape[1] for i in range(len(vc))]) for B in EP.BUDGETS}
            cn=nn[aq,:250]
        else:
            _,cn=ix.search(np.ascontiguousarray(vc[aq]),250)
        hits=np.array([np.isin(t,cn[j]).sum() for j,t in enumerate(cross)],np.int32)
        cross_arrays[f'{h}_hits']=hits
        cross_report[h]={'query_macro':float((hits/den).mean()),'edge_weighted':float(hits.sum()/den.sum())}
        provenance[h]={'model_path':str(p),'sha256':sha,'projection_cache':str(cp)}
        print(h,'scored/cached',flush=True)
    np.savez(OC/f'{CARD}-reception-perq.npz',val_idx=val_ids,val_group=grp,**{f'{h}_B{B}':pq[h][B] for h in pq for B in EP.BUDGETS})
    np.savez(OC/f'{CARD}-crossgroup-perq.npz',**cross_arrays)
    c2,c3=quality_gates(pq,grp,arriving,old,bool(CFG['chinese']))
    rep={h:{f'B{B}':{g:float(pq[h][B][grp==g].mean()) for g in groups} for B in EP.BUDGETS} for h in pq}
    selected=[g for g in groups if g in weights];total=sum(weights[g] for g in selected)
    natural={h:{f'B{B}':sum(weights[g]*rep[h][f'B{B}'][g] for g in selected)/total for B in EP.BUDGETS} for h in pq}
    paired={};cross_paired={}
    for other in ('in','anchored','frozen','unanchored','fresh'):
        paired[f'out_minus_{other}']={f'B{B}':{scope:paired_ci((pq['out'][B]-pq[other][B])[np.isin(grp,gs)]) for scope,gs in [('arriving',arriving),('old',old)]} for B in EP.BUDGETS}
        cross_paired[f'out_minus_{other}']=paired_ci((cross_arrays['out_hits']-cross_arrays[f'{other}_hits'])/den)
    movement=json.loads((OC/f'{CARD}-movement.json').read_text())
    dep=movement['deployment_criterion1_confirmation']
    result={'schema':f'{CARD}-reception-v2','card':CARD,'groups':{'arriving':arriving,'old':old,'excluded':CFG['exclude']},
            'reception_per_group':rep,'criterion2_old_retention':c2,'criterion3_arriving':c3,
            'deployment_pass':{t:bool(dep[t]['criterion1_pass'] and c2[t]['pass'] and c3[t]['pass']) for t in ('in','out')},
            'paired_query_differences':paired,'natural_query_weighted':natural,'natural_weights':{g:weights[g] for g in selected},
            'crossgroup':{'definition':'arriving query true neighbors outside all arriving cohorts; conditional on at least one such edge; DINO includes diagnostic diffusion among non-arriving refs',
                          'n_eligible_queries':len(aq),'n_truth_edges':int(den.sum()),'heads':cross_report,'paired':cross_paired},
            'provenance':provenance,'note':'Thresholds frozen in card before training; reporting code repaired after DINO results. All decisions use unrounded values. Equal-cohort gate means; query uncertainty is not seed uncertainty. Jina old seal has 22 cohorts (3 English registers +19 other languages), confirmation20 languages.'}
    (OC/f'{CARD}-reception.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'card':CARD,'deployment':result['deployment_pass'],'criterion2':c2,'criterion3':c3},indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
