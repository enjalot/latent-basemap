"""Card011 quality scoring on the ORIGINAL common evaluation instrument (reuses the canonical
score_card010_codex / score_card009 machinery so numbers are comparable). CPU only. Development
evidence (promotion needs fresh excluded confirmation). Four heads: the unmodified fixed15 START
(card010) + the three card011 continuations (ordinary / collision / verified_random).

Gate (frozen card011-prereg.md): verified_collision must beat ALL THREE references (ordinary,
verified_random, unmodified start) by B250 >= +.015 with positive paired CI; B2000 loss <= .005; no
cohort loses > .01 at either budget; >= 20% fewer SEVERE geometric false joins on a held-out panel; no
> .005 continuity decline.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '4'
import json, time
from pathlib import Path
import numpy as np
from score_card009 import project, recall, sha, write_json, BUDGETS, ParametricUMAP, faiss
from score_card010_codex import ranks, reliability, contrast
from scipy.spatial.distance import cdist

OC = Path('/data/latent-basemap/sandbox/overseer-codex'); OUT = OC / 'card011-scoring'
C10 = Path('/data/latent-basemap/sandbox/card010-train'); C11 = Path('/data/latent-basemap/sandbox/card011-train')
SEAL = Path('/data2/monet/eval-common-v2'); SEED = 10010
REAL5 = ['laion', 'coyo', 'commoncatalog-cc-by', 'megalith10m', 'cc12m']
HEADS = {'start_fixed15': C10 / 'model-fixed15.pt', 'ordinary': C11 / 'model-ordinary.pt',
         'collision': C11 / 'model-collision.pt', 'verified_random': C11 / 'model-verified_random.pt'}
REFERENCES = ['ordinary', 'verified_random', 'start_fixed15']       # collision must beat ALL THREE


def severe_false_joins(H, xy_panel, k=15, k60=60, rho_min=1.25):
    """Per-head SEVERE geometric false-join count on the panel, matching the FROZEN mining rule (panel-
    internal): for each i, its 15 nearest MAP points (EUCLIDEAN 2D); a map neighbor j is a severe false
    join iff BIDIRECTIONAL non-neighbor (j not in enc-k60(i) AND i not in enc-k60(j)) AND EUCLIDEAN
    encoder rho = ||x_i-x_j||_2 / max(d60_i,d60_j) > 1.25. All distances EUCLIDEAN on unit vectors (L2,
    not cosine-distance ratio). Unordered dedupe. Returns count + fraction."""
    n = len(H); rows = np.arange(n)
    DE = cdist(H, H, 'euclidean'); np.fill_diagonal(DE, np.inf)     # encoder L2 on unit vectors
    enc_order = np.argsort(DE, axis=1, kind='stable')              # nearest-first, self excluded (inf)
    enc_k60 = enc_order[:, :k60]                                    # (n,60) encoder k60 sets
    in_k60 = np.zeros((n, n), bool)
    in_k60[rows[:, None], enc_k60] = True                          # in_k60[i,j] = j in enc-k60(i)
    d60 = DE[rows[:, None], enc_k60[:, k60 - 1:k60]].ravel()       # i's 60th-nearest L2 distance
    D2 = cdist(xy_panel, xy_panel, 'euclidean'); np.fill_diagonal(D2, np.inf)
    map15 = np.argsort(D2, axis=1, kind='stable')[:, :k]           # (n,15) 2D nearest, self excluded
    severe_pairs = set()
    for i in range(n):
        for j in map15[i]:
            if in_k60[i, j] or in_k60[j, i]:                       # bidirectional non-neighbor required
                continue
            if DE[i, j] > rho_min * max(d60[i], d60[j]):           # Euclidean rho > 1.25
                severe_pairs.add((min(i, int(j)), max(i, int(j))))  # unordered dedupe
    total_map_pairs = n * k
    return {'severe_false_joins': int(len(severe_pairs)),
            'severe_frac_of_map15': float(len(severe_pairs) / total_map_pairs),
            'metric': 'bidirectional non-neighbor + Euclidean rho>1.25 on map-15 (panel-internal, unordered)'}


def main():
    start = time.time(); OUT.mkdir(exist_ok=True)
    names = ['ref_hd.f16.npy', 'val_hd.f16.npy', 'truth_val.npy', 'ref_idx.npy', 'val_idx.npy', 'val_source.npy']
    instrument = {n: sha(SEAL / n) for n in names}
    ref = np.load(SEAL / names[0], mmap_mode='r'); val = np.load(SEAL / names[1], mmap_mode='r')
    truth = np.load(SEAL / names[2]); rid = np.load(SEAL / names[3]); qid = np.load(SEAL / names[4])
    groups = np.load(SEAL / names[5], allow_pickle=True).astype(str); cohorts = np.unique(groups)
    assert truth.shape == (len(val), 15)
    draw = np.load('/data/latent-basemap/substrates/card010-adaptive/draw_ids.npy')
    assert not np.isin(draw, np.r_[rid, qid]).any(), "seal leaked into training draw"
    rng = np.random.default_rng(SEED)
    panel = np.sort(np.concatenate([rng.choice(np.flatnonzero(groups == g), 200, replace=False) for g in cohorts]))
    H = np.array(val[panel], dtype='f4'); H /= np.linalg.norm(H, axis=1, keepdims=True).clip(1e-12)
    hi, hr = ranks(cdist(H, H, 'cosine'))
    km = faiss.Kmeans(H.shape[1], 16, niter=20, nredo=1, seed=SEED, verbose=False)
    km.train(H); _, li = km.index.search(H, 1); labels = li[:, 0]

    instrument = {n: sha(SEAL / n) for n in names}
    provenance = {'scorer_sha256': sha(__file__), 'instrument': instrument, 'heads': {}}
    report = {}; perq = {}; pending = []; xy = {}
    for h, mp in HEADS.items():
        if not mp.exists():
            pending.append(h); continue
        provenance['heads'][h] = {'model_path': str(mp), 'model_sha256': sha(mp)}
        model = ParametricUMAP.load(str(mp), device='cpu').model.eval()
        rc = project(model, ref, True); vc = project(model, val, True); pq = recall(rc, vc, truth)
        del model
        for b, v in pq.items():
            assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), f"{h} B{b} not finite/in[0,1]"
        perq[h] = pq; xy[h] = vc
        by = {str(b): {g: float(pq[b][groups == g].mean()) for g in cohorts} for b in BUDGETS}
        rel = reliability(vc[panel].astype('f8'), hi, hr, labels, km.centroids)
        sfj = severe_false_joins(H, vc[panel].astype('f8'))
        report[h] = {'by_source': by,
                     'equal_cohort': {str(b): float(np.mean(list(by[str(b)].values()))) for b in BUDGETS},
                     'real5': {str(b): float(pq[b][np.isin(groups, REAL5)].mean()) for b in BUDGETS},
                     'reliability': rel, 'severe_false_joins': sfj}
        print(json.dumps({'head': h, 'equal_cohort': report[h]['equal_cohort'],
                          'severe_false_joins': sfj['severe_false_joins'],
                          'continuity': rel['continuity']}), flush=True)
    if pending:
        write_json(OUT / 'status.json', {'status': 'WAITING_HEADS', 'pending': pending}); print('pending', pending); return

    # ---- gates: collision vs ALL THREE references ----
    comparisons = {}; gate = {}
    for ref_h in REFERENCES:
        con = {str(b): contrast(perq['collision'][b], perq[ref_h][b], groups) for b in [250, 2000]}
        per = {str(b): {g: report['collision']['by_source'][str(b)][g] - report[ref_h]['by_source'][str(b)][g]
                        for g in cohorts} for b in [250, 2000]}
        sfj_c = report['collision']['severe_false_joins']['severe_false_joins']
        sfj_r = report[ref_h]['severe_false_joins']['severe_false_joins']
        # 0/0 is UNINFORMATIVE: a zero reference cannot demonstrate a 20% reduction. Ratio null, check
        # NOT satisfied (not a false pass). Only a positive reference count can evidence the reduction.
        sfj_ratio = round(sfj_c / sfj_r, 4) if sfj_r > 0 else None
        sfj_check = (sfj_c <= 0.80 * sfj_r) if sfj_r > 0 else None
        checks = {'B250_gain_ge_0015': con['250']['delta'] >= .015, 'B250_ci_positive': con['250']['ci95'][0] > 0,
                  'B2000_delta_ge_minus0005': con['2000']['delta'] >= -.005,
                  'every_cohort_both_budgets_loss_le_001': all(v >= -.01 for d in per.values() for v in d.values()),
                  'severe_false_joins_20pct_fewer': sfj_check,   # None = not demonstrated (zero reference)
                  'continuity_decline_le_0005': bool(report['collision']['reliability']['continuity']
                                                     >= report[ref_h]['reliability']['continuity'] - .005)}
        comparisons[ref_h] = {'aggregate': con, 'by_source_delta': per,
                              'severe_false_joins': {'collision': sfj_c, ref_h: sfj_r, 'ratio': sfj_ratio,
                                                     'note': 'counts are unordered severe pairs; frac denom = directed n*k slots (labeled per-directed-slot); zero reference => reduction NOT demonstrated'},
                              'checks': checks}
        # a None (undemonstrated) severe-join check does NOT count as satisfied
        gate[ref_h] = all(v is True for v in checks.values())

    # natural pool-weighted aggregate (same weights card010 used) + persistence
    try:
        weights = json.loads((OC / 'card009-result.json').read_text())['natural_pool_weights']
        natural = {h: {str(b): sum(weights[g] * report[h]['by_source'][str(b)][g] for g in weights) / sum(weights.values())
                       for b in BUDGETS} for h in report}
    except Exception:
        weights, natural = None, None
    arrays = {'query_sources': groups, 'query_ids': qid, 'panel_local': panel, 'panel_query_ids': qid[panel]}
    for h in report:
        for b in BUDGETS:
            arrays[f'{h}_B{b}'] = perq[h][b]
    np.savez(OUT / 'per-query.npz', **arrays)
    write_json(OUT / 'provenance.json', provenance)
    result = {'status': 'SCORED', 'card': 'card011', 'exploratory_development': True,
              'natural_weights': weights, 'natural_query_weighted': natural,
              'provenance_path': str(OUT / 'provenance.json'), 'per_query_path': str(OUT / 'per-query.npz'),
              'metric': 'B-recall of encoder-k15 on original eval-common-v2 (all 9 cohorts, out-of-sample seal queries)',
              'development_disclaimer': 'Development evidence; promotion needs fresh excluded confirmation.',
              'heads': report, 'comparisons': comparisons,
              'NUMERICAL_QUALITY_SCREEN_PASS': all(gate.values()),
              'gate_requires': 'collision beats ordinary, verified_random AND unmodified start on every check',
              'uncertainty': '2000 paired source-stratified equal-cohort bootstrap draws, seed10010; one trajectory; development data',
              'cpu_wall_s': time.time() - start}
    write_json(OUT / 'result.json', result); write_json(OC / 'card011-score.json', result)
    print(json.dumps({'comparisons': {r: comparisons[r]['checks'] for r in REFERENCES},
                      'NUMERICAL_QUALITY_SCREEN_PASS': all(gate.values())}, indent=2), flush=True)


if __name__ == '__main__':
    main()
