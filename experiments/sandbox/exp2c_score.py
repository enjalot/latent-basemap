"""exp-2c trade scorer (owner-approved, overseer 2026-09-04). For ONE pair-edge-injected map, measures the
pair-adjacency vs per-modality-fidelity trade:
  pair adjacency: crossmodal frac@k15 (2D), matched_partner_in_knn@15, pair-dist ratio (2D)
  quality guard : per-modality within-modality FFR on the ORIGINAL (un-injected) centered knn truth — i.e. how
                  much intra-image / intra-text neighborhood structure the injection COST.
The FFR mirrors quick_ffr_v2 (recall of the exact-k15 high-D truth among the 2D disc of rows*0.1% neighbors,
self excluded) but restricts queries to one modality and uses the original centered knn_indices as truth.
Usage: exp2c_score.py <map_dir_with_coordinates.npy> <centered_dat_dir> <w_mult_label>  -> exp2c_score.json"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

K = 15


def _within_ffr(xy, knn, rows_idx, n_total, n_queries=10000, seed=0):
    """Recall of original-knn[:K] among the 2D disc of n_total*0.1% neighbors, queries restricted to rows_idx."""
    rng = np.random.default_rng(seed)
    q = rng.choice(rows_idx, size=min(n_queries, len(rows_idx)), replace=False)
    tree = cKDTree(xy)
    disc = max(K, int(round(n_total * 0.001)))
    _, nbr = tree.query(xy[q], k=disc + 1, workers=-1)     # +1 for self
    hit = 0; tot = 0
    for r, qi in enumerate(q):
        disc_set = set(int(x) for x in nbr[r] if x != qi)
        truth = [int(t) for t in knn[qi][:K] if t != qi]
        if truth:
            hit += sum(t in disc_set for t in truth); tot += len(truth)
    return round(hit / tot, 4) if tot else None


def main():
    mapdir = Path(sys.argv[1]); dat = Path(sys.argv[2]); label = sys.argv[3]
    xy = np.asarray(np.load(mapdir / "coordinates.npy"), dtype=np.float32)
    mod = np.load(dat / "modality.npy")
    # ORIGINAL un-injected centered knn = the truth; the pplan copies it into each 2c arm dir (map_dir.parent),
    # NOT the /data2 dat dir. Fall back to the sandbox centered dir if the arm-local copy is absent.
    knn_path = mapdir.parent / "knn_indices.npy"
    if not knn_path.is_file():
        knn_path = Path("/data/latent-basemap/sandbox/monet-neomme-pairs-500k-centered/knn_indices.npy")
    knn = np.load(knn_path, mmap_mode="r")
    n_rows = xy.shape[0]; N = n_rows // 2
    span = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    # pair adjacency
    tree = cKDTree(xy)
    _, nbr = tree.query(xy, k=K + 1, workers=-1); nb = nbr[:, 1:K + 1]
    cm = (mod[nb] != mod[:, None]).mean()
    partner = np.concatenate([np.arange(N) + N, np.arange(N)])
    matched_row = (nb == partner[:, None]).any(1)          # per-row: is the matched partner in the 2D k15?
    matched_in_knn = float(matched_row.mean())
    # held-out split (owner exp-2c): pairs whose edge was NOT injected — matched rate on them = GENERALIZATION
    holdout_split = None
    if len(sys.argv) > 4 and Path(sys.argv[4]).is_file():
        ho = np.load(sys.argv[4])                          # held-out pair ids (image row i, text row N+i)
        ho_rows = np.concatenate([ho, ho + N]); tr_mask = np.ones(2 * N, bool); tr_mask[ho_rows] = False
        holdout_split = {"matched_partner_holdout": round(float(matched_row[ho_rows].mean()), 4),
                         "matched_partner_train": round(float(matched_row[tr_mask].mean()), 4),
                         "n_holdout_pairs": int(ho.size),
                         "note": "held-out pairs got NO injected edge; their matched rate tests generalization vs memorization"}
    d_match = np.linalg.norm(xy[:N] - xy[N:2 * N], axis=1)
    perm = np.random.default_rng(42).permutation(N)
    d_rand = np.linalg.norm(xy[:N] - xy[N:2 * N][perm], axis=1)
    # quality guard: per-modality within-modality FFR vs ORIGINAL centered knn
    img_ffr = _within_ffr(xy, knn, np.arange(N), n_rows)
    txt_ffr = _within_ffr(xy, knn, np.arange(N, 2 * N), n_rows)
    out = {"arm": label, "n_pairs": N, "map_span": round(span, 3),
           "pair_adjacency": {
               "crossmodal_frac_at_k15": round(float(cm), 4),
               "matched_partner_in_knn_at_k15": round(matched_in_knn, 4),
               "pair_dist_ratio_2d": round(float(np.median(d_match) / (np.median(d_rand) + 1e-9)), 4),
               "holdout_split": holdout_split},
           "quality_guard_within_modality_ffr": {
               "image_ffr": img_ffr, "text_ffr": txt_ffr,
               "note": "FFR of the 2c map vs the ORIGINAL un-injected centered k15 truth, per modality — the "
                       "intra-modality structure cost of pair injection"}}
    (mapdir / "exp2c_score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
