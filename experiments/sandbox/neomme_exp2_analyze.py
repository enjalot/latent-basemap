"""NeoMME exp-2 interleave-vs-islands analysis (owner NeoMME probe, overseer 2026-09-04). Given the 500K joint
champion map (250K image rows [0,N) + 250K text rows [N,2N), row i <-> N+i matched), answer the whole probe:
do matched image-caption pairs land together, and are the two modalities INTERLEAVED or in separate ISLANDS?

Deliverables (overseer):
  1. pair-distance vs random-pair baseline  (2D map + source 1024-d)
  2. cross-modal fraction @k=15              (2D map full; source on a sample)  -> islands vs interleave
  3. modality + pair_id provenance           (already on disk from the embed; summarized here for the viewer)

Reads: coords from the champion map dir; modality.npy / pair_id.npy / substrate from the pairs-500k dir.
CPU-only (.venv: scipy cKDTree for 2D kNN, chunked cosine for the source sample). Writes exp2_summary.json.
Usage: neomme_exp2_analyze.py"""
import json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

SB = Path("/data/latent-basemap/sandbox")
MAP = SB / "monet-neomme-pairs-500k" / "champion-bs16k"
DAT = Path("/data2/monet/neomme-pairs-500k")
K = 15; SEED = 42; SAMPLE = 50_000


def _crossmodal_frac_2d(xy, mod):
    """For each point, fraction of its K nearest 2D neighbors of the OTHER modality; + matched-partner-in-kNN."""
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=K + 1, workers=-1)      # col 0 = self
    nb = idx[:, 1:K + 1]
    other = (mod[nb] != mod[:, None]).mean(1)          # frac of neighbors that are the other modality
    return other


def _matched_in_knn_2d(xy, N):
    """Fraction of image rows whose matched caption (row i <-> N+i) is within its 2D kNN (and vice-versa)."""
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=K + 1, workers=-1)
    nb = idx[:, 1:K + 1]
    part = np.concatenate([np.arange(N) + N, np.arange(N)])   # partner row of each row
    return float((nb == part[:, None]).any(1).mean())


def _source_sample(sub, mod, N, rng):
    """Cross-modal frac@K and matched/random cosine in the 1024-d source space, on a sample of query rows."""
    q = np.sort(rng.choice(2 * N, min(SAMPLE, 2 * N), replace=False))
    Q = np.array(sub[q], dtype=np.float32)                    # copy only the sample (memory-safe)
    other = np.empty(len(q), np.float32)
    B = 2000
    for s in range(0, len(q), B):
        qb = Q[s:s + B]
        sims = np.empty((qb.shape[0], 2 * N), np.float32)
        C = 100_000
        for c in range(0, 2 * N, C):
            sims[:, c:c + C] = qb @ np.array(sub[c:c + C], dtype=np.float32).T
        # drop self (exact 1.0 on the diagonal position q[s+..]); take top K+1 then remove self
        for r in range(qb.shape[0]):
            sims[r, q[s + r]] = -2.0
        topk = np.argpartition(sims, -K, axis=1)[:, -K:]
        other[s:s + B] = (mod[topk] != mod[q[s:s + B]][:, None]).mean(1)
    # matched vs random pair cosine (image_i . text_i) vs (image_i . text_perm)
    ii = rng.choice(N, min(20000, N), replace=False)
    im = np.array(sub[ii], dtype=np.float32); tx = np.array(sub[N + ii], dtype=np.float32)
    matched = float((im * tx).sum(1).mean())
    perm = rng.permutation(len(ii)); rnd = float((im * tx[perm]).sum(1).mean())
    return float(other.mean()), matched, rnd


def _chunk_mean(sub, lo, hi):
    """Mean over rows [lo,hi) of the substrate memmap, memory-safe."""
    acc = np.zeros(sub.shape[1], np.float64); C = 100_000
    for c in range(lo, hi, C):
        acc += np.array(sub[c:min(c + C, hi)], dtype=np.float64).sum(0)
    return (acc / (hi - lo)).astype(np.float32)


def _centered_margins(sub, N, rng):
    """Matched vs random pair margin under 3 centerings — the ANISOTROPY diagnostic (overseer 2026-09-04):
    both-high raw cosines = tight cone; removing the shared mean should expose pair structure if it exists.
      raw            : no centering (the cone is present)
      joint_mean     : subtract the global mean over all 2N rows (removes the global cone)
      per_modality   : subtract the image-mean from images + text-mean from texts (also removes modality offset)
    Each vector re-normalized after centering, then cosine. A LARGE centered margin => pairs ARE resolvable and
    the raw-cosine kNN graph (champion's) is seeing the cone, not the pairs."""
    joint_m = _chunk_mean(sub, 0, 2 * N)
    img_m = _chunk_mean(sub, 0, N); txt_m = _chunk_mean(sub, N, 2 * N)
    ii = rng.choice(N, min(20000, N), replace=False)
    im = np.array(sub[ii], dtype=np.float32); tx = np.array(sub[N + ii], dtype=np.float32)
    perm = rng.permutation(len(ii))

    def _mr(a, b):
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        matched = float((an * bn).sum(1).mean()); rnd = float((an * bn[perm]).sum(1).mean())
        return {"matched": round(matched, 4), "random": round(rnd, 4), "margin": round(matched - rnd, 4)}

    return {"raw": _mr(im, tx),
            "joint_mean_centered": _mr(im - joint_m, tx - joint_m),
            "per_modality_centered": _mr(im - img_m, tx - txt_m),
            "cone_stats": {"mean_norm_joint": round(float(np.linalg.norm(joint_m)), 4),
                           "img_mean_norm": round(float(np.linalg.norm(img_m)), 4),
                           "txt_mean_norm": round(float(np.linalg.norm(txt_m)), 4),
                           "img_txt_mean_cos": round(float(img_m @ txt_m / (np.linalg.norm(img_m) * np.linalg.norm(txt_m) + 1e-9)), 4)},
            "note": "large margin under centering but small raw => anisotropic cone hides pair structure; "
                    "big img/txt mean_norm relative to unit vectors = strong shared component (Marlin-style)"}


def main():
    t0 = time.time()
    xy = np.asarray(np.load(MAP / "coordinates.npy"), dtype=np.float32)
    mod = np.load(DAT / "modality.npy")
    n_rows = xy.shape[0]; N = n_rows // 2
    assert mod.shape[0] == n_rows, f"provenance/coords mismatch {mod.shape} vs {xy.shape}"
    # map scale for normalization
    span = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    # 1. pair-distance vs random baseline (2D)
    d_match = np.linalg.norm(xy[:N] - xy[N:2 * N], axis=1)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    d_rand = np.linalg.norm(xy[:N] - xy[N:2 * N][perm], axis=1)
    # 2. cross-modal fraction @k=15 (2D, full)
    cm = _crossmodal_frac_2d(xy, mod)
    matched_in_knn = _matched_in_knn_2d(xy, N)
    # source-space companions (sample)
    sub = np.load(DAT / "substrate.f32.npy", mmap_mode="r")
    src_cm, src_match_cos, src_rand_cos = _source_sample(sub, mod, N, np.random.default_rng(SEED))
    anis = _centered_margins(sub, N, np.random.default_rng(SEED))
    out = {
        "n_pairs": N, "n_rows": n_rows, "k": K, "map_span": round(span, 4),
        "pair_distance_2d": {
            "median_matched": round(float(np.median(d_match)), 4),
            "median_random": round(float(np.median(d_rand)), 4),
            "matched_over_random_ratio": round(float(np.median(d_match) / (np.median(d_rand) + 1e-9)), 4),
            "median_matched_frac_of_span": round(float(np.median(d_match) / (span + 1e-9)), 5),
            "note": "ratio << 1 => matched image-caption pairs are pulled together in the map"},
        "crossmodal_frac_at_k15_2d": {
            "overall": round(float(cm.mean()), 4),
            "image_rows": round(float(cm[:N].mean()), 4),
            "text_rows": round(float(cm[N:].mean()), 4),
            "matched_partner_in_knn_frac": round(matched_in_knn, 4),
            "interpretation": "0.5 = fully interleaved (50/50 mix), ~0 = separate modality islands"},
        "source_space_1024d": {
            "crossmodal_frac_at_k15_sample": round(src_cm, 4),
            "matched_pair_cosine": round(src_match_cos, 4),
            "random_pair_cosine": round(src_rand_cos, 4),
            "note": "compare 2D crossmodal frac vs this to see if islands are already in NeoMME's space "
                    "or introduced by the flattening"},
        "anisotropy_diagnostic": anis,
        "wall_s": round(time.time() - t0, 1),
    }
    (MAP / "exp2_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
