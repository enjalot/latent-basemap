"""Panel v2 — the versioned, in-repo decision evaluator (P0.5, P0.6).

Replaces the machine-local scoring scripts (`/data/latent-basemap/track1/
score_a3_gate.py`, `score_minilm.py`, `e3_coarse.py`, …) whose outputs could not
be regenerated from a checked-in evaluator + config. Everything here is
formula-versioned and reproducible; large arrays stay at `gsv:` paths and are
referenced by manifests.

Canonical metrics (all against the FULL corpus via exact GPU kNN, streamed in
corpus chunks so they are bounded-memory at 150M — P0.6):
  - ffr    : fixed-fraction recall = |true-hiD top-10 ∩ 2D top-⌈frac·N⌉| / 10
  - purity : region-purity ratio (2D vs hiD label agreement) at k_clust
  - density: exact log-radius (k=15) correlation, hiD vs 2D
  - proj   : ffr for an out-of-sample query set projected into the map

Guards: recall@10 (top-10 vs top-10) is reported separately and MUST NOT be
written into `ffr` (P0.4). ID alignment (ls_index/row_id) is enforced when
present (P0.5).
"""
from __future__ import annotations
import os, json, glob, hashlib, dataclasses
import numpy as np

FORMULA_VERSION = "panel_v2.0-2026-07-14"
D_DEFAULT_FRAC = 0.001
D_K_DENSITY = 15
D_K_HIT = 10


@dataclasses.dataclass
class PanelV2Config:
    frac: float = D_DEFAULT_FRAC          # fixed-fraction (0.1%) for ffr/purity
    k_clust: tuple = (256, 1024)          # purity granularities
    k_density: int = D_K_DENSITY
    k_hit: int = D_K_HIT
    n_anchors: int = 10000
    anchor_seed: int = 42
    corpus_chunk: int = 1_000_000         # bounded-memory streaming chunk
    formula_version: str = FORMULA_VERSION


# ── loaders ────────────────────────────────────────────────────────────────────

def load_coords(path: str):
    """Load a coords parquet. Returns (Z_all_dims float32 [N,d], ids or None).

    Preserves EVERY coordinate dimension (x,y[,z,c3,…]) — the old loader dropped
    to x,y so G7 could not be enforced on 3D maps (P0.5). Recognizes `ls_index`
    or `row_id` as the alignment key; validates uniqueness.
    """
    import pandas as pd
    df = pd.read_parquet(path)
    id_col = "ls_index" if "ls_index" in df.columns else ("row_id" if "row_id" in df.columns else None)
    coord_cols = [c for c in df.columns if c not in ("ls_index", "row_id")]
    # canonical order x,y,z,c3,c4,…
    order = [c for c in ["x", "y", "z"] if c in coord_cols] + \
            sorted([c for c in coord_cols if c not in ("x", "y", "z")])
    Z = df[order].values.astype("float32")
    ids = None
    if id_col is not None:
        ids = df[id_col].values.astype("int64")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{path}: {id_col} has duplicate ids (P0.5).")
    return Z, ids


def load_embeddings(path: str, dim: int | None = None):
    """Memmap an embedding shard. Supports standard .npy AND raw-headerless
    float32 (MiniLM shards: no NUMPY magic). For raw, `dim` is required and the
    row count is inferred from file size (P0.5 — no separate machine-local loader).
    """
    with open(path, "rb") as fh:
        magic = fh.read(6)
    if magic == b"\x93NUMPY":
        return np.load(path, mmap_mode="r")
    if dim is None:
        raise ValueError(f"{path} is raw-headerless; `dim` is required to infer shape.")
    n = os.path.getsize(path) // (dim * 4)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, dim))


def _ids_hash(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


# ── bounded-memory kNN (P0.6) ───────────────────────────────────────────────────

def _bounded_self_knn(F, anchor_idx, k, cfg: PanelV2Config, hi_dim=True, want_dist=False,
                      block=500_000_000):
    """Top-k neighbours (self-excluded) of ``anchor_idx`` rows within corpus F,
    tiling BOTH anchors and corpus so peak VRAM per block ≤ ``block`` elements
    (~2 GB fp32). Within each (anchor-tile × corpus-chunk) block we topk locally,
    then merge into the running best — never materialising an (anchors × N)
    matrix. hi_dim uses the unit-norm matmul expansion; low_dim uses exact
    per-dim diffs (G1: no fp cancellation)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N = len(F); m = len(anchor_idx); kk = k + 1
    # Low-dim coords are tiny → keep the whole corpus in ONE chunk and tile only
    # anchors, so the top-k is a deterministic global pass (matches the exact
    # path). Corpus-chunking a low-D map with many duplicate coords breaks ties
    # by chunk boundary and perturbs density. High-D X can't fit → chunk it
    # (continuous 768-D distances have no ties, so chunked == exact there).
    if not hi_dim:
        cchunk = N
    else:
        cchunk = max(1, min(N, int(block // max(1, min(m, 4096)))))
    achunk = max(1, min(m, int(block // cchunk)))
    out_i = np.empty((m, k), dtype=np.int64)
    out_d = np.empty((m, k), dtype=np.float64) if want_dist else None
    for a0 in range(0, m, achunk):
        aids = anchor_idx[a0:a0 + achunk]
        Q = torch.from_numpy(np.asarray(F[aids], dtype=np.float32)).to(dev)
        ma, D = Q.shape
        qn = (Q * Q).sum(1, keepdim=True) if hi_dim else None
        best_d = torch.full((ma, kk), float("inf"), device=dev)
        best_i = torch.full((ma, kk), -1, dtype=torch.long, device=dev)
        for j in range(0, N, cchunk):
            Xc = torch.from_numpy(np.asarray(F[j:j + cchunk], dtype=np.float32)).to(dev)
            if hi_dim:
                d2 = qn - 2.0 * (Q @ Xc.T) + (Xc * Xc).sum(1)
            else:
                d2 = torch.zeros((ma, len(Xc)), device=dev)
                for c in range(D):
                    diff = Q[:, c:c + 1] - Xc[:, c]
                    d2.addcmul_(diff, diff)
            kloc = min(kk, len(Xc))
            ld, li = torch.topk(d2, kloc, dim=1, largest=False)   # within-chunk topk
            li = li + j
            best_d = torch.cat([best_d, ld], 1)
            best_i = torch.cat([best_i, li], 1)
            best_d, sel = torch.topk(best_d, kk, dim=1, largest=False)
            best_i = torch.gather(best_i, 1, sel)
            del Xc, d2
        if want_dist and hi_dim:
            # Recompute distances EXACTLY by gathering neighbour vectors and
            # subtracting — the matmul expansion qn−2qx+xn catastrophically
            # cancels for near-duplicate high-D rows (true dist ~1e-4 → 0),
            # which log-density turns into a huge spurious outlier (G1). Ranking
            # via the expansion is fine; the radius must be exact.
            nb = torch.from_numpy(np.asarray(F[best_i.reshape(-1).cpu().numpy()],
                                             dtype=np.float32)).to(dev).reshape(ma, kk, -1)
            best_d = (nb - Q[:, None, :]).norm(dim=2) ** 2   # squared, for consistency below
            del nb
        ids = best_i.cpu().numpy()
        dist = best_d.clamp_min(0).sqrt().cpu().numpy() if want_dist else None
        for r in range(ma):
            keep = ids[r] != aids[r]
            row = ids[r][keep][:k]
            out_i[a0 + r] = row if len(row) == k else ids[r][:k]
            if want_dist:
                dr = dist[r][keep][:k]
                out_d[a0 + r] = dr if len(dr) == k else dist[r][:k]
        del Q, best_d, best_i
    return (out_i, out_d) if want_dist else out_i


# ── metrics ─────────────────────────────────────────────────────────────────────

def sample_anchors(n, cfg: PanelV2Config):
    rng = np.random.RandomState(cfg.anchor_seed)
    return np.sort(rng.choice(n, min(cfg.n_anchors, n), replace=False))


def score_ffr(X, Z, cfg: PanelV2Config, anchor_idx=None, restrict=None):
    """ffr = |hiD top-10 ∩ 2D top-k_frac|/10. Also returns recall@10 SEPARATELY
    (never conflated — P0.4)."""
    n = len(Z)
    aidx = anchor_idx if anchor_idx is not None else sample_anchors(n, cfg)
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * n)))
    hi = _bounded_self_knn(X, aidx, cfg.k_hit, cfg, hi_dim=True)          # true hiD top-10
    lo = _bounded_self_knn(Z, aidx, kf, cfg, hi_dim=False)               # 2D top-k_frac
    sel = np.arange(len(aidx)) if restrict is None else restrict
    ffr = float(np.mean([len(np.intersect1d(hi[i], lo[i])) / cfg.k_hit for i in sel]))
    r10 = float(np.mean([len(np.intersect1d(hi[i], lo[i, :cfg.k_hit])) / cfg.k_hit for i in sel]))
    return {"ffr": round(ffr, 4), "recall@10": round(r10, 5), "k_frac": kf,
            "n_anchors_scored": int(len(sel))}


def score_density(X, Z, cfg: PanelV2Config, anchor_idx=None):
    """Exact log-radius (k=15) correlation, hiD vs 2D. Bounded-memory (P0.6)."""
    n = len(Z)
    aidx = anchor_idx if anchor_idx is not None else sample_anchors(n, cfg)
    _, hd = _bounded_self_knn(X, aidx, cfg.k_density, cfg, hi_dim=True, want_dist=True)
    _, ld = _bounded_self_knn(Z, aidx, cfg.k_density, cfg, hi_dim=False, want_dist=True)
    r_hd = hd.mean(1); r_ld = ld.mean(1); eps = 1e-12
    dens = float(np.corrcoef(np.log(r_hd + eps), np.log(r_ld + eps))[0, 1])
    return {"density": round(dens, 4)}


def score_purity(X, Z, centroids, cfg: PanelV2Config, anchor_idx=None, restrict=None):
    """Region-purity ratio (2D vs hiD label agreement at k_frac), per k_clust."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(Z)
    aidx = anchor_idx if anchor_idx is not None else sample_anchors(n, cfg)
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * n)))
    hi = _bounded_self_knn(X, aidx, kf, cfg, hi_dim=True)
    lo = _bounded_self_knn(Z, aidx, kf, cfg, hi_dim=False)
    sel = np.arange(len(aidx)) if restrict is None else restrict
    out = {}
    for C in centroids:
        Ct = torch.from_numpy(np.asarray(C, dtype=np.float32)).to(dev)
        lab = np.empty(n, dtype=np.int32)
        for i in range(0, n, 65536):
            xb = torch.from_numpy(np.asarray(X[i:i + 65536], dtype=np.float32)).to(dev)
            lab[i:i + 65536] = torch.cdist(xb, Ct).argmin(1).cpu().numpy()
        del Ct
        alab = lab[aidx][sel]
        hd = float((lab[hi[sel]] == alab[:, None]).mean())
        mp = float((lab[lo[sel]] == alab[:, None]).mean())
        out[f"purity_k{len(C)}"] = round(mp / hd, 4) if hd else None
    return out


def run_panel(X, Z, cfg: PanelV2Config, centroids=None, ids=None, restrict=None,
              manifest_extra=None):
    """Full panel v2 for a corpus X and its map Z. Returns a formula-versioned
    dict with metric values + provenance (anchor hash, N, k_frac, formula). If
    coords carry ids, they must align 1:1 with X row order (validated)."""
    n = len(Z)
    if len(X) != n:
        raise ValueError(f"panel_v2: len(X)={len(X)} != len(Z)={n}")
    if ids is not None and not (np.diff(ids) >= 0).all():
        # ids present but not row-aligned ascending → caller must reorder X/Z by id
        raise ValueError("panel_v2: coord ids are not in ascending row order; align X to ids first (P0.5).")
    aidx = sample_anchors(n, cfg)
    res = {"n": int(n), "formula_version": cfg.formula_version, "frac": cfg.frac,
           "anchor_hash": _ids_hash(aidx), "n_dims": int(Z.shape[1])}
    res.update(score_ffr(X, Z, cfg, aidx, restrict))
    if centroids is not None:
        res.update(score_purity(X, Z, centroids, cfg, aidx, restrict))
    res.update(score_density(X, Z, cfg, aidx))
    if manifest_extra:
        res.update(manifest_extra)
    return res
