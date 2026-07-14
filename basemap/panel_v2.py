"""Panel v2 — the versioned, in-repo decision evaluator (P0.5, P0.6, P0-C).

Replaces the machine-local scoring scripts (`/data/latent-basemap/track1/
score_a3_gate.py`, `score_minilm.py`, `e3_coarse.py`, …) whose outputs could not
be regenerated from a checked-in evaluator + config. There is now ONE canonical
entry point, `score_panel`, called by both the runner and the CLI, so every R1
decision comes from identical code.

Canonical metrics (all against the aligned corpus via bounded-memory GPU kNN):
  - ffr    : fixed-fraction recall = |true-hiD top-k_hit ∩ 2D top-k_frac| / k_hit
  - recall@k_hit : top-k_hit vs top-k_hit — reported SEPARATELY, never conflated
  - purity : region-purity ratio (2D vs hiD label agreement) at each k_clust
  - density: exact log-radius (k) correlation, hiD vs 2D
  - proj   : ffr for an out-of-sample query set projected into the map

High-D exactness contract (P0-C): neighbours are found by an *overselect +
exact rerank* scheme. Within each corpus chunk we take the top ``k+margin`` by
the fast normalised-matmul expansion (which only ever RANKS near-duplicates too
high, never drops them), merge a running candidate pool, then gather the exact
vectors and recompute fp32 distances to rerank and to report exact radii. This
is exact modulo the overselect margin; a boundary guard reports the gap between
the kept k-th candidate and the first dropped one so a too-small margin is
visible, not silent. Low-D coords are scored in a single deterministic pass
(no chunk-boundary tie perturbation).

ID alignment (P0-C) is EXACT: when both ``x_ids`` and ``z_ids`` are given they
must describe the same row universe and X is reordered to Z; alignment is never
inferred from sorting. recall@k is kept out of ``ffr`` (P0.4).
"""
from __future__ import annotations
import os, json, glob, time, hashlib, resource, subprocess, dataclasses
import numpy as np

FORMULA_VERSION = "panel_v2.1-2026-07-14"
PANEL_SCHEMA = "panel_v2"
D_DEFAULT_FRAC = 0.001
D_K_DENSITY = 15
D_K_HIT = 10


@dataclasses.dataclass
class PanelV2Config:
    frac: float = D_DEFAULT_FRAC           # fixed-fraction (0.1%) for ffr/purity
    k_clust: tuple = (256, 1024)           # purity granularities (kept for provenance)
    k_density: int = D_K_DENSITY
    k_hit: int = D_K_HIT
    n_anchors: int = 10000
    anchor_seed: int = 42
    corpus_chunk: int = 2_000_000          # rows streamed per corpus chunk (hiD)
    overselect: int = 8                    # extra candidates kept before exact rerank
    block_elems: int = 500_000_000         # per-block element cap (~2 GB fp32)
    formula_version: str = FORMULA_VERSION


# ── pure metric formulas (P0-C: score explicit neighbour-ID matrices) ────────────

def ffr_from_neighbors(hi_top: np.ndarray, lo_frac: np.ndarray, k_hit: int) -> float:
    """FFR = mean_i |hi_top[i] ∩ lo_frac[i]| / k_hit. ``hi_top`` is (m, k_hit) true
    high-D neighbours; ``lo_frac`` is (m, k_frac) low-D neighbours with k_frac ≥
    k_hit. This is the ONLY FFR formula; both transductive and projection paths
    call it so they are guaranteed identical."""
    m = len(hi_top)
    return float(np.mean([len(np.intersect1d(hi_top[i], lo_frac[i])) / k_hit
                          for i in range(m)])) if m else float("nan")


def recall_at_k_from_neighbors(hi_top: np.ndarray, lo_top: np.ndarray, k_hit: int) -> float:
    """recall@k = mean_i |hi_top[i] ∩ lo_top[i][:k_hit]| / k_hit. Distinct from FFR:
    the low-D set is truncated to k_hit, so it is strictly ≤ FFR (P0.4)."""
    m = len(hi_top)
    return float(np.mean([len(np.intersect1d(hi_top[i], lo_top[i][:k_hit])) / k_hit
                          for i in range(m)])) if m else float("nan")


# ── loaders ──────────────────────────────────────────────────────────────────────

def load_coords(path: str):
    """Load a coords parquet. Returns (Z_all_dims float32 [N,d], ids or None).

    Preserves EVERY coordinate dimension (x,y[,z,…]) so density/projection work on
    3D maps (P0.5). If BOTH ``ls_index`` and ``row_id`` are present they must agree
    (P0-C — no silent choice). Validates id uniqueness/integrality."""
    import pandas as pd
    df = pd.read_parquet(path)
    have = [c for c in ("ls_index", "row_id") if c in df.columns]
    ids = None
    if len(have) == 2:
        a = df["ls_index"].values.astype("int64"); b = df["row_id"].values.astype("int64")
        if not np.array_equal(a, b):
            raise ValueError(f"{path}: ls_index and row_id disagree; choose one explicitly (P0-C).")
        ids = a
    elif len(have) == 1:
        ids = df[have[0]].values.astype("int64")
    coord_cols = [c for c in df.columns if c not in ("ls_index", "row_id")]
    order = [c for c in ["x", "y", "z"] if c in coord_cols] + \
            sorted([c for c in coord_cols if c not in ("x", "y", "z")])
    Z = df[order].values.astype("float32")
    if ids is not None:
        _check_ids(ids, len(ids), name="coord ids")
    return Z, ids


def _raw_shard_len(path: str, dim: int) -> int:
    sz = os.path.getsize(path)
    if sz % (dim * 4) != 0:
        raise ValueError(f"{path}: size {sz} not divisible by dim*4={dim*4} — "
                         f"trailing bytes, wrong dim, or truncated shard (P0-C).")
    return sz // (dim * 4)


class _LazyConcat:
    """Read-only lazy concatenation of memmapped shards (raw-headerless float32 or
    .npy) that supports ``len``, ``[i:j]``, and fancy ``[idx_array]`` and returns
    float32 arrays — used so the 150M substrate never lands in RAM at once (P0-C)."""
    def __init__(self, mms, dim):
        self.mms = mms
        self.dim = dim
        self.lens = np.array([len(m) for m in mms], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lens)])
        self.N = int(self.offsets[-1])

    def __len__(self):
        return self.N

    @property
    def shape(self):
        return (self.N, self.dim)

    def _slice(self, s, e):
        parts = []
        for k, m in enumerate(self.mms):
            o0, o1 = int(self.offsets[k]), int(self.offsets[k + 1])
            if e <= o0 or s >= o1:
                continue
            a = max(s, o0) - o0; b = min(e, o1) - o0
            parts.append(np.asarray(m[a:b], dtype=np.float32))
        return np.concatenate(parts, axis=0) if parts else np.empty((0, self.dim), np.float32)

    def __getitem__(self, key):
        if isinstance(key, slice):
            s, e, st = key.indices(self.N)
            assert st == 1
            return self._slice(s, e)
        idx = np.asarray(key, dtype=np.int64)
        if idx.ndim == 0:
            return self._slice(int(idx), int(idx) + 1)[0]
        which = np.searchsorted(self.offsets, idx, side="right") - 1
        out = np.empty((len(idx), self.dim), dtype=np.float32)
        for k in np.unique(which):
            sel = which == k
            local = idx[sel] - int(self.offsets[k])
            out[sel] = np.asarray(self.mms[k][local], dtype=np.float32)
        return out


def load_embeddings(path, dim: int | None = None):
    """Memmap embeddings. ``path`` may be a single file, a directory of shards, or
    an ordered list of shard paths. Supports .npy AND raw-headerless float32
    (MiniLM). For raw shards ``dim`` is required; every shard's byte size must be
    divisible by ``dim*4`` (no silent trailing bytes). Returns a memmap (single
    .npy) or a lazy concat (multi-shard) — never a full in-RAM copy (P0-C)."""
    if isinstance(path, (list, tuple)):
        shards = list(path)
    elif os.path.isdir(path):
        shards = sorted(glob.glob(os.path.join(path, "*.npy"))) or \
                 sorted(glob.glob(os.path.join(path, "*.bin"))) or \
                 sorted(glob.glob(os.path.join(path, "*")))
        shards = [s for s in shards if os.path.isfile(s)]
        if not shards:
            raise ValueError(f"{path}: no shards found")
    else:
        shards = [path]
    mms = []
    for s in shards:
        with open(s, "rb") as fh:
            magic = fh.read(6)
        if magic == b"\x93NUMPY":
            mms.append(np.load(s, mmap_mode="r"))
        else:
            if dim is None:
                raise ValueError(f"{s} is raw-headerless; `dim` is required (P0-C).")
            n = _raw_shard_len(s, dim)
            mms.append(np.memmap(s, dtype=np.float32, mode="r", shape=(n, dim)))
    if len(mms) == 1:
        return mms[0]
    d = mms[0].shape[1]
    if any(m.shape[1] != d for m in mms):
        raise ValueError("shards have inconsistent dim")
    return _LazyConcat(mms, d)


# ── id handling (P0-C: exact, never sort-inferred) ───────────────────────────────

def _ids_hash(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:12]


def _check_ids(ids: np.ndarray, n_universe: int, name="ids") -> None:
    ids = np.asarray(ids)
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{name}: non-integral ids (P0-C).")
    if ids.min() < 0:
        raise ValueError(f"{name}: negative id {int(ids.min())} (P0-C).")
    if ids.max() >= n_universe:
        raise ValueError(f"{name}: id {int(ids.max())} >= universe {n_universe} (P0-C).")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate ids (P0-C).")


def align_x_to_z(X, Z, x_ids, z_ids):
    """Return (Xa, aligned_ids, alignment_note) with Xa row-aligned to Z. Exact:

      - x_ids None, z_ids None  → X and Z assumed already 1:1 (same length).
      - x_ids None, z_ids given → Z rows index full X; gather Xa = X[z_ids].
      - x_ids given, z_ids given → same universe required; X reordered to Z.
      - x_ids given, z_ids None → error (cannot align a subset without z_ids).
    """
    n_full = len(X)
    if x_ids is None and z_ids is None:
        if len(X) != len(Z):
            raise ValueError(f"len(X)={len(X)} != len(Z)={len(Z)} and no ids to align (P0-C).")
        return X, None, "positional_1to1"
    if x_ids is None and z_ids is not None:
        _check_ids(z_ids, n_full, name="z_ids(into X)")
        if len(z_ids) != len(Z):
            raise ValueError(f"z_ids len {len(z_ids)} != len(Z) {len(Z)} (P0-C).")
        return X[np.asarray(z_ids, np.int64)], np.asarray(z_ids, np.int64), "gather_X_by_z_ids"
    if x_ids is not None and z_ids is None:
        raise ValueError("x_ids given but z_ids missing: cannot align without coord ids (P0-C).")
    # both present: same universe, reorder X rows into Z order
    x_ids = np.asarray(x_ids, np.int64); z_ids = np.asarray(z_ids, np.int64)
    _check_ids(x_ids, max(int(x_ids.max()) + 1, len(X)), name="x_ids")
    _check_ids(z_ids, max(int(z_ids.max()) + 1, len(X)), name="z_ids")
    if len(x_ids) != len(X):
        raise ValueError(f"x_ids len {len(x_ids)} != len(X) {len(X)} (P0-C).")
    if len(z_ids) != len(Z):
        raise ValueError(f"z_ids len {len(z_ids)} != len(Z) {len(Z)} (P0-C).")
    if set(x_ids.tolist()) != set(z_ids.tolist()):
        raise ValueError("x_ids and z_ids describe different row universes (P0-C).")
    pos = {int(i): r for r, i in enumerate(x_ids)}       # id → row in X (no sorting)
    take = np.fromiter((pos[int(i)] for i in z_ids), dtype=np.int64, count=len(z_ids))
    return X[take], z_ids, "reorder_X_to_z"


# ── bounded-memory kNN with overselect + exact rerank (P0-C) ──────────────────────

def _self_knn(F, anchor_idx, k, cfg: PanelV2Config, hi_dim=True, want_dist=False):
    """Top-k self-excluded neighbours of ``anchor_idx`` within corpus F.

    hi_dim: fast normalised-matmul expansion selects ``k+overselect`` candidates
    per corpus chunk; after streaming all chunks the candidate vectors are gathered
    and distances recomputed EXACTLY (fp32 subtraction) to rerank and to report
    radii. Returns (ids[m,k], dist[m,k] or None, guard) where guard carries the
    minimum boundary gap (kept k-th exact dist vs first dropped) across anchors.

    low_dim: single deterministic pass over the whole (tiny) coord corpus — no
    chunk-boundary tie perturbation; distances are already exact."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N = len(F); m = len(anchor_idx)
    kk = k + 1                                            # +1 for self
    if hi_dim:
        cand = k + max(cfg.overselect, 1) + 1
        cchunk = min(N, max(1, int(cfg.corpus_chunk)))
    else:
        cand = kk
        cchunk = N                                       # one deterministic pass
    achunk = max(1, min(m, int(cfg.block_elems // max(1, cchunk))))
    out_i = np.empty((m, k), dtype=np.int64)
    out_d = np.empty((m, k), dtype=np.float64) if want_dist else None
    min_gap = float("inf")
    for a0 in range(0, m, achunk):
        aids = anchor_idx[a0:a0 + achunk]
        Q = torch.from_numpy(np.asarray(F[aids], dtype=np.float32)).to(dev)
        ma, D = Q.shape
        qn = (Q * Q).sum(1, keepdim=True) if hi_dim else None
        best_d = torch.full((ma, cand), float("inf"), device=dev)
        best_i = torch.full((ma, cand), -1, dtype=torch.long, device=dev)
        for j in range(0, N, cchunk):
            Xc = torch.from_numpy(np.asarray(F[j:j + cchunk], dtype=np.float32)).to(dev)
            if hi_dim:
                d2 = qn - 2.0 * (Q @ Xc.T) + (Xc * Xc).sum(1)     # fast expansion (rank only)
            else:
                d2 = torch.zeros((ma, len(Xc)), device=dev)
                for c in range(D):
                    diff = Q[:, c:c + 1] - Xc[:, c]
                    d2.addcmul_(diff, diff)
            kloc = min(cand, len(Xc))
            ld, li = torch.topk(d2, kloc, dim=1, largest=False)
            li = li + j
            best_d = torch.cat([best_d, ld], 1)
            best_i = torch.cat([best_i, li], 1)
            best_d, sel = torch.topk(best_d, cand, dim=1, largest=False)
            best_i = torch.gather(best_i, 1, sel)
            del Xc, d2
        if hi_dim:
            # EXACT rerank: gather candidate vectors, recompute fp32 distances.
            flat = best_i.reshape(-1).cpu().numpy()
            nb = torch.from_numpy(np.asarray(F[flat], dtype=np.float32)).to(dev).reshape(ma, cand, -1)
            exact = (nb - Q[:, None, :]).float().pow(2).sum(2)   # (ma, cand) squared L2
            exact = torch.where(best_i >= 0, exact, torch.full_like(exact, float("inf")))
            ed, es = torch.sort(exact, dim=1)
            best_i = torch.gather(best_i, 1, es)
            best_d = ed
            del nb, exact
        ids = best_i.cpu().numpy()
        dist = best_d.clamp_min(0).sqrt().cpu().numpy() if want_dist or hi_dim else None
        for r in range(ma):
            keep = ids[r] != aids[r]
            row = ids[r][keep]
            drow = dist[r][keep] if dist is not None else None
            out_i[a0 + r] = row[:k] if len(row) >= k else ids[r][:k]
            if want_dist:
                out_d[a0 + r] = drow[:k] if (drow is not None and len(drow) >= k) else dist[r][:k]
            if hi_dim and drow is not None and len(drow) > k:
                min_gap = min(min_gap, float(drow[k] - drow[k - 1]))   # boundary safety
        del Q, best_d, best_i
    guard = {"boundary_min_gap": None if min_gap == float("inf") else round(min_gap, 6),
             "overselect": cfg.overselect} if hi_dim else {}
    return out_i, out_d, guard


# ── anchors + masks ──────────────────────────────────────────────────────────────

def sample_anchors(n, cfg: PanelV2Config):
    rng = np.random.RandomState(cfg.anchor_seed)
    return np.sort(rng.choice(n, min(cfg.n_anchors, n), replace=False))


def _resolve_mask(mask, m):
    """Normalise a per-metric mask (bool len-m, or integer indices into the anchor
    array, or None → all) to an integer selection array."""
    if mask is None:
        return np.arange(m)
    a = np.asarray(mask)
    if a.dtype == bool:
        if len(a) != m:
            raise ValueError(f"boolean mask len {len(a)} != n_anchors {m} (P0-C).")
        return np.nonzero(a)[0]
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError("mask must be boolean or integer indices (P0-C).")
    if a.size and (a.min() < 0 or a.max() >= m):
        raise ValueError("mask indices out of anchor range (P0-C).")
    return a


# ── guards ───────────────────────────────────────────────────────────────────────

def _data_guards(Xa, Z):
    """Finite / norm / collapse / radius-order sanity on a small sample."""
    g = {}
    zs = np.asarray(Z[:min(len(Z), 4096)], dtype=np.float64)
    g["coords_finite"] = bool(np.isfinite(zs).all())
    g["coords_collapsed"] = bool(np.allclose(zs.std(0), 0))
    xs = np.asarray(Xa[:min(len(Xa), 4096)], dtype=np.float64)
    g["emb_finite"] = bool(np.isfinite(xs).all())
    norms = np.linalg.norm(xs, axis=1)
    g["emb_norm_mean"] = round(float(norms.mean()), 4)
    g["emb_zero_rows"] = int((norms == 0).sum())
    return g


# ── provenance ───────────────────────────────────────────────────────────────────

def _git_state():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                         text=True, timeout=10).strip()[:12]
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"],
                                             text=True, timeout=10).strip())
        return commit, dirty
    except Exception:
        return None, None


def _peak_rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2), 3)


# ── projection fidelity (P0-C) ───────────────────────────────────────────────────

def score_projection(Xa, Z, cfg: PanelV2Config, projection: dict, x_ids=None):
    """Out-of-sample FFR: high-D query→corpus top-k_hit vs projected query→map
    top-k_frac, via the SAME ffr formula. ``projection`` carries:
      Xq (nq, D) held-out query embeddings, Zq (nq, d) their projected coords,
      query_ids (nq,) ids proven disjoint from the training rows, and optional
      checkpoint/query fingerprints. Returns metrics + provenance."""
    import torch
    Xq = np.asarray(projection["Xq"], dtype=np.float32)
    Zq = np.asarray(projection["Zq"], dtype=np.float32)
    qids = np.asarray(projection.get("query_ids", np.arange(len(Xq))), np.int64)
    if len(Xq) != len(Zq) or len(Xq) != len(qids):
        raise ValueError("projection Xq/Zq/query_ids length mismatch (P0-C).")
    # PROOF that queries are out-of-sample: their ids must not be training rows.
    train_ids = set((np.asarray(x_ids, np.int64) if x_ids is not None
                     else np.arange(len(Xa))).tolist())
    overlap = int(sum(1 for q in qids if int(q) in train_ids))
    if overlap:
        raise ValueError(f"{overlap} projection query ids overlap training rows — "
                         f"not held-out (P0-C).")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * len(Z))))

    def cross_topk(Q, corpus, ksel, lo):
        Qt = torch.from_numpy(np.asarray(Q, dtype=np.float32)).to(dev)
        out = np.empty((len(Q), ksel), dtype=np.int64)
        best_d = torch.full((len(Q), ksel), float("inf"), device=dev)
        best_i = torch.full((len(Q), ksel), -1, dtype=torch.long, device=dev)
        cc = len(corpus) if lo else min(len(corpus), cfg.corpus_chunk)
        qn = (Qt * Qt).sum(1, keepdim=True)
        for j in range(0, len(corpus), cc):
            Xc = torch.from_numpy(np.asarray(corpus[j:j + cc], dtype=np.float32)).to(dev)
            if lo:
                d2 = torch.cdist(Qt, Xc) ** 2
            else:
                d2 = qn - 2.0 * (Qt @ Xc.T) + (Xc * Xc).sum(1)
            kloc = min(ksel, len(Xc))
            ld, li = torch.topk(d2, kloc, dim=1, largest=False)
            best_d = torch.cat([best_d, ld], 1); best_i = torch.cat([best_i, li + j], 1)
            best_d, sel = torch.topk(best_d, ksel, dim=1, largest=False)
            best_i = torch.gather(best_i, 1, sel); del Xc, d2
        return best_i.cpu().numpy()

    hi = cross_topk(Xq, Xa, cfg.k_hit, lo=False)     # query → high-D corpus
    lo = cross_topk(Zq, Z, kf, lo=True)              # projected query → map
    ffr = ffr_from_neighbors(hi, lo, cfg.k_hit)
    r_at = recall_at_k_from_neighbors(hi, lo, cfg.k_hit)
    return {"proj_ffr": round(ffr, 4), "proj_recall@k": round(r_at, 5),
            "proj_n_queries": int(len(Xq)), "proj_k_frac": kf,
            "proj_query_hash": _ids_hash(qids),
            "proj_checkpoint": projection.get("checkpoint_hash"),
            "proj_convention": projection.get("convention", "l2")}


# ── canonical entry point (P0-C) ─────────────────────────────────────────────────

def score_panel(X, Z, *, config: PanelV2Config, x_ids=None, z_ids=None,
                centroids_by_k=None, anchor_masks=None, projection=None,
                provenance):
    """The single evaluator both the runner and CLI call. Aligns X to Z exactly,
    runs ONE high-D and ONE low-D neighbour pass shared across ffr/recall/purity,
    an exact-radius pass for density, optional projection fidelity, and emits a
    formula-versioned payload with mandatory provenance and guard outcomes.

    ``anchor_masks`` (dict, optional) gives per-metric anchor selections:
    ``{"ffr": mask, "purity": mask, "density": mask}`` where each mask is a
    boolean (len n_anchors) or integer index array; missing keys use all anchors.
    For the mixed-corpus protocol, e.g. ffr over all anchors, purity over FineWeb
    anchors only — expressed directly here.

    ``provenance`` is REQUIRED: a dict of caller-supplied fingerprints (data,
    coord, checkpoint, query paths/hashes). Computed provenance (git, memory,
    device, alignment, guards) is added here."""
    import torch
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cfg = config
    Xa, aligned_ids, align_note = align_x_to_z(X, Z, x_ids, z_ids)
    n = len(Z)
    if len(Xa) != n:
        raise ValueError(f"post-align len(Xa)={len(Xa)} != len(Z)={n} (P0-C).")

    aidx = sample_anchors(n, cfg)
    m = len(aidx)
    masks = anchor_masks or {}
    sel_ffr = _resolve_mask(masks.get("ffr"), m)
    sel_pur = _resolve_mask(masks.get("purity"), m)
    sel_den = _resolve_mask(masks.get("density"), m)

    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * n)))
    # ONE hi pass (k_frac) + ONE lo pass (k_frac) shared by ffr/recall/purity.
    hi_kf, _, guard_hi = _self_knn(Xa, aidx, kf, cfg, hi_dim=True)
    lo_kf, _, _ = _self_knn(Z, aidx, kf, cfg, hi_dim=False)
    hi_hit = hi_kf[:, :cfg.k_hit]

    res = {"schema": PANEL_SCHEMA, "formula_version": cfg.formula_version,
           "n": int(n), "n_dims_hi": int(np.asarray(Xa[:1]).shape[1]),
           "n_dims_lo": int(Z.shape[1]), "frac": cfg.frac, "k_hit": cfg.k_hit,
           "k_frac": kf, "k_density": cfg.k_density,
           "anchor_seed": cfg.anchor_seed, "n_anchors": m,
           "anchor_hash": _ids_hash(aidx)}

    # ffr + recall@k (separate masks, separate numbers — P0.4)
    res["ffr"] = round(ffr_from_neighbors(hi_hit[sel_ffr], lo_kf[sel_ffr], cfg.k_hit), 4)
    res["recall@k"] = round(recall_at_k_from_neighbors(hi_hit[sel_ffr], lo_kf[sel_ffr], cfg.k_hit), 5)
    res["n_ffr_anchors"] = int(len(sel_ffr))

    # purity per centroid granularity, with centroid provenance
    if centroids_by_k:
        res["purity"] = {}
        res["centroid_hashes"] = {}
        lab = _label_by_centroids(Xa, centroids_by_k)   # {k: labels[N]}
        for kc, labels in lab.items():
            alab = labels[aidx][sel_pur]
            hd = float((labels[hi_kf[sel_pur]] == alab[:, None]).mean())
            mp = float((labels[lo_kf[sel_pur]] == alab[:, None]).mean())
            res["purity"][f"k{kc}"] = round(mp / hd, 4) if hd else None
            res["centroid_hashes"][f"k{kc}"] = _ids_hash(
                np.round(np.asarray(centroids_by_k[kc], np.float32), 4))
        res["n_purity_anchors"] = int(len(sel_pur))

    # density: exact-radius pass (k_density), hiD vs loD
    _, hd_r, guard_den = _self_knn(Xa, aidx[sel_den], cfg.k_density, cfg, hi_dim=True, want_dist=True)
    _, ld_r, _ = _self_knn(Z, aidx[sel_den], cfg.k_density, cfg, hi_dim=False, want_dist=True)
    r_hd = hd_r.mean(1); r_ld = ld_r.mean(1); eps = 1e-12
    res["density"] = round(float(np.corrcoef(np.log(r_hd + eps), np.log(r_ld + eps))[0, 1]), 4)
    res["n_density_anchors"] = int(len(sel_den))

    # projection (out-of-sample) — kept in its own labelled keys, never merged
    # into the transductive ffr column (P0-C).
    if projection is not None:
        res["projection"] = score_projection(Xa, Z, cfg, projection, x_ids=aligned_ids)

    commit, dirty = _git_state()
    res["provenance"] = {
        **(provenance or {}),
        "code_commit": commit, "code_dirty": dirty,
        "alignment": align_note,
        "aligned_ids_hash": _ids_hash(aligned_ids) if aligned_ids is not None else None,
        "exactness": "hi:overselect+exact_rerank; lo:single_pass_exact",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "peak_gpu_gb": (round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
                        if torch.cuda.is_available() else None),
        "peak_rss_gb": _peak_rss_gb(),
        "wall_s": round(time.time() - t0, 2),
    }
    res["guards"] = {**_data_guards(Xa, Z), **guard_hi, "density_guard": guard_den}
    return res


def _label_by_centroids(Xa, centroids_by_k):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(Xa); out = {}
    for kc, C in centroids_by_k.items():
        Ct = torch.from_numpy(np.asarray(C, dtype=np.float32)).to(dev)
        lab = np.empty(n, dtype=np.int32)
        for i in range(0, n, 65536):
            xb = torch.from_numpy(np.asarray(Xa[i:i + 65536], dtype=np.float32)).to(dev)
            lab[i:i + 65536] = torch.cdist(xb, Ct).argmin(1).cpu().numpy()
        out[kc] = lab; del Ct
    return out


# ── back-compat shim for existing callers/tests ──────────────────────────────────

def run_panel(X, Z, cfg: PanelV2Config, centroids=None, ids=None, restrict=None,
              manifest_extra=None):
    """Deprecated thin wrapper over :func:`score_panel` kept so older call sites
    keep working. ``centroids`` (list) → ``centroids_by_k`` keyed by len; a single
    ``restrict`` maps to the ffr+purity masks (density unrestricted, as before)."""
    cbk = {len(c): c for c in centroids} if centroids else None
    masks = {"ffr": restrict, "purity": restrict} if restrict is not None else None
    return score_panel(X, Z, config=cfg, z_ids=(ids if ids is not None else None),
                        centroids_by_k=cbk, anchor_masks=masks,
                        provenance=manifest_extra or {"caller": "run_panel_shim"})


# ── CLI (P0-C: same core as the runner) ──────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Panel v2 canonical evaluator")
    ap.add_argument("--emb", required=True, help="file | dir | comma-list of embedding shards")
    ap.add_argument("--coords", required=True, help="coords parquet")
    ap.add_argument("--dim", type=int, default=None, help="embedding dim (raw shards)")
    ap.add_argument("--centroids", default=None, help="comma-list of .npy centroid files")
    ap.add_argument("--frac", type=float, default=D_DEFAULT_FRAC)
    ap.add_argument("--n-anchors", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = PanelV2Config(frac=args.frac, n_anchors=args.n_anchors)
    emb = args.emb.split(",") if "," in args.emb else args.emb
    X = load_embeddings(emb, dim=args.dim)
    Z, z_ids = load_coords(args.coords)
    cbk = None
    if args.centroids:
        cbk = {}
        for p in args.centroids.split(","):
            C = np.load(p); cbk[len(C)] = C
    prov = {"emb": args.emb, "coords": args.coords, "coords_sha": _file_sha(args.coords)}
    res = score_panel(X, Z, config=cfg, z_ids=z_ids, centroids_by_k=cbk, provenance=prov)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def _file_sha(path, cap=1 << 20):
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            h.update(f.read(cap))
        return h.hexdigest()[:16]
    except Exception:
        return None


if __name__ == "__main__":
    main()
