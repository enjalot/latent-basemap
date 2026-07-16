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

from .artifact_identity import (canonical_json, ordered_array_sha256, path_signature,
                                sha256_bytes)

FORMULA_VERSION = "panel_v2.2-2026-07-15"   # exact top-k_hit truth + approx k_frac membership; byte-capped rerank
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
    # S2.4: profiled default. The 2M-row tile made A3 scoring take ~292 s/map;
    # the 500k tile made the otherwise-comparable bridge scorer ~94 s/map at the
    # same peak accuracy. 500k is the production default; do not leave tile size
    # to per-script guesswork. Raise only with a matching peak_byte_cap check.
    corpus_chunk: int = 500_000            # rows streamed per corpus chunk (hiD)
    overselect: int = 8                    # extra candidates kept before exact rerank
    block_elems: int = 500_000_000         # per-block element cap (~2 GB fp32)
    rerank_byte_cap: int = 2_000_000_000   # P2: cap the exact-rerank gather tile (~2 GB)
    rerank_scratch: float = 3.0            # scratch multiplier for the rerank alloc
    peak_byte_cap: int = 26_000_000_000    # P2: refuse a scoring stage above this (~26 GB)
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
        a = _coerce_int_ids(df["ls_index"].values, "ls_index")   # integral-check BEFORE cast
        b = _coerce_int_ids(df["row_id"].values, "row_id")
        if not np.array_equal(a, b):
            raise ValueError(f"{path}: ls_index and row_id disagree; choose one explicitly (P0-C).")
        ids = a
    elif len(have) == 1:
        ids = _coerce_int_ids(df[have[0]].values, have[0])
    coord_cols = [c for c in df.columns if c not in ("ls_index", "row_id")]
    order = [c for c in ["x", "y", "z"] if c in coord_cols] + \
            sorted([c for c in coord_cols if c not in ("x", "y", "z")])
    Z = df[order].values.astype("float32")
    if ids is not None:
        # nonneg + unique + integral, but NOT bounded by len(coords): a valid
        # offset subset (e.g. rows 10..12 of a larger X) must load (P0-4). The
        # universe bound is checked at alignment against X.
        _check_ids_no_universe(ids, name="coord ids")
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
    def __init__(self, mms, dim, shard_paths=None):
        self.mms = mms
        self.dim = dim
        self.shard_paths = [os.path.realpath(p) for p in (shard_paths or [])]
        self.lens = np.array([len(m) for m in mms], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lens)])
        self.N = int(self.offsets[-1])

    def __len__(self):
        return self.N

    @property
    def shape(self):
        return (self.N, self.dim)

    @property
    def dtype(self):
        return self.mms[0].dtype

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
    return _LazyConcat(mms, d, shard_paths=shards)


# ── id handling (P0-C: exact, never sort-inferred) ───────────────────────────────

def _ids_hash(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:12]


def _coerce_int_ids(values, name="ids") -> np.ndarray:
    """Validate that ``values`` are INTEGRAL before any int cast (P0-4): a float
    column like [0.2,1.2,2.2] must FAIL, not be silently floored to [0,1,2].
    Returns an int64 array."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64)
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all() or not np.all(arr == np.floor(arr)):
            raise ValueError(f"{name}: non-integral / non-finite ids (P0-4).")
        return arr.astype(np.int64)
    raise ValueError(f"{name}: ids must be integer or integral-float, got dtype {arr.dtype} (P0-4).")


def _check_ids_no_universe(ids: np.ndarray, name="ids") -> None:
    """Nonnegative + unique + integral, WITHOUT assuming a universe size (P0-4):
    a valid subset like [10,11,12] with 3 rows must NOT be rejected. The universe
    bound is enforced later at alignment against X."""
    ids = _coerce_int_ids(ids, name)
    if ids.min() < 0:
        raise ValueError(f"{name}: negative id {int(ids.min())} (P0-4).")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate ids (P0-4).")


def _check_ids(ids: np.ndarray, n_universe: int, name="ids") -> None:
    ids = _coerce_int_ids(ids, name)
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
        z_ids = _coerce_int_ids(z_ids, "z_ids")          # validate integral BEFORE cast
        _check_ids(z_ids, n_full, name="z_ids(into X)")
        if len(z_ids) != len(Z):
            raise ValueError(f"z_ids len {len(z_ids)} != len(Z) {len(Z)} (P0-C).")
        return X[z_ids], z_ids, "gather_X_by_z_ids"
    if x_ids is not None and z_ids is None:
        raise ValueError("x_ids given but z_ids missing: cannot align without coord ids (P0-C).")
    # both present: same universe, reorder X rows into Z order
    x_ids = _coerce_int_ids(x_ids, "x_ids"); z_ids = _coerce_int_ids(z_ids, "z_ids")
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

def estimate_panel_peak_bytes(cfg: PanelV2Config, n_dims: int, k_frac: int) -> dict:
    """Shape-only peak-byte estimate per scoring stage (P2). Computes the dominant
    allocations from (corpus_chunk, block_elems, k_frac, k_hit, overselect, D)
    WITHOUT touching the GPU — so a stage that would OOM is caught before any
    coordinate is loaded. Returns bytes per stage + the dominant term."""
    cc = min(cfg.corpus_chunk, cfg.block_elems)      # corpus rows resident per chunk
    # exact top-k_hit rerank gather (byte-capped achunk × cand × D)
    cand_hit = cfg.k_hit + cfg.overselect + 1
    achunk = max(1, min(int(cfg.block_elems // max(1, cc)),
                        int(cfg.rerank_byte_cap // (cand_hit * n_dims * 4 * cfg.rerank_scratch))))
    rerank_hit = achunk * cand_hit * n_dims * 4 * cfg.rerank_scratch
    corpus_tile = cc * n_dims * 4                    # a streamed hi-D corpus chunk
    # approximate k_frac membership: distance matrix (achunk_frac × cc)
    achunk_frac = max(1, int(cfg.block_elems // max(1, cc)))
    dmat_frac = achunk_frac * cc * 4
    stages = {
        "hi_k_hit_rerank": int(rerank_hit),
        "hi_k_frac_dmat": int(dmat_frac),
        "corpus_chunk": int(corpus_tile),
    }
    dom = max(stages, key=stages.get)
    return {"stages": stages, "dominant": dom, "dominant_bytes": stages[dom],
            "cap": int(cfg.peak_byte_cap)}


def _peak_byte_preflight(cfg: PanelV2Config, n_dims: int, k_frac: int) -> None:
    est = estimate_panel_peak_bytes(cfg, n_dims, k_frac)
    if est["dominant_bytes"] > est["cap"]:
        raise MemoryError(
            f"panel peak-byte preflight: stage '{est['dominant']}' needs "
            f"{est['dominant_bytes']/1e9:.1f} GB > cap {est['cap']/1e9:.1f} GB (P2). "
            f"Lower n_anchors won't help (per-tile); raise corpus_chunk/rerank_byte_cap "
            f"or peak_byte_cap deliberately. stages={ {k: round(v/1e9,2) for k,v in est['stages'].items()} }")


def _self_knn(F, anchor_idx, k, cfg: PanelV2Config, hi_dim=True, want_dist=False, exact=True):
    """Top-k self-excluded neighbours of ``anchor_idx`` within corpus F (panel v2.2).

    hi_dim + exact: fast normalised-matmul expansion selects ``k+overselect``
    candidates per corpus chunk, then the candidate vectors are gathered and
    distances recomputed EXACTLY (fp32) to rerank / report radii. This is the
    high-D top-k_hit TRUTH (FFR/recall) and the density-radius path. The rerank
    gather (achunk × cand × D) is BYTE-CAPPED — `block_elems`/`corpus_chunk` does
    NOT bound it, which is what OOM'd at k_frac≈8000 (v2.1).

    hi_dim + not exact: fast-expansion top-k with NO rerank — the k_frac MEMBERSHIP
    pass (purity). Order inside the set is irrelevant to a label count; boundary
    membership is approximate and labelled so. want_dist REQUIRES exact.

    low_dim: single deterministic exact pass over the (tiny) coord corpus."""
    import torch
    if want_dist and not exact:
        raise ValueError("radii (want_dist) require exact reranking (P2)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N = len(F); m = len(anchor_idx)
    kk = k + 1                                            # +1 for self
    if hi_dim:
        cand = (k + max(cfg.overselect, 1) + 1) if exact else kk
        cchunk = min(N, max(1, int(cfg.corpus_chunk)))
    else:
        cand = kk
        cchunk = N                                       # one deterministic pass
    achunk = max(1, min(m, int(cfg.block_elems // max(1, cchunk))))
    if hi_dim and exact:
        # BYTE-CAP the exact-rerank tile: achunk × cand × D × 4 × scratch. This is
        # the allocation `corpus_chunk` never bounded (v2.1 OOM at 24 GB).
        D = int(np.asarray(F[anchor_idx[:1]]).shape[1])
        cap_rows = max(1, int(cfg.rerank_byte_cap // (cand * D * 4 * cfg.rerank_scratch)))
        achunk = max(1, min(achunk, cap_rows))
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
        if hi_dim and exact:
            # EXACT rerank (byte-capped achunk): gather candidate vectors, recompute
            # fp32 distances. Used for the top-k_hit TRUTH (FFR) and density radii —
            # both small k, so the gather is tiny. NOT used for k_frac membership.
            flat = best_i.reshape(-1).cpu().numpy()
            nb = torch.from_numpy(np.asarray(F[flat], dtype=np.float32)).to(dev).reshape(ma, cand, -1)
            ex = (nb - Q[:, None, :]).float().pow(2).sum(2)      # (ma, cand) squared L2
            ex = torch.where(best_i >= 0, ex, torch.full_like(ex, float("inf")))
            ed, es = torch.sort(ex, dim=1)
            best_i = torch.gather(best_i, 1, es)
            best_d = ed
            del nb, ex
        ids = best_i.cpu().numpy()
        dist = best_d.clamp_min(0).sqrt().cpu().numpy() if want_dist else None
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

def cross_knn(Q, corpus, k, cfg: PanelV2Config, hi_dim=True, q_tile=4096, exact=True):
    """Canonical tiled top-k of each query row over a corpus (cross, not self).
    Tiles BOTH the query rows (q_tile) and the corpus (cfg.corpus_chunk) so a full
    (all-queries × corpus) matrix is never materialised. The ONE cross-neighbour
    implementation for projection + the decision scorer. hi_dim + exact: overselect
    fast-expansion candidates, then EXACT fp32 rerank (byte-capped q_tile) — so
    high-D projection top-k_hit is exact truth, not the v2.1 fast-only IDs (P2)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cc = min(len(corpus), max(1, cfg.corpus_chunk))
    cand = (k + cfg.overselect + 1) if (hi_dim and exact) else k
    # cap q_tile by the DISTANCE-MATRIX size (q_tile × cc), not just the rerank
    # gather — otherwise q_tile × corpus_chunk OOMs (the golden projection OOM).
    q_tile = max(1, min(q_tile, int(cfg.block_elems // cc)))
    if hi_dim and exact:
        D = int(np.asarray(Q[:1]).shape[1])
        q_tile = max(1, min(q_tile, int(cfg.rerank_byte_cap // (cand * D * 4 * cfg.rerank_scratch))))
    out = np.empty((len(Q), k), dtype=np.int64)
    for q0 in range(0, len(Q), q_tile):
        Qt = torch.from_numpy(np.asarray(Q[q0:q0 + q_tile], dtype=np.float32)).to(dev)
        qn = (Qt * Qt).sum(1, keepdim=True)
        best_d = torch.full((len(Qt), cand), float("inf"), device=dev)
        best_i = torch.full((len(Qt), cand), -1, dtype=torch.long, device=dev)
        for j in range(0, len(corpus), cc):
            Xc = torch.from_numpy(np.asarray(corpus[j:j + cc], dtype=np.float32)).to(dev)
            d2 = (qn - 2.0 * (Qt @ Xc.T) + (Xc * Xc).sum(1)) if hi_dim else (torch.cdist(Qt, Xc) ** 2)
            kloc = min(cand, len(Xc))
            ld, li = torch.topk(d2, kloc, dim=1, largest=False)
            best_d = torch.cat([best_d, ld], 1); best_i = torch.cat([best_i, li + j], 1)
            best_d, sel = torch.topk(best_d, cand, dim=1, largest=False)
            best_i = torch.gather(best_i, 1, sel); del Xc, d2
        if hi_dim and exact:            # exact fp32 rerank of the candidate pool
            flat = best_i.reshape(-1).cpu().numpy()
            nb = torch.from_numpy(np.asarray(corpus[flat], dtype=np.float32)).to(dev).reshape(len(Qt), cand, -1)
            ex = (nb - Qt[:, None, :]).float().pow(2).sum(2)
            ex = torch.where(best_i >= 0, ex, torch.full_like(ex, float("inf")))
            best_i = torch.gather(best_i, 1, torch.sort(ex, dim=1).indices); del nb, ex
        out[q0:q0 + len(Qt)] = best_i[:, :k].cpu().numpy(); del Qt, best_d, best_i
    return out


def score_projection(Xa, Z, cfg: PanelV2Config, projection: dict, x_ids=None):
    """Out-of-sample FFR: high-D query→corpus top-k_hit vs projected query→map
    top-k_frac, via the SAME ffr formula and the canonical ``cross_knn`` (P0-4).
    ``projection`` carries Xq (nq, D), Zq (nq, d), query_ids (proven disjoint from
    training rows), and optional checkpoint/query fingerprints."""
    Xq = np.asarray(projection["Xq"], dtype=np.float32)
    Zq = np.asarray(projection["Zq"], dtype=np.float32)
    qids = _coerce_int_ids(projection.get("query_ids", np.arange(len(Xq))), "query_ids")
    if len(Xq) != len(Zq) or len(Xq) != len(qids):
        raise ValueError("projection Xq/Zq/query_ids length mismatch (P0-C).")
    if len(np.unique(qids)) != len(qids):
        raise ValueError("projection query_ids contain duplicates (P0-4).")
    # PROOF that queries are out-of-sample: their ids must not be training rows.
    train_ids = set((np.asarray(x_ids, np.int64) if x_ids is not None
                     else np.arange(len(Xa))).tolist())
    overlap = int(sum(1 for q in qids if int(q) in train_ids))
    if overlap:
        raise ValueError(f"{overlap} projection query ids overlap training rows — "
                         f"not held-out (P0-C).")
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * len(Z))))
    hi = cross_knn(Xq, Xa, cfg.k_hit, cfg, hi_dim=True)     # query → high-D corpus
    lo = cross_knn(Zq, Z, kf, cfg, hi_dim=False)            # projected query → map
    ffr = ffr_from_neighbors(hi, lo, cfg.k_hit)
    r_at = recall_at_k_from_neighbors(hi, lo, cfg.k_hit)
    return {"proj_ffr": round(ffr, 4), "proj_recall@k": round(r_at, 5),
            "proj_n_queries": int(len(Xq)), "proj_k_frac": kf,
            "proj_query_hash": _ids_hash(qids),
            "proj_checkpoint": projection.get("checkpoint_hash"),
            "proj_convention": projection.get("convention", "l2")}


# ── canonical entry point (P0-C) ─────────────────────────────────────────────────

def _matrix_identity(F):
    """Full ordered data identity: every shard byte or every in-memory row."""
    shard_paths = list(getattr(F, "shard_paths", []) or [])
    if not shard_paths and isinstance(F, np.memmap) and getattr(F, "filename", None):
        shard_paths = [os.path.realpath(os.fspath(F.filename))]
    shape = [int(v) for v in F.shape]
    dtype = np.dtype(F.dtype if hasattr(F, "dtype") else np.asarray(F[:1]).dtype).str
    if shard_paths:
        shards = []
        for position, path in enumerate(shard_paths):
            sig = path_signature(path)
            shards.append({"position": position, "name": os.path.basename(path),
                           "bytes": sig["bytes"], "sha256": sig["sha256"]})
        return {"kind": "ordered_shards", "shape": shape, "dtype": dtype, "shards": shards}
    return {"kind": "ordered_array", "shape": shape, "dtype": dtype,
            "sha256": ordered_array_sha256(F)}


def _centroid_identities(centroids_by_k):
    identities = {}
    for kc, centroids in sorted((centroids_by_k or {}).items()):
        identities[str(kc)] = {
            "shape": [int(v) for v in np.asarray(centroids).shape],
            "dtype": np.asarray(centroids).dtype.str,
            "sha256": ordered_array_sha256(np.asarray(centroids)),
        }
    return identities


def hiD_reference_key(Xa, aidx, cfg: PanelV2Config, centroids_by_k=None, kf=None,
                      *, query_ids=None, convention=None, data_identity=None,
                      centroid_identities=None):
    """The content-addressed key that makes a hi-D reference REUSABLE only for an
    identical (data, anchors, k-params, formula, centroids) tuple. Any drift ->
    a different key -> the verify below fails closed rather than mis-scoring."""
    anchors = np.ascontiguousarray(np.asarray(aidx, dtype=np.int64))
    queries = (None if query_ids is None else
               np.ascontiguousarray(np.asarray(query_ids, dtype=np.int64)))
    parts = {
        "schema": "hiD_reference_identity.v2",
        "data": data_identity or _matrix_identity(Xa),
        "anchors": {"count": len(anchors), "sha256": sha256_bytes(anchors.tobytes())},
        "queries": (None if queries is None else
                    {"count": len(queries), "sha256": sha256_bytes(queries.tobytes())}),
        "config": dataclasses.asdict(cfg),
        "k_frac_effective": int(kf) if kf is not None else None,
        "formula": cfg.formula_version,
        "convention": convention or {
            "row_order": "ordered corpus rows",
            "distance": "squared L2; exact fp32 rerank for k_hit/density",
            "self_exclusion": True,
            "anchor_namespace": "zero-based corpus row IDs",
        },
        "centroids": centroid_identities or _centroid_identities(centroids_by_k),
    }
    return sha256_bytes(canonical_json(parts)), parts


def build_hiD_reference(Xa, aidx, cfg: PanelV2Config, centroids_by_k=None,
                        *, query_ids=None, convention=None, data_identity=None,
                        centroid_identities=None):
    """Compute every MAP-INDEPENDENT high-D quantity ONCE: exact top-k_hit truth,
    approximate k_frac membership, exact density radii (all anchors), and centroid
    labels. Scoring N maps of the same corpus then reuses this single reference and
    only repeats the cheap low-D passes (S2.5 'done when')."""
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * len(Xa))))
    key, parts = hiD_reference_key(
        Xa, aidx, cfg, centroids_by_k, kf=kf, query_ids=query_ids,
        convention=convention, data_identity=data_identity,
        centroid_identities=centroid_identities)
    hi_hit, _, guard_hit = _self_knn(Xa, aidx, cfg.k_hit, cfg, hi_dim=True, exact=True)
    hi_frac, _, _ = _self_knn(Xa, aidx, kf, cfg, hi_dim=True, exact=False)
    _, hd_r, guard_den = _self_knn(Xa, aidx, cfg.k_density, cfg, hi_dim=True, want_dist=True)
    r_hd = hd_r.mean(1)                              # per-anchor hi-D radius (all anchors)
    labels = _label_by_centroids(Xa, centroids_by_k) if centroids_by_k else {}
    return {"key": key, "key_parts": parts, "kf": int(kf),
            "hi_hit": hi_hit, "hi_frac": hi_frac, "r_hd": r_hd,
            "labels": {int(k): v for k, v in labels.items()},
            "guard_hit": guard_hit, "guard_den": guard_den,
            "formula_version": cfg.formula_version}


def save_hiD_reference(ref, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, key=ref["key"], kf=ref["kf"], hi_hit=ref["hi_hit"],
             hi_frac=ref["hi_frac"], r_hd=ref["r_hd"],
             label_keys=np.array(sorted(ref["labels"]), dtype=np.int64),
             meta=json.dumps({"key_parts": ref["key_parts"], "guard_hit": ref["guard_hit"],
                              "guard_den": ref["guard_den"],
                              "formula_version": ref["formula_version"]}),
             **{f"labels_{k}": v for k, v in ref["labels"].items()})
    return path


def load_hiD_reference(path):
    z = np.load(path if path.endswith(".npz") else path + ".npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    labels = {int(k): z[f"labels_{int(k)}"] for k in z["label_keys"].tolist()}
    return {"key": str(z["key"]), "kf": int(z["kf"]), "hi_hit": z["hi_hit"],
            "hi_frac": z["hi_frac"], "r_hd": z["r_hd"], "labels": labels,
            "key_parts": meta["key_parts"], "guard_hit": meta["guard_hit"],
            "guard_den": meta["guard_den"], "formula_version": meta["formula_version"]}


def _resolve_reference(Xa, aidx, cfg, centroids_by_k, hiD_reference,
                       reference_identity=None):
    """Return a verified reference: compute inline if none given, else CHECK the
    supplied one against a freshly recomputed content key and RAISE on any drift
    (fail-closed, S2.5 'verified')."""
    if hiD_reference is None:
        return build_hiD_reference(Xa, aidx, cfg, centroids_by_k,
                                   **(reference_identity or {})), False
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * len(Xa))))
    key, _ = hiD_reference_key(Xa, aidx, cfg, centroids_by_k, kf=kf,
                               **(reference_identity or {}))
    if key != hiD_reference.get("key"):
        raise ValueError(f"hiD_reference key mismatch: supplied {hiD_reference.get('key')} "
                         f"!= recomputed {key} (data/anchors/params/centroids drifted).")
    if int(hiD_reference.get("kf", -1)) != int(kf):
        raise ValueError("hiD_reference k_frac mismatch (stale reference).")
    return hiD_reference, True


def score_panel(X, Z, *, config: PanelV2Config, x_ids=None, z_ids=None,
                centroids_by_k=None, anchor_masks=None, projection=None,
                hiD_reference=None, reference_identity=None, provenance):
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
    _peak_byte_preflight(cfg, n_dims=int(np.asarray(Xa[:1]).shape[1]), k_frac=kf)
    # v2.2: SEPARATE passes —
    #   hi_hit  : EXACT-reranked top-k_hit (FFR/recall high-D TRUTH; small k → cheap).
    #   hi_frac : APPROXIMATE fast-expansion top-k_frac MEMBERSHIP (purity; no rerank,
    #             so no 24 GB gather). Order inside the set is irrelevant to a label
    #             count; boundary membership is approximate and labelled so.
    #   lo_kf   : exact low-D top-k_frac (single deterministic pass).
    # S2.5: hi_hit/hi_frac/labels/density-radii are MAP-INDEPENDENT — computed once
    # in a content-verified reference (identical to the inline path by construction),
    # so rescoring N maps of one corpus repeats only the low-D passes below.
    ref, ref_reused = _resolve_reference(Xa, aidx, cfg, centroids_by_k, hiD_reference,
                                         reference_identity=reference_identity)
    hi_hit, hi_frac, guard_hit = ref["hi_hit"], ref["hi_frac"], ref["guard_hit"]
    lo_kf, _, _ = _self_knn(Z, aidx, kf, cfg, hi_dim=False)

    res = {"schema": PANEL_SCHEMA, "formula_version": cfg.formula_version,
           "n": int(n), "n_dims_hi": int(np.asarray(Xa[:1]).shape[1]),
           "n_dims_lo": int(Z.shape[1]), "frac": cfg.frac, "k_hit": cfg.k_hit,
           "k_frac": kf, "k_density": cfg.k_density,
           "anchor_seed": cfg.anchor_seed, "n_anchors": m,
           "anchor_hash": _ids_hash(aidx)}

    # ffr + recall@k use the EXACT high-D top-k_hit truth (separate masks — P0.4)
    res["ffr"] = round(ffr_from_neighbors(hi_hit[sel_ffr], lo_kf[sel_ffr], cfg.k_hit), 4)
    res["recall@k"] = round(recall_at_k_from_neighbors(hi_hit[sel_ffr], lo_kf[sel_ffr], cfg.k_hit), 5)
    res["n_ffr_anchors"] = int(len(sel_ffr))

    # purity per centroid granularity — uses the APPROXIMATE high-D k_frac membership
    if centroids_by_k:
        res["purity"] = {}
        res["purity_numerators"] = {}     # R2: high-D vs map label agreement, not just the ratio
        res["purity_exactness"] = "hi_frac_membership: approximate (fast expansion, no rerank)"
        res["centroid_hashes"] = {}
        lab = ref["labels"]                             # {k: labels[N]} (cached)
        for kc, labels in lab.items():
            alab = labels[aidx][sel_pur]
            hd = float((labels[hi_frac[sel_pur]] == alab[:, None]).mean())   # high-D agreement
            mp = float((labels[lo_kf[sel_pur]] == alab[:, None]).mean())     # map agreement
            res["purity"][f"k{kc}"] = round(mp / hd, 4) if hd else None
            # R2: ratio > 1 (map > high-D) means the map OVER-separates centroid
            # labels vs the source; report both numerators so the ratio can be read
            # as fidelity vs over-separation, not treated as pure fidelity.
            res["purity_numerators"][f"k{kc}"] = {"hi_D_agreement": round(hd, 4),
                                                  "map_agreement": round(mp, 4)}
            res["centroid_hashes"][f"k{kc}"] = _ids_hash(
                np.round(np.asarray(centroids_by_k[kc], np.float32), 4))
        res["n_purity_anchors"] = int(len(sel_pur))

    # density: exact-radius pass (k_density), hiD vs loD. hi-D radii are cached
    # for ALL anchors (per-anchor independent); index the density mask here.
    guard_den = ref["guard_den"]
    _, ld_r, _ = _self_knn(Z, aidx[sel_den], cfg.k_density, cfg, hi_dim=False, want_dist=True)
    r_hd = np.asarray(ref["r_hd"])[sel_den]; r_ld = ld_r.mean(1); eps = 1e-12
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
        # P2: honest per-path label. hi top-k_hit (FFR/recall) is exact-reranked
        # (validated ≥ tolerance vs fp64); hi k_frac membership (purity) is
        # approximate fast-expansion; density radii are exact; low-D is exact.
        "exactness": ("hi_k_hit:exact_rerank(byte_capped); hi_k_frac:approximate_membership; "
                      "density:exact; lo:single_pass_exact"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "peak_gpu_gb": (round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
                        if torch.cuda.is_available() else None),
        "peak_rss_gb": _peak_rss_gb(),
        "wall_s": round(time.time() - t0, 2),
        # S2.5: which high-D reference produced the truth, and whether it was a
        # shared cache reuse (one reference for N maps) or computed inline.
        "hiD_reference_key": ref["key"], "hiD_reference_reused": bool(ref_reused),
    }
    res["guards"] = {**_data_guards(Xa, Z), "hit_guard": guard_hit, "density_guard": guard_den}
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
