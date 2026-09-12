"""Card011 negative-pool mining (per frozen card011-prereg.md). Importable by the trainer for inline
refresh AND runnable as a preflight (mine step-0 collisions on the fixed15 start, check viability, audit
ANN-k60 vs exact, verify matched-random feasibility). Training rows ONLY. Encoder = DINO-1536 unit-normed
(substrate). Map = a given 2D coords array.

Design (reviewer-corrected): the COLLISION arm mines collisions from the CURRENT map and PERSISTS its
per-stage histogram H_step + pairs; the VERIFIED-RANDOM arm matches the COLLISION arm's EXACT H_step at
each stage (0/20K/40K) — not its own diverged map. So the collision arm must run BEFORE the random arm.

Frozen bins: source-pair = unordered pair over the 5 real sources (15 cats); rho = d_enc/max(d60_i,d60_j)
bins [1.25,1.5,1.75,2.0,2.5,3.0,inf) (6). COLLISION: j in i's 15 map-NN, verified non-neighbor (j not in
enc-k60(i) AND i not in enc-k60(j)), rho>1.25. VERIFIED-RANDOM: verified non-neighbor, rho>1.25, drawn w/o
map condition, subsampled to match a target H cell-by-cell.
"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np

SUBD = Path("/data/latent-basemap/substrates/card010-adaptive")
OC = Path("/data/latent-basemap/sandbox/overseer-codex")
STAGE_DIR = Path("/data/latent-basemap/sandbox/card011-train/stages")
SOURCES = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
RHO_EDGES = np.array([1.25, 1.5, 1.75, 2.0, 2.5, 3.0, np.inf], np.float64)  # 6 bins
NRHO = len(RHO_EDGES) - 1
MAP_K = 15; ENC_K60 = 60; RHO_MIN = 1.25; WORKERS = 6
SP_PAIRS = [(a, b) for a in range(5) for b in range(a, 5)]                    # 15 unordered
SP_INDEX = {p: i for i, p in enumerate(SP_PAIRS)}
NCELL = len(SP_PAIRS) * NRHO


def _device():
    import torch
    d = os.environ.get("CARD011_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    return d


def load_static():
    sub = np.asarray(np.load(SUBD / "substrate.f16.npy", mmap_mode="r"))       # (300K,1536) fp16, unit-normed
    knn_idx = np.load(SUBD / "knn_idx.npy")                                    # (300K,61) self col0
    knn_dist = np.load(SUBD / "knn_dist.npy")                                  # (300K,61) self=0 col0
    dsrc = np.load(SUBD / "draw_source.npy", allow_pickle=True).astype(str)
    src_code = np.array([SOURCES.index(s) for s in dsrc], np.int8)
    return sub, knn_idx, knn_dist, src_code


def _sp_cell(si, sj):
    a, b = (si, sj) if si <= sj else (sj, si)
    return SP_INDEX[(int(a), int(b))]


def _norm_subt(sub, device):
    """fp16 substrate -> fp32 UNIT-normalized, matching knn_dist's convention (norm was applied to
    fp32 features BEFORE fp16 storage; renormalize the half-rounded vectors so audit L2/cosine is exact,
    not raw half-rounded dot products)."""
    import torch
    t = torch.as_tensor(np.ascontiguousarray(sub), device=device).float()
    return t / torch.linalg.vector_norm(t, dim=1, keepdim=True).clamp_min(1e-12)


def mine_collisions(coords, sub, knn_idx, knn_dist, src_code, device=None, block=4000):
    """Return dict: src, dst (unordered i<j collision pairs), rho, cell (0..NCELL-1), hist(NCELL),
    per_source (i-endpoint source counts). Map 15-NN from coords; verified non-neighbor + rho>1.25."""
    import torch
    from scipy.spatial import cKDTree
    device = device or _device()
    n = coords.shape[0]
    tree = cKDTree(np.asarray(coords, np.float64))
    _, mnn = tree.query(np.asarray(coords, np.float64), k=MAP_K + 1, workers=WORKERS)  # (n,16) incl self
    kidx = torch.as_tensor(knn_idx[:, 1:ENC_K60 + 1].astype(np.int64), device=device)  # (n,60) enc-k60
    d60 = torch.as_tensor(knn_dist[:, ENC_K60].astype(np.float32), device=device)      # (n,)
    subt = _norm_subt(sub, device)                                                     # (n,1536) fp32 unit
    mnn_t = torch.as_tensor(mnn.astype(np.int64), device=device)
    out_i, out_j, out_rho = [], [], []
    for s in range(0, n, block):
        e = min(s + block, n); B = e - s
        rows = torch.arange(s, e, device=device)
        cand = mnn_t[s:e]                                                       # (B,16) sorted by 2D dist
        # drop self, keep 15 real map neighbors (vectorized, robust to 2D ties / self absent):
        # push self entries past the 15 kept slots while preserving distance order among non-self.
        selfmask = cand == rows[:, None]
        colrank = torch.arange(MAP_K + 1, device=device)[None, :].expand(B, MAP_K + 1)
        order = torch.argsort(selfmask.int() * (MAP_K + 5) + colrank, dim=1)
        jv = torch.gather(cand, 1, order)[:, :MAP_K]                            # (B,15) non-self, dist-ordered
        # verified non-neighbor: j not in enc-k60(i)
        k60_i = kidx[s:e]                                                       # (B,60)
        in_i = (jv[:, :, None] == k60_i[:, None, :]).any(-1)                    # (B,15)
        # i not in enc-k60(j)
        jflat = jv.reshape(-1)                                                  # (B*15,)
        k60_j = kidx.index_select(0, jflat)                                     # (B*15,60)
        irep = rows.repeat_interleave(MAP_K)                                    # (B*15,)
        in_j = (k60_j == irep[:, None]).any(-1).reshape(B, MAP_K)
        verified = (~in_i) & (~in_j)
        # exact encoder L2 (fp32) + rho
        xi = subt[s:e].float()                                                  # (B,1536)
        xj = subt.index_select(0, jflat).float().reshape(B, MAP_K, -1)          # (B,15,1536)
        denc = torch.linalg.vector_norm(xi[:, None, :] - xj, dim=-1)            # (B,15)
        d60max = torch.maximum(d60[s:e][:, None], d60.index_select(0, jflat).reshape(B, MAP_K))
        rho = denc / d60max.clamp_min(1e-12)
        keep = verified & (rho > RHO_MIN) & (jv >= 0)
        ii = rows[:, None].expand(B, MAP_K)[keep]
        jj = jv[keep]; rr = rho[keep]
        out_i.append(ii.cpu().numpy()); out_j.append(jj.cpu().numpy()); out_rho.append(rr.cpu().numpy())
    I = np.concatenate(out_i); J = np.concatenate(out_j); R = np.concatenate(out_rho)
    # dedupe unordered (i<j); RETAIN the query row i of the first retained instance so per-source
    # viability is counted by endpoint-i (the query row), per the frozen prereg, after canonicalization.
    lo = np.minimum(I, J); hi = np.maximum(I, J)
    key = lo.astype(np.int64) * n + hi.astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    src = lo[uniq]; dst = hi[uniq]; rho = R[uniq]; query_i = I[uniq]
    rbin = np.clip(np.digitize(rho, RHO_EDGES[1:-1], right=False), 0, NRHO - 1)
    sp = np.array([_sp_cell(src_code[a], src_code[b]) for a, b in zip(src, dst)], np.int64)
    cell = sp * NRHO + rbin
    hist = np.bincount(cell, minlength=NCELL)
    per_source = np.bincount(src_code[query_i].astype(np.int64), minlength=5)   # endpoint-i (query row), post-canonicalization
    return {"src": src, "dst": dst, "rho": rho, "cell": cell, "hist": hist,
            "per_source": per_source, "query_i": query_i}


def sample_matched_random(target_hist, sub, knn_idx, knn_dist, src_code, rng, device=None,
                          max_rounds=60, m_per_round=200000, eval_chunk=50000):
    """Sample verified-non-neighbor random pairs (rho>1.25, NO map condition), WITHOUT REPLACEMENT
    (globally-unique unordered pairs), matched cell-by-cell to target_hist. Bounded draw + chunked
    feature gather (memory). Recomputes the ACTUAL histogram of the selected pairs and asserts it equals
    the target. Raises if a populated cell cannot be filled. Encoder truth = renormalized fp16->fp32."""
    import torch
    device = device or _device()
    n = sub.shape[0]
    kidx = torch.as_tensor(knn_idx[:, 1:ENC_K60 + 1].astype(np.int64), device=device)
    d60 = torch.as_tensor(knn_dist[:, ENC_K60].astype(np.float32), device=device)
    subt = _norm_subt(sub, device)                                             # fp32 unit
    need = target_hist.astype(np.int64).copy()
    have = {c: [] for c in np.where(need > 0)[0]}
    got = np.zeros(NCELL, np.int64)
    used = set()                                                               # global (lo,hi) uniqueness
    gen = torch.Generator(device=device); gen.manual_seed(int(rng.integers(1 << 30)))
    for _ in range(max_rounds):
        if (got >= need).all():
            break
        i0 = torch.randint(0, n, (m_per_round,), generator=gen, device=device)
        off = torch.randint(1, n, (m_per_round,), generator=gen, device=device)
        j0 = (i0 + off) % n
        for s in range(0, m_per_round, eval_chunk):                            # chunk feature gather (memory)
            e = min(s + eval_chunk, m_per_round)
            i, j = i0[s:e], j0[s:e]
            in_i = (kidx.index_select(0, i) == j[:, None]).any(-1)
            in_j = (kidx.index_select(0, j) == i[:, None]).any(-1)
            denc = torch.linalg.vector_norm(subt.index_select(0, i) - subt.index_select(0, j), dim=-1)
            d60max = torch.maximum(d60.index_select(0, i), d60.index_select(0, j)).clamp_min(1e-12)
            rho = denc / d60max
            ok = (~in_i) & (~in_j) & (rho > RHO_MIN)
            ii, jj, rr = i[ok].cpu().numpy(), j[ok].cpu().numpy(), rho[ok].cpu().numpy()
            lo = np.minimum(ii, jj); hi = np.maximum(ii, jj)
            rbin = np.clip(np.digitize(rr, RHO_EDGES[1:-1], right=False), 0, NRHO - 1)
            sp = np.array([_sp_cell(src_code[a], src_code[b]) for a, b in zip(lo, hi)], np.int64) if len(lo) else np.zeros(0, np.int64)
            cell = sp * NRHO + rbin
            for t in range(len(lo)):
                c = int(cell[t])
                if c not in have or got[c] >= need[c]:
                    continue
                pair = (int(lo[t]), int(hi[t]))
                if pair in used:                                               # without replacement
                    continue
                used.add(pair); have[c].append(pair); got[c] += 1
    if not (got >= need).all():
        deficit = {int(c): int(need[c] - got[c]) for c in np.where(got < need)[0]}
        raise RuntimeError(f"matched-random cannot fill cells without replacement (deficit): {deficit}")
    src = []; dst = []
    for c in np.where(need > 0)[0]:
        for (a, b) in have[c][: int(need[c])]:
            src.append(a); dst.append(b)
    src = np.array(src, np.int64); dst = np.array(dst, np.int64)
    # verify actual uniqueness + recomputed histogram == target
    keys = src.astype(np.int64) * n + dst.astype(np.int64)
    assert len(np.unique(keys)) == len(keys), "matched-random produced duplicate pairs"
    rbin2 = np.clip(np.digitize(  # recompute rho bins from the SELECTED pairs (exact, renormed)
        _pair_rho(src, dst, subt, d60), RHO_EDGES[1:-1], right=False), 0, NRHO - 1)
    sp2 = np.array([_sp_cell(src_code[a], src_code[b]) for a, b in zip(src, dst)], np.int64)
    actual = np.bincount(sp2 * NRHO + rbin2, minlength=NCELL)
    assert np.array_equal(actual, need), "recomputed matched-random histogram != target"
    return src, dst


def _pair_rho(src, dst, subt, d60):
    import torch
    i = torch.as_tensor(src, device=subt.device); j = torch.as_tensor(dst, device=subt.device)
    out = []
    for s in range(0, len(src), 100000):
        e = min(s + 100000, len(src)); a, b = i[s:e], j[s:e]
        denc = torch.linalg.vector_norm(subt.index_select(0, a) - subt.index_select(0, b), dim=-1)
        d60max = torch.maximum(d60.index_select(0, a), d60.index_select(0, b)).clamp_min(1e-12)
        out.append((denc / d60max).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def viability(coll, src_code):
    n_coll = int(len(coll["src"]))
    per_src = {SOURCES[k]: int(coll["per_source"][k]) for k in range(5)}
    ok = bool(n_coll >= 20000 and all(v >= 1000 for v in per_src.values()))
    return {"n_collisions": n_coll, "per_source_i": per_src, "min_per_source": int(min(per_src.values())),
            "ge_20k": n_coll >= 20000, "ge_1k_per_source": all(v >= 1000 for v in per_src.values()),
            "viable": ok}


def _exact_k60(rows, subt):
    """Exact top-60 non-self neighbors (by cosine=IP on unit vectors) for the given row ids. Identity is
    excluded BY INDEX (a distance/cos tie can otherwise place another image at rank 0)."""
    import torch
    out = []
    for s in range(0, len(rows), 256):
        e = min(s + 256, len(rows)); ids = rows[s:e]
        sims = subt.index_select(0, ids) @ subt.T                              # (b,n)
        sims[torch.arange(e - s, device=subt.device), ids] = -2.0             # mask self by index
        out.append(torch.topk(sims, ENC_K60, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def audit_ann_vs_exact(sub, knn_idx, device=None, panel=2000, seed=10011):
    """Exact-search panel: recall of ANN enc-k60 vs exact top-60 on a random row panel (identity excluded
    by index; renormalized encoder truth)."""
    import torch
    device = device or _device()
    n = sub.shape[0]
    rng = np.random.default_rng(seed); rows = np.sort(rng.choice(n, min(panel, n), replace=False))
    subt = _norm_subt(sub, device)
    exact = _exact_k60(torch.as_tensor(rows, device=device), subt)
    recs = [len(set(int(x) for x in knn_idx[rows[r], 1:ENC_K60 + 1]) & set(int(x) for x in exact[r])) / ENC_K60
            for r in range(len(rows))]
    return {"panel": int(len(rows)), "ann_k60_recall_vs_exact": round(float(np.mean(recs)), 4)}


def audit_selected_pairs(src, dst, sub, knn_dist, device=None, panel=2000, seed=10012):
    """DIRECT false-admission audit of the SELECTED mined collision pairs: on a random panel, recompute
    each endpoint's EXACT enc-k60 (identity excluded by index) and verify the pair is genuinely a
    verified non-neighbor (j not in exact-k60(i) AND i not in exact-k60(j)) with EXACT rho>1.25. Reports
    the fraction of selected pairs the ANN mining ADMITTED that exact search rejects (contamination)."""
    import torch
    device = device or _device()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(src), min(panel, len(src)), replace=False))
    si = src[idx]; di = dst[idx]
    subt = _norm_subt(sub, device)
    d60 = torch.as_tensor(knn_dist[:, ENC_K60].astype(np.float32), device=device)
    ends = np.unique(np.concatenate([si, di]))
    ek = _exact_k60(torch.as_tensor(ends, device=device), subt)               # exact-k60 per endpoint
    exact_set = {int(e): set(int(x) for x in ek[r]) for r, e in enumerate(ends)}
    rho = _pair_rho(si, di, subt, d60)
    false_nn = 0; false_rho = 0
    for t in range(len(si)):
        a, b = int(si[t]), int(di[t])
        if b in exact_set[a] or a in exact_set[b]:
            false_nn += 1
        if rho[t] <= RHO_MIN:
            false_rho += 1
    return {"panel": int(len(si)), "exact_false_neighbor_admission_frac": round(false_nn / len(si), 5),
            "exact_rho_le_1p25_frac": round(false_rho / len(si), 5),
            "selected_pair_rho_median": round(float(np.median(rho)), 4)}


def _persist_stage(step, coll):
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(STAGE_DIR / f"collision-stage{step}.npz", src=coll["src"], dst=coll["dst"],
             rho=coll["rho"], cell=coll["cell"], hist=coll["hist"], per_source=coll["per_source"])


def load_stage_hist(step):
    z = np.load(STAGE_DIR / f"collision-stage{step}.npz")
    return z["hist"]


def main():
    """Preflight: mine step-0 collisions on the fixed15 start map, check viability, audit ANN vs exact,
    verify matched-random feasibility for H0. Persist stage0 + card011-viability.json. Fail-closed."""
    import torch
    device = _device()
    coords = np.asarray(np.load("/data/latent-basemap/sandbox/card010-train/coords-fixed15.npy"), np.float64)
    sub, knn_idx, knn_dist, src_code = load_static()
    print(f"[card011 preflight] device={device} mining step-0 collisions on fixed15 map ({coords.shape[0]} rows)", flush=True)
    coll = mine_collisions(coords, sub, knn_idx, knn_dist, src_code, device=device)
    via = viability(coll, src_code)
    audit = audit_ann_vs_exact(sub, knn_idx, device=device)
    sel_audit = audit_selected_pairs(coll["src"], coll["dst"], sub, knn_dist, device=device)
    # matched-random feasibility for H0 (does not commit the pool; just proves fillable without replacement)
    rng = np.random.default_rng(11)
    feas = {"feasible": True}
    try:
        rsrc, rdst = sample_matched_random(coll["hist"], sub, knn_idx, knn_dist, src_code, rng, device=device)
        feas["random_pairs"] = int(len(rsrc))
    except (RuntimeError, AssertionError) as ex:
        feas = {"feasible": False, "reason": str(ex)}
    _persist_stage(0, coll)
    # frozen contamination bound: exact search must reject <=3% of ANN-mined collisions as actually-neighbors
    contam_ok = sel_audit["exact_false_neighbor_admission_frac"] <= 0.03
    result = {"schema": "card011-viability-2026-09-11", "stage": 0, "map": "card010 fixed15 start",
              "n_cells_populated": int((coll["hist"] > 0).sum()), "ncell": NCELL,
              "rho_edges": RHO_EDGES.tolist(), "sources": SOURCES,
              "viability": via, "ann_audit": audit, "selected_pair_exact_audit": sel_audit,
              "contamination_le_3pct": bool(contam_ok), "matched_random_feasibility": feas,
              "VIABLE": bool(via["viable"] and feas.get("feasible", False) and contam_ok)}
    (OC / "card011-viability.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("viability", "ann_audit", "selected_pair_exact_audit",
                      "contamination_le_3pct", "matched_random_feasibility", "VIABLE")}, indent=1), flush=True)
    return 0 if result["VIABLE"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
