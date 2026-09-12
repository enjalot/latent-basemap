"""Card012 replay-bank selection from the fixed 1M candidate pool (per card012-prereg.md). At each refresh
(0/35K/70K/105K) both arms score the SAME pool through the CURRENT student; the bank differs by rule:
  uniform:        200K uniform within fixed per-source quotas (40K/source).
  error_directed: 100K uniform + 100K from the highest-error 20% WITHIN each source, WITHOUT replacement,
                  EXCLUDING the uniform half. Error = NATIVE squared student-coord minus ORIGINAL-T0-teacher
                  displacement (no rigid align on candidate rows).
Bank format matches the OUT bank + core's replay contract: replay_X (fp16, EXACT stored pool X, no renorm),
replay_targets (fp32 = T0 teacher), replay_ids (int64), source. Writes bank .npz + a content SHA matching
core's _replay_bank_sha = sha256(sorted ids int64 + X + targets float32). Importable by the trainer refresh.
"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()

POOLD = Path("/data/latent-basemap/sandbox/card012-pool")
SOURCES = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
BANK_N = 200000; PER_SRC = BANK_N // 5                      # 40K/source
UNIF_PER_SRC = 20000; ERR_PER_SRC = 20000                  # error arm: 20K+20K per source


def load_pool():
    ids = np.load(POOLD / "pool_ids.npy"); src = np.load(POOLD / "pool_source.npy", allow_pickle=True).astype(str)
    X = np.load(POOLD / "pool_X.f16.npy", mmap_mode="r"); tea = np.load(POOLD / "pool_teacher.npy")
    return ids, src, X, tea


def _content_sha(ids, X, tea):
    return hashlib.sha256(np.ascontiguousarray(np.sort(ids.astype(np.int64))).tobytes()
                          + np.ascontiguousarray(np.asarray(X, np.float16)).tobytes()
                          + np.ascontiguousarray(np.asarray(tea, np.float32)).tobytes()).hexdigest()[:16]


def student_error(model, X, tea, device, batch=100000):
    """NATIVE squared displacement ||student(stored fp16 X -> fp32, no renorm) - T0 teacher||^2 per row."""
    import torch
    model.eval(); err = np.empty(X.shape[0], np.float64)
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            e = min(s + batch, X.shape[0])
            xb = torch.from_numpy(np.asarray(X[s:e], np.float32)).to(device)   # NO renorm (exact stored input)
            c = model(xb).float().cpu().numpy()
            err[s:e] = ((c - tea[s:e]) ** 2).sum(1)
    return err


def select(arm, model, device, out_path, seed):
    ids, src, X, tea = load_pool()
    rng = np.random.default_rng(seed)
    err = student_error(model, X, tea, device)
    chosen = []
    for s in SOURCES:
        pos = np.where(src == s)[0]
        if arm == "uniform":
            chosen.append(rng.choice(pos, PER_SRC, replace=False))
        elif arm == "error_directed":
            unif = rng.choice(pos, UNIF_PER_SRC, replace=False)
            unif_set = set(unif.tolist())
            # highest-error 20% within this source (top quintile by native squared displacement)
            top_n = max(ERR_PER_SRC, int(round(0.20 * pos.size)))
            order = pos[np.argsort(-err[pos])][:top_n]                        # top-20% error rows
            elig = np.array([p for p in order if p not in unif_set], np.int64)  # exclude the uniform half
            assert elig.size >= ERR_PER_SRC, f"{s}: error stratum {elig.size} < {ERR_PER_SRC} (STOP)"
            errpick = rng.choice(elig, ERR_PER_SRC, replace=False)
            chosen.append(np.concatenate([unif, errpick]))
        else:
            raise ValueError(arm)
    sel = np.sort(np.concatenate(chosen))
    assert len(np.unique(sel)) == len(sel) == BANK_N, f"{arm}: bank size {len(sel)} / unique {len(np.unique(sel))}"
    assert np.isfinite(err).all(), "student error scores are non-finite"        # finite scores before ranking
    bX = np.asarray(X[sel], np.float16); bt = np.asarray(tea[sel], np.float32); bid = ids[sel].astype(np.int64)
    bsrc = src[sel]
    np.savez(out_path, replay_X=bX, replay_targets=bt, replay_ids=bid, source=bsrc)
    sha = _content_sha(bid, bX, bt)                                             # SORTED-id content sha (core contract)
    # ORDERED row->target identity digest: detects per-row misalignment / reorder that a sorted-id hash cannot.
    ordered_sha = hashlib.sha256(np.ascontiguousarray(bid).tobytes()
                                 + np.ascontiguousarray(bX).tobytes()
                                 + np.ascontiguousarray(bt).tobytes()).hexdigest()[:16]
    meta = {"arm": arm, "bank_n": int(len(sel)), "content_sha": sha, "ordered_row_identity_sha": ordered_sha,
            "per_source": {s: int((bsrc == s).sum()) for s in SOURCES},
            "err_mean": round(float(err.mean()), 6), "err_p99": round(float(np.percentile(err, 99)), 6),
            "selected_err_mean": round(float(err[sel].mean()), 6)}
    return meta


if __name__ == "__main__":
    # CLI self-check: select both arms from the T0 head (step-0 state) to verify strata feasibility.
    import torch
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ParametricUMAP.load(str(Path("/data/latent-basemap/sandbox/dino-arrival-t0/champion-bs16k/model.pt")), device=dev).model
    OUT = Path("/data/latent-basemap/sandbox/card012-pool/banks"); OUT.mkdir(exist_ok=True)
    for arm in ("uniform", "error_directed"):
        meta = select(arm, m, dev, OUT / f"selfcheck-{arm}-step0.npz", seed=12012)
        print(json.dumps(meta), flush=True)
