"""Card009 deriv-bank provenance + exact-neighbor audit (CPU, off-flock) — binds the FRESH
residual-0.0 bank for admission (does NOT recompute the 200K JVP). Produces
card009-deriv-bank-provenance.json with:
  - integrity: recompute deriv_X/dir/scale/teacher_jv/ids/neighbor_ids content hashes, assert they
    match the build manifest (freshness/identity bound to the on-disk bank).
  - teacher checkpoint sha256 + IVF params (nlist/nprobe/pool seed) + pool-ID hash.
  - exact-input fidelity re-check on a sample (reload fp16 -> CPU teacher JVP == stored, residual).
  - exact-neighbor audit: reconstruct the SAME training-only pool (seed 9009), exact top-1 search for
    a bounded sample of deriv rows, and report IVF-vs-exact recall@1 + mean cosine gap — the directions
    are DECLARED approximate IVF top-1; this quantifies their actual closeness (not a new gate).
Usage: card009_deriv_bank_provenance.py
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch, faiss
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
POOL = Path("/data2/monet/pool-20m"); SEAL = Path("/data2/monet/eval-common-v2")
T0SUB = Path("/data/latent-basemap/substrates/dino-arrival-t0")
TEACHER = SB / "dino-arrival-t0/champion-bs16k/model.pt"
POOL_N, SEED, AUDIT_K = 200000, 9009, 3000


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def _h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
def _fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()[:16]


def main():
    z = np.load(OC / "card009_deriv_bank.npz"); man = json.load(open(OC / "card009-deriv-bank-manifest.json"))
    dX = np.asarray(z["deriv_X"]); dv = np.asarray(z["deriv_dir"], np.float32); ds = np.asarray(z["deriv_scale"], np.float32)
    djv = np.asarray(z["deriv_teacher_jv"], np.float32); dids = np.asarray(z["deriv_ids"]); dnbr = np.asarray(z["deriv_neighbor_ids"])
    # integrity: content hashes must match the build manifest
    integ = {"deriv_X": _h(dX) == man["hashes"]["deriv_X"], "deriv_dir": _h(dv) == man["hashes"]["deriv_dir"],
             "deriv_scale": _h(ds) == man["hashes"]["deriv_scale"], "deriv_teacher_jv": _h(djv) == man["hashes"]["deriv_teacher_jv"],
             "deriv_ids": _h(np.sort(dids.astype(np.int64))) == man["hashes"]["deriv_ids"],
             "deriv_neighbor_ids": _h(np.sort(dnbr)) == man["hashes"]["deriv_neighbor_ids"]}
    integrity_ok = all(integ.values())

    # exact-input fidelity re-check on a sample (CPU deterministic)
    teacher = ParametricUMAP.load(str(TEACHER), device="cpu"); teacher.model.eval()
    for p in teacher.model.parameters(): p.requires_grad_(False)
    smp = np.sort(np.random.default_rng(1).choice(dX.shape[0], min(4000, dX.shape[0]), replace=False))
    xs = np.asarray(dX[smp], np.float32); vs = dv[smp]
    with torch.no_grad():
        _, jvr = torch.autograd.functional.jvp(lambda i: teacher.model(i), torch.from_numpy(xs), torch.from_numpy(vs))
    fidelity_resid = float(np.abs(jvr.float().numpy().astype(np.float32) - djv[smp]).max())

    # reconstruct the SAME training-only pool (seed 9009) for the exact-neighbor audit
    t0_draw = np.load(T0SUB / "draw_idx.npy")
    seal = set(np.load(SEAL / "ref_idx.npy").tolist()) | set(np.load(SEAL / "val_idx.npy").tolist())
    bankset = set(dids.tolist())
    pool_pos = np.array([p for p in t0_draw if p not in seal and p not in bankset])
    rng = np.random.default_rng(SEED)
    if pool_pos.size > POOL_N: pool_pos = rng.choice(pool_pos, POOL_N, replace=False)
    pool_pos_sorted = pool_pos[np.argsort(pool_pos)]
    pool_id_hash = _h(np.sort(pool_pos_sorted))
    Xmm = np.load(POOL / "dino1536.f16.npy", mmap_mode="r")
    Ppool = _norm(np.asarray(Xmm[pool_pos_sorted], np.float32))
    # audit sample: exact top-1 in the pool vs the recorded IVF neighbor
    faiss.omp_set_num_threads(4)
    aidx = np.sort(np.random.default_rng(2).choice(dX.shape[0], min(AUDIT_K, dX.shape[0]), replace=False))
    q = _norm(np.asarray(dX[aidx], np.float32))
    flat = faiss.IndexFlatIP(Ppool.shape[1]); flat.add(np.ascontiguousarray(Ppool))
    Dex, Iex = flat.search(np.ascontiguousarray(q), 1)                 # exact top-1 (bounded sample)
    exact_nbr_global = pool_pos_sorted[Iex[:, 0]]
    recall_at_1 = float((exact_nbr_global == dnbr[aidx]).mean())       # IVF top-1 == exact top-1?
    # cosine of recorded IVF neighbor vs exact neighbor (closeness of the approximation)
    g2l = {int(g): i for i, g in enumerate(pool_pos_sorted)}
    ivf_local = np.array([g2l.get(int(g), -1) for g in dnbr[aidx]])
    valid = ivf_local >= 0
    cos_ivf = np.sum(q[valid] * Ppool[ivf_local[valid]], axis=1)
    cos_exact = Dex[:, 0][valid]
    cos_gap = float(np.mean(cos_exact - cos_ivf))

    out = {"schema": "card009-deriv-bank-provenance-2026-09-11", "n_rows": int(dX.shape[0]),
           "bank_integrity_ok": bool(integrity_ok), "integrity_detail": {k: bool(v) for k, v in integ.items()},
           "exact_input_fidelity_resid_sample": fidelity_resid, "fidelity_ok": bool(fidelity_resid < 1e-4),
           "teacher_sha16": _fsha(TEACHER), "teacher_path": str(TEACHER),
           "ivf": {"nlist": max(64, min(4096, int(np.sqrt(POOL_N)))), "nprobe": 32, "pool_seed": SEED,
                   "pool_n": int(pool_pos_sorted.size), "pool_id_hash": pool_id_hash,
                   "directions": "APPROXIMATE IVF top-1 (declared) — NOT exact nearest neighbor"},
           "exact_neighbor_audit": {"sample": int(q.shape[0]), "recall_at_1_vs_exact": round(recall_at_1, 4),
                                    "mean_cosine_gap_exact_minus_ivf": round(cos_gap, 6),
                                    "note": "recall<1 expected for approximate IVF; small cosine gap = directions still point at genuinely near old content."},
           "bank_manifest_fidelity_resid": man["exact_input_teacher_vs_reload_jv_resid"],
           "per_source": man["per_source"],
           "note": "binds the fresh residual-0.0 bank (content hashes verified); IVF directions declared approximate with measured closeness; not a new gate."}
    (OC / "card009-deriv-bank-provenance.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("bank_integrity_ok", "exact_input_fidelity_resid_sample", "teacher_sha16", "exact_neighbor_audit")}, indent=1))
    return 0 if (integrity_ok and fidelity_resid < 1e-4) else 3


if __name__ == "__main__":
    raise SystemExit(main())
