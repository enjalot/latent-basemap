"""Card009 derivative-preservation bank builder (CPU, off-flock).
For rows of the existing DINO OUT replay bank (card006_out_bank), builds the directional-
derivative supervision the Sobolev hook consumes:
  - deriv_X    : the OUT-bank input (fp16, L2-preprocessed) — same rows/preprocessing as replay.
  - deriv_dir  : unit TANGENT direction v_i toward a real nearby OLD-content embedding drawn from
                 a TRAINING-ONLY neighborhood pool (the T0 draw), projected orthogonal to x_i
                 (valid tangent for L2-normalised inputs).
  - deriv_scale: local neighbor scale s_i = ||neighbor - x_i|| (HD).
  - deriv_teacher_jv : the FROZEN T0 teacher's cached directional derivative J_old(x_i) v_i,
                 via batched create_graph-free autograd.functional.jvp.
  - deriv_ids  : pool positions (provenance).
R0 = fixed DINO radius 33.6717. Directions/pool exclude the seal and the OUT row itself.
Env: BANK_N (rows, default all OUT), POOL_N (training-only pool sample, default 500000).
Usage: build_card009_deriv_bank.py
"""
import os, sys, json, hashlib, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch, faiss
from basemap.pumap.parametric_umap.core import ParametricUMAP
DEVICE = os.environ.get("DERIV_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
def _log(m): print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
POOL = Path("/data2/monet/pool-20m"); SEAL = Path("/data2/monet/eval-common-v2")
T0SUB = Path("/data/latent-basemap/substrates/dino-arrival-t0")
TEACHER = SB / "dino-arrival-t0/champion-bs16k/model.pt"
R0 = 33.6717
BANK_N = int(os.environ.get("BANK_N", "0"))          # 0 = all OUT rows
POOL_N = int(os.environ.get("POOL_N", "200000"))
SEED = 9009


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def main():
    faiss.omp_set_num_threads(4)
    rng = np.random.default_rng(SEED)
    out_bank = np.load(OC / "card006_out_bank.npz")
    Xb = _norm(np.asarray(out_bank["replay_X"], np.float32))     # OUT rows, normalized
    out_ids = np.asarray(out_bank["replay_ids"]); src = out_bank["source"].astype(str)
    if BANK_N and BANK_N < Xb.shape[0]:
        sel = rng.choice(Xb.shape[0], BANK_N, replace=False); Xb = Xb[sel]; out_ids = out_ids[sel]; src = src[sel]
    n = Xb.shape[0]

    # training-only neighborhood pool: real OLD rows from the T0 draw (in the graph), excluding
    # the seal and any OUT-bank row; sample POOL_N.
    t0_draw = np.load(T0SUB / "draw_idx.npy")
    seal = set(np.load(SEAL / "ref_idx.npy").tolist()) | set(np.load(SEAL / "val_idx.npy").tolist())
    bankset = set(out_ids.tolist())
    pool_pos = np.array([p for p in t0_draw if p not in seal and p not in bankset])
    if pool_pos.size > POOL_N:
        pool_pos = rng.choice(pool_pos, POOL_N, replace=False)
    Xmm = np.load(POOL / "dino1536.f16.npy", mmap_mode="r")
    order = np.argsort(pool_pos); pool_pos_sorted = pool_pos[order]
    _log(f"gathering {pool_pos_sorted.size} pool rows from memmap (sorted for I/O locality)")
    Ppool = _norm(np.asarray(Xmm[pool_pos_sorted], np.float32)); _log("pool gathered + normalized")

    # nearest OLD neighbor for each OUT row (cosine == IP on normalized). Exact flat search over
    # 200K x 200K x 1536 is infeasible on CPU; use an APPROXIMATE IVF index — the direction only needs
    # "a real nearby old neighbor", so an approximate top-1 is sufficient (recorded as such).
    faiss.omp_set_num_threads(4)
    d = Ppool.shape[1]; nlist = max(64, min(4096, int(np.sqrt(Ppool.shape[0]))))
    quant = faiss.IndexFlatIP(d); index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    _log(f"training IVF (nlist={nlist})"); index.train(np.ascontiguousarray(Ppool)); index.add(np.ascontiguousarray(Ppool))
    index.nprobe = 32
    _log(f"faiss IVF searching {n} OUT rows (nprobe=32, approximate top-1)")
    D, I = index.search(np.ascontiguousarray(Xb), 1)  # approximate top-1 old neighbor
    nbr = Ppool[I[:, 0]].copy()                                        # [n, D] neighbor embeddings
    nbr_ids = pool_pos_sorted[I[:, 0]].astype(np.int64)               # provenance: neighbor global pool positions
    del Ppool, index, Xmm; import gc; gc.collect(); _log("neighbors extracted; pool/index freed")

    # EXACT-INPUT teacher fidelity (review item 2): freeze the STORED fp16 input first and compute
    # everything (tangent direction, teacher Jv) at exactly the fp32-cast stored input the hook uses.
    Xb16 = Xb.astype(np.float16)
    x_used = Xb16.astype(np.float32)                                  # == deriv_X.to(fp32) in the hook
    raw = nbr - x_used                                                # displacement from the exact input
    scale = np.linalg.norm(raw, axis=1).astype(np.float32)           # local neighbor scale s_i
    # tangent projection on x_used with the CORRECT norm denominator (fp16 round-trip -> not exactly unit)
    xn2 = np.sum(x_used * x_used, axis=1, keepdims=True).clip(1e-12)
    dot = np.sum(raw * x_used, axis=1, keepdims=True)
    v = raw - (dot / xn2) * x_used
    vnorm = np.linalg.norm(v, axis=1, keepdims=True)
    good = (vnorm[:, 0] > 1e-6) & (scale > 1e-6)
    v = (v / vnorm.clip(1e-12)).astype(np.float32)

    # teacher directional derivative J_old(x_used) v via batched jvp at the EXACT stored input.
    teacher = ParametricUMAP.load(str(TEACHER), device=DEVICE); teacher.model.eval()
    for p in teacher.model.parameters(): p.requires_grad_(False)
    def _teacher_jv(Xarr, Varr, tag=""):
        outs = []; N = Xarr.shape[0]
        with torch.no_grad():
            for i in range(0, N, 20000):
                xt = torch.from_numpy(np.asarray(Xarr[i:i + 20000], np.float32)).to(DEVICE)
                vt = torch.from_numpy(np.asarray(Varr[i:i + 20000], np.float32)).to(DEVICE)
                _, jv = torch.autograd.functional.jvp(lambda inp: teacher.model(inp), xt, vt)
                outs.append(jv.float().cpu().numpy().astype(np.float32))
                if tag: _log(f"  teacher-jv {tag} {min(i + 20000, N)}/{N}")
        return np.concatenate(outs)
    _log(f"teacher Jv on {DEVICE} for {n} rows"); jv_teacher = _teacher_jv(x_used, v, tag="main")

    # keep only rows with a valid tangent direction (filter ALL arrays incl the stored fp16 input)
    Xb16, x_used, v, scale, jv_teacher, out_ids, src, nbr_ids = \
        Xb16[good], x_used[good], v[good], scale[good], jv_teacher[good], out_ids[good], src[good], nbr_ids[good]
    ng = int(good.sum())
    assert np.isfinite(v).all() and np.isfinite(jv_teacher).all() and np.isfinite(scale).all()
    ortho = float(np.abs(np.sum(v * x_used, axis=1)).max())           # tangent on the EXACT input: v . x_used ~ 0
    unit = float(np.abs(np.linalg.norm(v, axis=1) - 1).max())

    def _h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    np.savez(OC / "card009_deriv_bank.npz", deriv_X=Xb16, deriv_dir=v.astype(np.float32),
             deriv_scale=scale.astype(np.float32), deriv_teacher_jv=jv_teacher.astype(np.float32),
             deriv_ids=out_ids.astype(np.int64), deriv_neighbor_ids=nbr_ids, source=src)

    # EXACT-INPUT FIDELITY CHECK: reload the stored fp16, cast to fp32 (what the hook feeds), recompute
    # teacher Jv with the SAME directions -> must be bit-identical to the stored teacher Jv (near-floor).
    z = np.load(OC / "card009_deriv_bank.npz")
    x_reload = np.asarray(z["deriv_X"], np.float32)                   # fp16 -> fp32 exactly as the hook does
    smp = np.sort(np.random.default_rng(SEED).choice(x_reload.shape[0], min(5000, x_reload.shape[0]), replace=False))
    jv_reload = _teacher_jv(x_reload[smp], np.asarray(z["deriv_dir"], np.float32)[smp])  # sample proves the property
    fidelity_resid = float(np.abs(jv_reload - jv_teacher[smp]).max())

    manifest = {"schema": "card009-deriv-bank-2026-09-11", "R0": R0, "n_rows": ng, "n_dropped_no_tangent": int(n - ng),
                "pool_n": int(pool_pos_sorted.size), "teacher": str(TEACHER), "seed": SEED,
                "max_abs_v_dot_x": round(ortho, 8), "max_abs_unit_dev": round(unit, 8),
                "exact_input_teacher_vs_reload_jv_resid": fidelity_resid,
                "scale_mean": round(float(scale.mean()), 5), "jv_teacher_norm_mean": round(float(np.linalg.norm(jv_teacher, axis=1).mean()), 5),
                "per_source": {s: int((src == s).sum()) for s in sorted(set(src.tolist()))},
                "hashes": {"deriv_X": _h(Xb16), "deriv_dir": _h(v.astype(np.float32)), "deriv_scale": _h(scale.astype(np.float32)),
                           "deriv_teacher_jv": _h(jv_teacher.astype(np.float32)), "deriv_ids": _h(np.sort(out_ids.astype(np.int64))),
                           "deriv_neighbor_ids": _h(np.sort(nbr_ids))},
                "note": "exact-input teacher Jv at fp16-cast stored input; tangent on x_used w/ correct norm; "
                        "training-only T0-pool APPROXIMATE-IVF top-1 neighbors (global ids persisted); frozen T0 teacher."}
    (OC / "card009-deriv-bank-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    ok = (np.isfinite(jv_teacher).all() and ortho < 1e-4 and unit < 1e-4 and ng > 0
          and fidelity_resid < 1e-5)   # exact-input teacher fidelity at the numerical floor
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
