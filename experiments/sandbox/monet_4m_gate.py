"""4M-vs-2M regression gate (owner image phase, item a; 2026-09-05). GPU (project) + CPU (faiss/KDTree).

Compares the 4M champion against the 2M champion on a COMMON held-out population (the 4M draw's test set,
200K rows OUTSIDE the 4M union -> unseen by BOTH heads, since random-2m subset 4M). Two metrics per head, same
convention as quick_ffr_at_0.1pct / heldout_reception:
  - held-out FFR (internal): test rows' high-D 15NN among TEST itself vs their 0.1%-of-test 2D disc after
    projection through the head. Structure preservation on unseen rows.
  - reception: test rows' high-D 15NN among the head's TRAINING set vs their 0.1%-of-training 2D disc in the
    head's existing map (coordinates.npy). New-to-existing-map reception.
Gate (frozen before scoring): 4M must be within 0.02 ABS of 2M on BOTH (i.e. 4M >= 2M - 0.02). Retain the 2M
fallback if 4M regresses. Verdict written to the 4M champion dir; separate receipts (never overwrites 2M).

Heads: 2M = monet-random-clip-2m/champion-bs16k, 4M = monet-random-clip-4m/champion-bs16k. Test high-D =
random-clip-4m/test-clip.f32.npy (already gathered, unit-norm). Usage: monet_4m_gate.py [NQ=20000] [SEED=0].
"""
import json, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
RM = Path("/data2/monet/random-2m")
FOURM = Path("/data2/monet/random-clip-4m")
K = 15
GATE_ABS = 0.02
HEADS = {
    "2m": {"model": SB / "monet-random-clip-2m/champion-bs16k/model.pt",
           "coords": SB / "monet-random-clip-2m/champion-bs16k/coordinates.npy",
           "train_hd": RM / "clip-substrate.f32.npy"},
    "4m": {"model": SB / "monet-random-clip-4m/champion-bs16k/model.pt",
           "coords": SB / "monet-random-clip-4m/champion-bs16k/coordinates.npy",
           "train_hd": FOURM / "clip-substrate.f32.npy"},
}


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _recall(q_hd, q_2d, ref_hd, ref_2d, index, tree, disc):
    """recall@K: fraction of a query's high-D K-NN among ref that fall in its 0.1%-of-ref 2D disc."""
    _, hd = index.search(q_hd, K)
    _, d2 = tree.query(q_2d, k=disc, workers=-1)
    rec = np.empty(q_hd.shape[0])
    for i in range(q_hd.shape[0]):
        ds = set(int(x) for x in d2[i])
        rec[i] = sum(int(x) in ds for x in hd[i][:K]) / K
    return float(rec.mean())


def _project(model_path, X):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    pumap = ParametricUMAP.load(str(model_path), device="cuda")
    return np.asarray(pumap.transform(X, batch_size=16384), dtype=np.float32)


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    import faiss
    from scipy.spatial import cKDTree

    for h in HEADS.values():
        for k in ("model", "coords", "train_hd"):
            if not h[k].exists():
                raise SystemExit(f"missing {h[k]} — is the 4M champion done?")

    test_hd = _norm(np.asarray(np.load(FOURM / "test-clip.f32.npy"), np.float32))
    rng = np.random.default_rng(seed)
    qsel = np.sort(rng.choice(test_hd.shape[0], min(nq, test_hd.shape[0]), replace=False))
    tq = test_hd[qsel]

    # internal (held-out FFR): test high-D 15NN among test, vs test 2D disc after projection
    ti = faiss.IndexFlatIP(512); ti.add(test_hd)
    disc_int = max(int(round(test_hd.shape[0] * 0.001)), K)

    out = {"schema": "monet-4m-gate-2026-09-05", "k": K, "nq": int(qsel.size), "gate_abs": GATE_ABS,
           "test_population": "random-clip-4m test set (outside the 4M union; unseen by both heads)", "heads": {}}
    for name, h in HEADS.items():
        t0 = time.time()
        test_2d = _project(h["model"], test_hd)             # project ALL test (for internal FFR neighborhoods)
        tree_int = cKDTree(np.asarray(test_2d, np.float64))
        ffr = _recall(tq, np.asarray(test_2d[qsel], np.float64), test_hd, test_2d, ti, tree_int, disc_int)

        ref_hd = _norm(np.asarray(np.load(h["train_hd"], mmap_mode="r"), np.float32))
        ref_2d = np.asarray(np.load(h["coords"]), np.float64)
        idx = faiss.IndexFlatIP(512); idx.add(ref_hd)
        tree_ref = cKDTree(ref_2d)
        disc_ref = max(int(round(ref_hd.shape[0] * 0.001)), K)
        recep = _recall(tq, np.asarray(test_2d[qsel], np.float64), ref_hd, ref_2d, idx, tree_ref, disc_ref)
        out["heads"][name] = {"heldout_ffr": round(ffr, 4), "reception": round(recep, 4),
                              "n_train": int(ref_hd.shape[0]), "wall_s": round(time.time() - t0, 1)}
        print(f"[gate] {name}: held-out FFR {ffr:.4f}  reception {recep:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        del ref_hd, idx, tree_ref

    f2, f4 = out["heads"]["2m"]["heldout_ffr"], out["heads"]["4m"]["heldout_ffr"]
    r2, r4 = out["heads"]["2m"]["reception"], out["heads"]["4m"]["reception"]
    ffr_ok = f4 >= f2 - GATE_ABS
    rec_ok = r4 >= r2 - GATE_ABS
    out["deltas"] = {"ffr_4m_minus_2m": round(f4 - f2, 4), "reception_4m_minus_2m": round(r4 - r2, 4)}
    out["gate"] = {"ffr_within_0.02": bool(ffr_ok), "reception_within_0.02": bool(rec_ok),
                   "PASS": bool(ffr_ok and rec_ok),
                   "verdict": ("4M joins the projection set" if (ffr_ok and rec_ok)
                               else "4M regresses >0.02 — retain 2M fallback for the demo")}
    (SB / "monet-random-clip-4m" / "gate-vs-2m.json").write_text(json.dumps(out, indent=1))
    print(f"[gate] 2M ffr {f2} recep {r2} | 4M ffr {f4} recep {r4} | "
          f"delta ffr {f4-f2:+.4f} recep {r4-r2:+.4f} | PASS={ffr_ok and rec_ok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
