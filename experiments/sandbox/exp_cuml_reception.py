"""cuML transform experiment — phase 2 reception (.venv: faiss-CPU + cKDTree, no GPU). Scores each cohort's
cuML-transform() 2D coords with the SAME reception instrument as our heads' three-way: reception recall@15 =
fraction of a query's high-D 15NN (among the 4M cuML REFERENCE map) that are also its 2D neighbors within the
0.1%-disc (4000) of the reference 2D layout. Directly comparable to the frozen-head reception columns.

Reads /data/latent-basemap/sandbox/cuml-transform-20260906/{ref-coords-4m.npy, <cohort>-{coords,idx}.npy} +
merges into cuml-transform-reception.json. Usage: exp_cuml_reception.py [NQ=6000]
"""
import json, sys, time
from pathlib import Path
import numpy as np

OUT = Path("/data/latent-basemap/sandbox/cuml-transform-20260906")
SUB4M = "/data2/monet/random-clip-4m/clip-substrate.f32.npy"
SRC = {"testhd": "/data2/monet/random-clip-4m/test-clip.f32.npy",
       "complement": "/data2/monet/pool-complement-88m/clip512.f32.npy",
       "bl": "/data2/monet/bl-clip/bl-clip.f32.npy",
       "member": SUB4M}
K = 15


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    import faiss
    from scipy.spatial import cKDTree
    ref_hd = _norm(np.asarray(np.load(SUB4M, mmap_mode="r"), np.float32)); n_ref = ref_hd.shape[0]
    ref_2d = np.asarray(np.load(OUT / "ref-coords-4m.npy"), np.float64)
    t0 = time.time(); index = faiss.IndexFlatIP(ref_hd.shape[1]); index.add(ref_hd); tree = cKDTree(ref_2d)
    disc = max(int(round(n_ref * 0.001)), K)
    print(f"[cuml-recep] ref {n_ref:,}, index+tree {time.time()-t0:.0f}s, disc={disc}", flush=True)
    man = json.loads((OUT / "phase1-manifest.json").read_text()) if (OUT / "phase1-manifest.json").exists() else {}
    recep = {}
    for name, src in SRC.items():
        cf = OUT / f"{name}-coords.npy"
        if not cf.exists():
            recep[name] = {"skipped": "no coords (deferred/failed in phase1)"}; continue
        idx = np.load(OUT / f"{name}-idx.npy"); c2d = np.asarray(np.load(cf), np.float64)
        self_in_ref = (name == "member")
        rng = np.random.default_rng(0); sel = rng.choice(idx.size, min(nq, idx.size), replace=False)
        qs = np.sort(idx[sel]); q_hd = _norm(np.asarray(np.load(src, mmap_mode="r")[qs], np.float32))
        # map sampled source-positions back to their row in the saved coords (idx is sorted-unique in phase1)
        pos = np.searchsorted(idx, qs); q_2d = c2d[pos]
        _, hd = index.search(q_hd, K + (1 if self_in_ref else 0))
        _, d2 = tree.query(q_2d, k=disc + (1 if self_in_ref else 0), workers=-1)
        rec = np.empty(qs.size)
        for i in range(qs.size):
            h = hd[i]; t = d2[i]
            if self_in_ref: h = h[1:]; t = t[1:]
            h = h[:K]; ds = set(int(x) for x in t)
            rec[i] = sum(int(x) in ds for x in h) / K
        recep[name] = {"reception_recall@15": round(float(rec.mean()), 4), "nq": int(qs.size)}
        print(f"[cuml-recep] {name}: recall@15 {rec.mean():.4f} (n={qs.size})", flush=True)
    man["reception"] = recep; man["reception_note"] = (
        "recall@15 into the 4M cuML reference map (high-D 15NN among ref vs 2D 4000-disc); same instrument as the "
        "frozen-head three-way. member = training-row self-reception (self dropped). testhd = in-distribution held-out; "
        "complement + bl = OOD. Honest cuML projection column: fit speed + transform throughput/determinism/member-repro "
        "(phase1) + these reception numbers. Compare bl to the frozen-head BL reception.")
    (OUT / "cuml-transform-reception.json").write_text(json.dumps(man, indent=1))
    print(f"[cuml-recep] DONE -> {OUT/'cuml-transform-reception.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
