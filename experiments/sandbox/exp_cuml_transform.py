"""cuML UMAP transform() experiment — phase 1 (owner ask via claims-review 2026-09-06). cuml-env (cuml_py).
Measures what we WRONGLY declared impossible (transform_supported=false was hardcoded, never tested). Measurement,
not advocacy — prereg nothing directional.

1. Fit cuML UMAP on the SAME 4M CLIP-512 rows / same as-run config / seed 42 (~7min). PERSIST the fitted model
   (pickle) + save the 4M reference coords.
2. transform() cohorts -> save each cohort's 2D coords for the .venv reception scorer (phase 2):
     canary : 100K of the 4M TRAINING rows (fail-fast throughput + VRAM probe FIRST)
     member : 100K training rows -> displacement vs their fitted coords = cuML's member-reproduction error
     testhd : 100K held-out (random-clip-4m/test-clip, outside the 4M draw) = in-distribution reception
     complement : 100K complement clip512 (OOD, local)
     bl : the 1.08M BL thumb-CLIP (OOD — directly comparable to our frozen-head BL)
   Records throughput (rows/s), VRAM (free/total via cupy), transform determinism (same 10K twice -> max coord delta).

Writes /data/latent-basemap/sandbox/cuml-transform-20260906/{model.pkl, ref-coords-4m.npy, <cohort>-coords.npy,
phase1-manifest.json}. Usage: cuml_py exp_cuml_transform.py
"""
import json, sys, time, pickle
from pathlib import Path
import numpy as np

SUB4M = "/data2/monet/random-clip-4m/clip-substrate.f32.npy"
TESTHD = "/data2/monet/random-clip-4m/test-clip.f32.npy"
COMPL = "/data2/monet/pool-complement-88m/clip512.f32.npy"
BL = "/data2/monet/bl-clip/bl-clip.f32.npy"
OUT = Path("/data/latent-basemap/sandbox/cuml-transform-20260906"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    from cuml.manifold import UMAP
    import cuml, cupy as cp

    def vram():
        free, total = cp.cuda.Device().mem_info
        return {"used_gb": round((total - free) / 1e9, 2), "total_gb": round(total / 1e9, 2)}

    X = _norm(np.asarray(np.load(SUB4M, mmap_mode="r"), np.float32)); n = X.shape[0]
    kw = dict(n_neighbors=15, min_dist=0.0, n_components=2, metric="cosine",
              build_algo="auto", random_state=SEED, verbose=True)
    print(f"[cuml-tf] fit {n:,}x{X.shape[1]} cuml {cuml.__version__} kw={kw}", flush=True)
    t0 = time.time(); reducer = UMAP(**kw)
    ref = np.asarray(reducer.fit_transform(X), dtype=np.float32); fit_wall = time.time() - t0
    np.save(OUT / "ref-coords-4m.npy", ref)
    print(f"[cuml-tf] fit {fit_wall:.0f}s ({fit_wall/60:.1f}min) build_algo={getattr(reducer,'build_algo','?')} vram={vram()}", flush=True)
    try:
        with open(OUT / "model.pkl", "wb") as f: pickle.dump(reducer, f)
        print("[cuml-tf] model persisted -> model.pkl", flush=True)
    except Exception as e:
        print(f"[cuml-tf] WARN model pickle failed ({e}) — continuing with live reducer", flush=True)

    def transform(x):
        out = reducer.transform(cp.asarray(x))           # cuML returns a cupy array — .get() to numpy (no implicit conv)
        return (out.get() if hasattr(out, "get") else np.asarray(out)).astype(np.float32)

    man = {"schema": "cuml-transform-phase1-2026-09-06", "cuml_version": cuml.__version__, "n_ref": int(n),
           "fit_wall_s": round(fit_wall, 1), "params": kw, "vram_after_fit": vram(), "cohorts": {}}

    # CANARY FIRST (fail-fast): 100K training rows — proves transform() works + throughput + VRAM
    rng = np.random.default_rng(0); cidx = np.sort(rng.choice(n, 100_000, replace=False))
    try:
        t = time.time(); cc = transform(X[cidx]); dt = time.time() - t
        man["cohorts"]["canary"] = {"n": 100_000, "throughput_rows_s": round(100_000 / dt, 1), "wall_s": round(dt, 2), "vram_during": vram()}
        print(f"[cuml-tf] CANARY transform OK: {100_000/dt:.0f} rows/s, vram={vram()}", flush=True)
    except Exception as e:
        man["transform_works"] = False; man["canary_error"] = str(e)
        (OUT / "phase1-manifest.json").write_text(json.dumps(man, indent=1))
        print(f"[cuml-tf] CANARY transform FAILED: {e} — transform unusable, recording as such", flush=True); return 0
    man["transform_works"] = True

    # determinism: same 10K twice
    d0 = transform(X[cidx[:10_000]]); d1 = transform(X[cidx[:10_000]])
    man["transform_determinism_max_coord_delta"] = float(np.abs(d0 - d1).max())
    print(f"[cuml-tf] determinism max coord delta {man['transform_determinism_max_coord_delta']:.3e}", flush=True)

    # member reproduction: transform 100K training rows -> displacement vs their FITTED coords
    disp = np.linalg.norm(cc - ref[cidx], axis=1)
    span = float(np.linalg.norm(ref.max(0) - ref.min(0)))
    man["member_reproduction"] = {"n": 100_000, "mean_displacement": round(float(disp.mean()), 5),
                                  "p95_displacement": round(float(np.percentile(disp, 95)), 5),
                                  "map_span": round(span, 3), "mean_frac_of_span": round(float(disp.mean()) / span, 5)}
    np.save(OUT / "member-coords.npy", cc); np.save(OUT / "member-idx.npy", cidx)
    print(f"[cuml-tf] member repro: mean disp {disp.mean():.4f} ({disp.mean()/span*100:.2f}% span)", flush=True)

    # reception cohorts -> save coords for phase 2
    def cohort(name, path, k=100_000, valid=None):
        a = np.load(path, mmap_mode="r"); m = a.shape[0]
        idx = np.arange(m) if valid is None else np.where(valid)[0]
        if k and idx.size > k:
            idx = np.sort(np.random.default_rng(1).choice(idx, k, replace=False))
        x = _norm(np.asarray(a[idx], np.float32))
        t = time.time(); c = transform(x); dt = time.time() - t
        np.save(OUT / f"{name}-coords.npy", c); np.save(OUT / f"{name}-idx.npy", idx)
        man["cohorts"][name] = {"n": int(idx.size), "throughput_rows_s": round(idx.size / dt, 1), "wall_s": round(dt, 2)}
        print(f"[cuml-tf] cohort {name}: {idx.size:,} rows, {idx.size/dt:.0f} rows/s", flush=True)

    cohort("testhd", TESTHD, 100_000)                 # in-distribution held-out (outside 4M draw)
    try:
        cohort("complement", COMPL, 100_000)          # OOD, local column
    except Exception as e:
        man["cohorts"]["complement"] = {"deferred": str(e)}; print(f"[cuml-tf] complement deferred: {e}", flush=True)
    cohort("bl", BL, None, valid=np.load("/data2/monet/bl-clip/bl-valid.npy"))   # full 1.08M OOD

    man["vram_final"] = vram()
    (OUT / "phase1-manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[cuml-tf] PHASE 1 DONE -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
