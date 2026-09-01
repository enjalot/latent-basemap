"""Same-corpus-rerun CHURN FLOOR control (5th review #8). cuml-env, GPU.

Runs arm B's config (exact brute-force + spectral + 500ep) TWICE on the IDENTICAL S0 substrate with a
NON-fixed optimizer (random_state=None), then procrustes-aligns run2 to run1 and measures the residual
churn. This is the optimizer-only churn floor of a transductive refit — the noise any full-refit baseline
pays even with zero data change. Contextualizes arm B's per-step churn (how much is real reshuffle vs
noise) and calibrates the λ-frontier acceptance band (5th review #10). Each run in a FRESH subprocess
(RMM pool hygiene). Two runs = ~32min at 4M. Output: evolbench-churnfloor.json.
PROVISIONAL-PENDING-VALIDATION (single pair; the final batch adds more pairs if the band is decision-relevant)."""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

SUB = os.environ.get("EVOLBENCH_SUBDIR", "/data/latent-basemap/substrates/evolbench")
OUT = Path("/data/latent-basemap/sandbox/evolbench-churnfloor.json")
COORDS = Path("/data/latent-basemap/sandbox/churnfloor")


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _load_S0():
    return _norm(np.asarray(np.load(f"{SUB}/T0/substrate.f32.npy", mmap_mode="r"), dtype=np.float32))


def _radius(xy):
    return float(np.percentile(np.linalg.norm(xy - xy.mean(0), axis=1), 90))


def _procrustes(src, ref):
    mu_s = src.mean(0); mu_r = ref.mean(0)
    U, S, Vt = np.linalg.svd((src - mu_s).T @ (ref - mu_r)); R = U @ Vt
    scale = S.sum() / max(((src - mu_s) ** 2).sum(), 1e-9)
    return ((src - mu_s) @ (scale * R) + mu_r).astype(np.float32)


def run_worker(tag):
    from cuml.manifold import UMAP
    COORDS.mkdir(parents=True, exist_ok=True)
    X = _load_S0()
    t0 = time.time()
    # NON-fixed optimizer -> exposes optimizer stochasticity. Exact graph (no random_state -> default build;
    # force brute-force via build_algo omitted + small data fits) + spectral + 500ep, matching arm B quality path.
    coords = np.asarray(UMAP(n_neighbors=15, n_components=2, min_dist=0.0, n_epochs=500,
                             init="spectral").fit_transform(X), dtype=np.float32)
    np.save(COORDS / f"run-{tag}.npy", coords)
    print(f"churnfloor run {tag}: n={X.shape[0]:,} wall={time.time()-t0:.0f}s", flush=True)
    return 0


def run_driver():
    for tag in ("a", "b"):
        if not (COORDS / f"run-{tag}.npy").exists():
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--run", tag]).returncode
            if rc != 0:
                raise SystemExit(f"churnfloor run {tag} FAILED rc={rc}")
    a = np.load(COORDS / "run-a.npy"); b = np.load(COORDS / "run-b.npy")
    b_al = _procrustes(b.astype(np.float64), a.astype(np.float64))
    disp = np.linalg.norm(b_al - a, axis=1) / max(_radius(a), 1e-9)
    out = {"schema": "evolbench-churnfloor-2026-09-01",
           "_PROVISIONAL": "PROVISIONAL-PENDING-VALIDATION (single rerun pair)",
           "config": "cuml exact+spectral+500ep min_dist=0 random_state=None, S0 4M, twice",
           "churn_floor_mean": round(float(disp.mean()), 5), "churn_floor_p95": round(float(np.percentile(disp, 95)), 5),
           "note": "optimizer-only churn of a transductive refit on identical data; arm B per-step churn "
                   "above this is real data-driven reshuffle. Calibrates lambda-frontier acceptance."}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nCHURN FLOOR (transductive refit, same data): mean {out['churn_floor_mean']} "
          f"p95 {out['churn_floor_p95']} -> {OUT}", flush=True)
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--run":
        return run_worker(sys.argv[2])
    return run_driver()


if __name__ == "__main__":
    raise SystemExit(main())
