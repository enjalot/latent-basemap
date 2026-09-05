"""Noise-floor for Article 2's full-UMAP churn claim (owner item e, overseer spec 2026-09-05). CPU, umap-learn
0.6 (umap06dev-env) — MUST match the competitor baseline implementation (NOT cuML: that would confound
implementation with randomness). GPU untouched (stays with the drain).

Same-snapshot rerun pair = the optimizer-randomness floor: TWO full umap-learn fits of the SAME evolbench S3
data (6.4M = T0+T1+T2+T3), identical params, differing only in run randomness -> rigid-align (frame.py) ->
churn between the pair. This bounds how much of the full-UMAP timeline's cumulative churn is growth-response vs
noise: growth-churn >= measured_timeline - floor.

PARAMS match the competitor timeline (p_evolbench_competitor.py): n_neighbors=15, min_dist=0.0, n_components=2,
metric=cosine. RANDOMNESS choice (documented): random_state=None for BOTH fits — matches the baseline's
multi-threaded config; setting random_state=42/43 would force umap single-threaded, confounding THREADING with
randomness (violating the same-implementation requirement) and making 6.4M CPU-infeasible. Two independent
random_state=None fits give the pure run-to-run floor with the baseline's own threading.

Env: EVOLBENCH_TRANCHES (default T0,T1,T2,T3 = S3 6.4M; T0,T1 = S1 4.8M), CANARY_ROWS (subsample for timing).
Run with: /data/latent-basemap/umap06dev-env/bin/python noise_floor_umap.py <out_tag>
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

EVOL = Path("/data/latent-basemap/substrates/evolbench")
OUT = Path("/data/latent-basemap/sandbox/noise-floor-20260905")
HERE = Path(__file__).resolve().parent


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "S3"
    tranches = os.environ.get("EVOLBENCH_TRANCHES", "T0,T1,T2,T3").split(",")
    canary = int(os.environ.get("CANARY_ROWS", "0"))
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import frame
    import umap

    X = _norm(np.concatenate([np.asarray(np.load(EVOL / t / "substrate.f32.npy", mmap_mode="r"), np.float32)
                              for t in tranches]))
    if canary > 0 and canary < X.shape[0]:
        rng = np.random.default_rng(0); X = X[np.sort(rng.choice(X.shape[0], canary, replace=False))]
    n = X.shape[0]
    print(f"[floor {tag}] {tranches} -> {n:,} x {X.shape[1]}  (canary={canary or 'no'})", flush=True)

    def fit(label):
        t0 = time.time()
        c = np.asarray(umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, metric="cosine",
                                 random_state=None).fit_transform(X), np.float32)
        w = time.time() - t0
        print(f"[floor {tag}] fit {label} done {w:.0f}s ({w/60:.1f}min)", flush=True)
        np.save(OUT / f"coords-{tag}-{label}.npy", c)
        return c, w

    cA, wA = fit("A")
    cB, wB = fit("B")
    disp, info = frame.churn(cA.astype(np.float64), cB.astype(np.float64))   # rigid-aligned + frame-normed
    churn = {"mean": round(float(disp.mean()), 5), "p95": round(float(np.percentile(disp, 95)), 5),
             "median": round(float(np.median(disp)), 5), "learned_scale": info.get("learned_scale"),
             "rmsd": info.get("rmsd")}
    out = {"schema": "noise-floor-umap-2026-09-05", "tag": tag, "tranches": tranches, "n_rows": int(n),
           "canary_rows": canary or None,
           "params": {"lib": f"umap-learn {umap.__version__}", "n_neighbors": 15, "min_dist": 0.0,
                      "metric": "cosine", "n_components": 2, "random_state": "None (both fits)"},
           "randomness_note": "random_state=None both fits — matches baseline umap06dev threading; 42/43 would "
                              "force single-thread, confounding implementation with randomness (overseer's warning).",
           "fit_wall_s": [round(wA, 1), round(wB, 1)], "churn": churn,
           "fixed_seed_timeline": "DEFERRED to the owner-deferred full multi-baseline campaign (overseer 2026-09-05): "
                                  "the same-snapshot floor already bounds growth-churn >= timeline - floor.",
           "article2_use": "of the full-UMAP timeline's cumulative churn C, at least (C - floor_mean) is growth "
                           "response above the optimizer floor. Recompute C with THIS churn instrument for a clean subtraction."}
    (OUT / f"floor-{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[floor {tag}] CHURN {churn} -> {OUT/('floor-'+tag+'.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
