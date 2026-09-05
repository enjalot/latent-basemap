"""Parametric determinism check (owner/overseer 2026-09-05, Article-2 asymmetry leg). GPU.

Runs the SAME 2M-CLIP champion config TWICE at seed 42 (short horizon — determinism is per-step, not
per-duration) and compares trained-state sha AND coords sha. Equal -> the parametric refit-noise floor is 0
(bitwise-deterministic on CURRENT code) -> the anchored tier's 0.369 triggered churn is entirely growth, vs the
transductive baseline's ~0.193/step randomness tax. NOT equal -> a determinism regression (a real finding); the
Article-2 sentence downgrades to init-reproducibility + measured near-zero empirical variance (the coords churn
between the two runs). Encodes whatever is measured.

Config replicates image_map_pipeline train() exactly (BASE_KWARGS + MD[000] + champion extra), short horizon.
Fits on the 2M-CLIP substrate + the existing 2M fuzzy edges. Usage: determinism_check.py [HORIZON=8000].
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SUB = "/data2/monet/random-2m/clip-substrate.f32.npy"
EDGES = SB / "monet-random-clip-2m/edges-k15-fuzzy.npz"
OUT = SB / "determinism-check-20260905"
HERE = Path(__file__).resolve().parent


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _state_sha(model):
    h = hashlib.sha256()
    for p in model.model.parameters():
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    from _paths import ensure_paths; ensure_paths()
    import image_map_pipeline as m
    from knobs_2m import BASE_KWARGS, MD
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch, math

    x = _norm(np.asarray(np.load(SUB, mmap_mode="r"), np.float32))
    champ_extra = m.DATASETS["monet-random-clip-2m"]["arms"]["champion-bs16k"]["extra"]
    kwargs = dict(BASE_KWARGS)
    kwargs.update({"low_dim_kernel": "umap", **MD["000"], **champ_extra,
                   "total_steps_estimate": horizon, "n_epochs": max(1, math.ceil(horizon * 16384 * 0.10 / (x.shape[0] * 15)))})

    def run(label):
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        model = ParametricUMAP(**kwargs)
        t0 = time.time()
        model.fit(x, precomputed_edges_path=str(EDGES), random_state=42)
        sha = _state_sha(model)
        coords = np.asarray(model.transform(x, batch_size=16384), dtype=np.float32)
        csha = hashlib.sha256(np.ascontiguousarray(coords).tobytes()).hexdigest()[:16]
        print(f"[det] run {label}: trained_sha {sha} coords_sha {csha} ({time.time()-t0:.0f}s)", flush=True)
        return sha, csha, coords

    shaA, cshaA, cA = run("A")
    shaB, cshaB, cB = run("B")
    state_eq = shaA == shaB; coords_eq = cshaA == cshaB
    # empirical variance even if not bitwise-equal (max/mean abs coord diff)
    d = np.abs(cA.astype(np.float64) - cB.astype(np.float64))
    out = {"schema": "parametric-determinism-check-2026-09-05", "horizon": horizon, "seed": 42,
           "n": int(x.shape[0]), "trained_sha": [shaA, shaB], "coords_sha": [cshaA, cshaB],
           "trained_state_equal": bool(state_eq), "coords_equal": bool(coords_eq),
           "coord_absdiff_mean": round(float(d.mean()), 8), "coord_absdiff_max": round(float(d.max()), 8),
           "verdict": ("BITWISE-DETERMINISTIC (refit-noise floor = 0 on current code)" if (state_eq and coords_eq)
                       else "NOT bitwise-equal — downgrade to init-reproducible + near-zero empirical variance"),
           "article2_leg": ("parametric refit noise = 0 (confirmed current-code)" if (state_eq and coords_eq)
                            else f"parametric refit noise near-zero: mean coord diff {float(d.mean()):.2e} (not bitwise; regression to locate)")}
    (OUT / "determinism.json").write_text(json.dumps(out, indent=1))
    print(f"[det] state_equal={state_eq} coords_equal={coords_eq} | {out['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
