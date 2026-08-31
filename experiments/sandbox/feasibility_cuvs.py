#!/usr/bin/env python3
"""cuVS feasibility check for the evolution benchmark's arm B (plan §2, owner 2026-08-31).
Runs in cuml-env. Arm B must run FULL transductive UMAP at the benchmark's final corpus size
(final = 2xT0). Tests cuml UMAP at candidate scales with the nn-descent + data-on-host (out-of-core)
path (0197) so the graph build can exceed in-core VRAM. Reports wall + peak VRAM per scale; the caller
picks the LARGEST scale that runs comfortably (ratios fixed): T0=5M->final 10M target, T0=2M->4M fallback.

384-dim MiniLM shape. Synthetic data is fine (UMAP wall/memory depend on N+dim+graph, not content).
Output: /data/latent-basemap/sandbox/cuvs-feasibility.json
"""
import json, os, time, traceback
from pathlib import Path
import numpy as np

DIM = 384
SCALES = [int(s) for s in os.environ.get("CUVS_SCALES", "4000000,8000000,10000000").split(",")]
N_EPOCHS = int(os.environ.get("CUVS_EPOCHS", "200"))   # short: feasibility, not a quality run
OUT = Path("/data/latent-basemap/sandbox/cuvs-feasibility.json")


def main():
    import cupy
    from cuml.manifold import UMAP
    results = []
    for N in SCALES:
        rec = {"N": N, "dim": DIM, "n_epochs": N_EPOCHS, "path": "nn_descent+data_on_host"}
        try:
            rng = np.random.default_rng(0)
            X = rng.standard_normal((N, DIM), dtype=np.float32)  # host-resident
            cupy.get_default_memory_pool().free_all_blocks()
            t0 = time.time()
            um = UMAP(n_neighbors=15, n_components=2, n_epochs=N_EPOCHS, min_dist=0.1,
                      build_algo="nn_descent", build_kwds={"nnd_data_on_host": True},
                      random_state=42, verbose=False)
            emb = um.fit_transform(X)
            wall = time.time() - t0
            mp = cupy.get_default_memory_pool()
            rec.update({"status": "OK", "wall_s": round(wall, 1),
                        "peak_vram_gb": round(mp.total_bytes() / 1e9, 2),
                        "emb_shape": list(emb.shape)})
            print(f"N={N/1e6:.0f}M: OK wall {wall:.0f}s peak_vram {rec['peak_vram_gb']}GB", flush=True)
            del X, emb, um
            mp.free_all_blocks()
        except Exception as e:
            rec.update({"status": f"FAIL: {type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-800:]})
            print(f"N={N/1e6:.0f}M: FAIL {type(e).__name__}: {e}", flush=True)
            try:
                cupy.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        results.append(rec)

    ok = [r for r in results if r.get("status") == "OK"]
    largest_ok = max((r["N"] for r in ok), default=0)
    # recommend T0 = largest comfortable final / 2. comfortable = wall < ~20 min AND vram headroom.
    comfortable = [r for r in ok if r.get("wall_s", 1e9) < 1200 and r.get("peak_vram_gb", 99) < 28]
    largest_comfortable = max((r["N"] for r in comfortable), default=0)
    rec_scale = {"largest_ok_final": largest_ok, "largest_comfortable_final": largest_comfortable,
                 "recommended_T0": largest_comfortable // 2 if largest_comfortable else 0,
                 "note": ("arm B reruns UMAP at EVERY snapshot up to final=2xT0, so 'comfortable' must "
                          "hold at the final size AND the per-rerun wall must not eat the benchmark "
                          "budget (5 reruns). T0=5M needs final-10M comfortable; else fall to T0=2M.")}
    OUT.write_text(json.dumps({"schema": "cuvs-feasibility-2026-08-31", "scales": results,
                               "scale_decision": rec_scale}, indent=1))
    print(f"\n=== SCALE: largest_comfortable_final={largest_comfortable/1e6:.0f}M -> "
          f"recommended_T0={rec_scale['recommended_T0']/1e6:.0f}M ===\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
