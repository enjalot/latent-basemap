#!/usr/bin/env python3
"""Evolution benchmark — ARM B (cuVS+procrustes), the competitor baseline (plan §2). cuml-env, GPU.

For each snapshot Sk = T0∪T1..Tk (k=0..K), run a FULL transductive UMAP (cuml in-core nn_descent — the
feasible path; out-of-core deadlocks in cuml 25.02), then procrustes-align its layout to the PREVIOUS
snapshot's layout on the SHARED unchanged points. Sk's concat order is [T0,T1,..,Tk], so the first
n_{k-1} rows ARE S_{k-1} in the same order → the shared points are exactly coords[:n_{k-1}].

Emits per snapshot: aligned coords (for the scorer's churn + quality), wall (placement latency), peak VRAM
(cost proxy). Churn itself (displacement of the shared points vs the previous snapshot, cumulative) is
computed by the scorer from these coords — measured against the PREVIOUS snapshot (not T0) so arm B's
repeated reshuffles accumulate rather than cancel [overseer 2026-08-31].

Env: EVOLBENCH_K (default 5), EVOLBENCH_EPOCHS (default 500), EVOLBENCH_DIR. Output coords ->
<sandbox>/evolbench-armB/coords-S{k}.npy + armB-manifest.json.
"""
import json, os, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
OUTD = SB / "evolbench-armB"
K = int(os.environ.get("EVOLBENCH_K", "5"))
EPOCHS = int(os.environ.get("EVOLBENCH_EPOCHS", "500"))
DIM = 384


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _load_Sk(k):
    # Sk = concat([T0, T1, .., Tk]) in that order (first n_{k-1} rows = S_{k-1}).
    parts = ["T0"] + [f"T{j}" for j in range(1, k + 1)]
    return _norm(np.concatenate([
        np.asarray(np.load(f"/data/latent-basemap/substrates/evolbench/{t}/substrate.f32.npy",
                           mmap_mode="r"), dtype=np.float32) for t in parts]))


def _procrustes_align(src, ref_shared, src_shared):
    """Return src mapped by the similarity transform (scale*R + t) that best aligns src_shared→ref_shared
    (orthogonal Procrustes with scaling), applied to ALL of src. Reference frame = the PREVIOUS snapshot."""
    mu_s = src_shared.mean(0); mu_r = ref_shared.mean(0)
    A = src_shared - mu_s; B = ref_shared - mu_r
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    scale = S.sum() / (A ** 2).sum()
    return (src - mu_s) @ (scale * R) + mu_r


def _peak_vram(stop, peak):
    while not stop.is_set():
        try:
            u = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                         "--format=csv,noheader,nounits"], timeout=5)
            peak[0] = max(peak[0], int(u.decode().splitlines()[0]) / 1024.0)
        except Exception:
            pass
        time.sleep(1.0)


def main():
    from cuml.manifold import UMAP
    OUTD.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "evolbench-armB-2026-08-31", "epochs": EPOCHS, "snapshots": []}
    prev_coords = None; prev_n = 0; cum_cost = 0.0
    for k in range(K + 1):
        X = _load_Sk(k); n = int(X.shape[0])
        stop = threading.Event(); peak = [0.0]
        th = threading.Thread(target=_peak_vram, args=(stop, peak), daemon=True); th.start()
        t0 = time.time()
        um = UMAP(n_neighbors=15, n_components=2, n_epochs=EPOCHS, min_dist=0.1, random_state=42,
                  build_algo="nn_descent")
        coords = np.asarray(um.fit_transform(X), dtype=np.float32)
        wall = time.time() - t0
        stop.set(); th.join(timeout=2); cum_cost += wall
        # procrustes-align to PREVIOUS snapshot on the shared first-n_{k-1} points
        if prev_coords is not None:
            coords = _procrustes_align(coords, prev_coords, coords[:prev_n])
        np.save(OUTD / f"coords-S{k}.npy", coords)
        manifest["snapshots"].append({"k": k, "n": n, "wall_s": round(wall, 1),
                                      "peak_vram_gb": round(peak[0], 2),
                                      "cum_gpu_s": round(cum_cost, 1)})
        print(f"armB S{k}: n={n:,} wall={wall:.0f}s vram={peak[0]:.1f}GB (cum {cum_cost:.0f}s)", flush=True)
        prev_coords = coords; prev_n = n
        del X, um
    (OUTD / "armB-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {OUTD}/armB-manifest.json ({K+1} snapshots)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
