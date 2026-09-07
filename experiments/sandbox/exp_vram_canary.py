"""VRAM CANARY LADDER — SYNTHETIC-TOPOLOGY, MEMORY/THROUGHPUT ONLY, NO QUALITY CLAIMS (owner GO 2026-09-07).

Measures where the champion device-int8-resident path (768-d PCA substrate) stops fitting in the 5090's 32 GB, and
its steady-state it/s below that ceiling. EVERYTHING is fabricated: the substrate is an all-zero sparse memmap (the
int8 quantizer floors all-zero rows to a positive fp16 scale, so it quantizes cleanly — content is irrelevant to
the resident-set SIZE) and the graph is random k15-shaped pairs. Says NOTHING about map quality; a pure memory +
throughput probe of the device_int8 code path at scale.

ONE RUNG PER PROCESS (arg N): in-process CUDA + a CUDA-OOM RuntimeError's traceback pin device tensors so the next
rung can't get clean free-VRAM — so run_vram_canary.sh invokes this once per N and each process exits (fully
freeing the GPU) before the next. Writes vram-canary/rung-<N>.json. Records the fail-closed pre-check verdict +
breakdown, and when it RUNS: torch.cuda.max_memory_{allocated,reserved} + steady it/s + int8-X buffer device.
Usage: exp_vram_canary.py <N>
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); OUT = SB / "vram-canary"; DIM = 768
NMAX = 40_000_000; BENCH_STEPS = 220; BENCH_WARMUP = 20; K = 15


def main():
    N = int(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from knobs_2m import BASE_KWARGS, MD
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch

    sub_path = OUT / "synthetic-substrate.f32.npy"
    if not sub_path.exists():
        m = np.lib.format.open_memmap(sub_path, mode="w+", dtype=np.float32, shape=(NMAX, DIM)); del m  # sparse zeros
    rng = np.random.default_rng(42)
    edges = OUT / f"edges-{N//1_000_000}m.npz"
    srcs = np.repeat(np.arange(N, dtype=np.int32), K); tgts = rng.integers(0, N, size=N * K, dtype=np.int32)
    np.savez(edges, sources=srcs, targets=tgts, weights=np.ones(N * K, np.float32), n_nodes=np.int64(N)); del srcs, tgts
    edge_bytes = N * K * 12; need_x = N * DIM + N * 2
    x = np.load(sub_path, mmap_mode="r")[:N]

    kwargs = dict(BASE_KWARGS)
    kwargs.update({"low_dim_kernel": "umap", **MD["000"], "fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                   "pos_ratio": 0.10, "rankneg_window": int(0.25 * N), "batch_size": 16384,
                   "x_residency": "device_int8", "n_epochs": 1, "total_steps_estimate": BENCH_STEPS})
    torch.cuda.reset_peak_memory_stats(); free_pre, total = torch.cuda.mem_get_info()
    rec = {"N": N, "need_x_gb": round(need_x / 1e9, 2), "resident_edge_gb": round(edge_bytes / 1e9, 2),
           "margin_gb": 3.5, "predicted_total_gb": round((need_x + edge_bytes) / 1e9 + 3.5, 2),
           "free_vram_gb": round(free_pre / 1e9, 2), "gpu_total_gb": round(total / 1e9, 1)}
    try:
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        model = ParametricUMAP(**kwargs)
        model._max_train_steps = BENCH_STEPS; model._bench_warmup = BENCH_WARMUP
        t0 = time.time(); model.fit(x, precomputed_edges_path=str(edges), random_state=42); wall = time.time() - t0
        bench_s = getattr(model, "_bench_seconds", None)
        its = round((BENCH_STEPS - BENCH_WARMUP) / bench_s, 1) if bench_s else None
        xdev = str(getattr(getattr(model, "_X_dev", None), "device", "unknown"))
        rec.update({"verdict": "RAN", "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                    "steady_it_per_s": its, "bench_seconds": round(bench_s, 2) if bench_s else None,
                    "wall_s": round(wall, 1), "int8_X_buffer_device": xdev})
        print(f"[canary] N={N//1_000_000}M RAN | max_alloc {rec['max_alloc_gb']} reserved {rec['max_reserved_gb']} GB | {its} it/s | X on {xdev}", flush=True)
    except RuntimeError as e:
        msg = str(e).replace("\n", " ")
        rec.update({"verdict": "PRECHECK_FAIL" if "resident set exceeds" in msg else "RUNTIME_ERROR", "error": msg[:400]})
        print(f"[canary] N={N//1_000_000}M {rec['verdict']} | predicted {rec['predicted_total_gb']} GB vs free {rec['free_vram_gb']} GB", flush=True)
    (OUT / f"rung-{N//1_000_000}m.json").write_text(json.dumps(rec, indent=1))
    try: edges.unlink()
    except OSError: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
