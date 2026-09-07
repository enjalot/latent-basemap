"""VRAM CANARY LADDER — SYNTHETIC-TOPOLOGY, MEMORY/THROUGHPUT ONLY, NO QUALITY CLAIMS (owner GO 2026-09-07).

Measures where the champion device-int8-resident path (768-d PCA substrate) stops fitting in the 5090's 32 GB, and
its steady-state it/s below that ceiling. EVERYTHING here is fabricated: the substrate is an all-zero sparse memmap
(the int8 quantizer floors all-zero rows to a positive fp16 scale, so it quantizes cleanly — content is irrelevant
to the resident-set SIZE) and the graph is random k15-shaped pairs. This says NOTHING about map quality; it is a
pure memory + throughput probe of the device_int8 code path at scale.

Per rung N: champion kwargs (md000/dose4/pos0.10/rankneg=25%N/bs16k) + x_residency=device_int8, ~200 bench steps
(20 warmup excluded). Records: the fail-closed pre-check verdict + breakdown (need_x int8+scales, resident edge
bytes, margin, free VRAM), and when it RUNS: torch.cuda.max_memory_{allocated,reserved} + steady-state it/s + where
the int8 X buffer lives (device). Rungs 20/25M bracket the 30/35/40M the owner asked for, to locate the actual
comfortable-max rather than only confirm the top fails. GPU (flock-held). Writes vram-canary/ladder.json.
"""
import json, os, time, gc
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); OUT = SB / "vram-canary"; DIM = 768
RUNGS = [20_000_000, 25_000_000, 30_000_000, 35_000_000, 40_000_000]
NMAX = RUNGS[-1]; BENCH_STEPS = 220; BENCH_WARMUP = 20; K = 15


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from knobs_2m import BASE_KWARGS, MD
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch

    # sparse all-zero synthetic substrate (NMAX x 768 f32); sparse file -> ~0 disk; slice [:N] per rung.
    sub_path = OUT / "synthetic-substrate.f32.npy"
    sub = np.lib.format.open_memmap(sub_path, mode="w+", dtype=np.float32, shape=(NMAX, DIM))  # sparse zeros
    del sub  # written lazily; reopen mmap per rung
    free0, total = torch.cuda.mem_get_info()
    print(f"[canary] GPU free {free0/1e9:.1f} / {total/1e9:.1f} GB | synthetic substrate {sub_path} (sparse, all-zero)", flush=True)

    results = []
    for N in RUNGS:
        rng = np.random.default_rng(42)
        # fabricated k15-shaped edges: each of N nodes -> 15 random targets in [0,N). Temp npz, deleted after the rung.
        edges = OUT / f"edges-{N//1_000_000}m.npz"
        srcs = np.repeat(np.arange(N, dtype=np.int32), K)
        tgts = rng.integers(0, N, size=N * K, dtype=np.int32)
        wts = np.ones(N * K, dtype=np.float32)
        np.savez(edges, sources=srcs, targets=tgts, weights=wts, n_nodes=np.int64(N))
        del srcs, tgts, wts; gc.collect()
        edge_bytes = N * K * 12                                   # src+dst i32 + wt f32 resident on device
        need_x = N * DIM * 1 + N * 2                              # int8 codes + fp16 scales
        x = np.load(sub_path, mmap_mode="r")[:N]

        kwargs = dict(BASE_KWARGS)
        kwargs.update({"low_dim_kernel": "umap", **MD["000"], "fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                       "pos_ratio": 0.10, "rankneg_window": int(0.25 * N), "batch_size": 16384,
                       "x_residency": "device_int8", "n_epochs": 1, "total_steps_estimate": BENCH_STEPS})
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        free_pre, _ = torch.cuda.mem_get_info()
        rec = {"N": N, "need_x_gb": round(need_x / 1e9, 2), "resident_edge_gb": round(edge_bytes / 1e9, 2),
               "margin_gb": 3.5, "predicted_total_gb": round((need_x + edge_bytes) / 1e9 + 3.5, 2),
               "free_vram_gb": round(free_pre / 1e9, 2)}
        model = None
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
            print(f"[canary] N={N//1_000_000}M RAN | max_alloc {rec['max_alloc_gb']} reserved {rec['max_reserved_gb']} GB "
                  f"| {its} it/s | X on {xdev}", flush=True)
        except RuntimeError as e:
            msg = str(e).replace("\n", " ")
            rec.update({"verdict": "PRECHECK_FAIL" if "resident set exceeds" in msg else "RUNTIME_ERROR",
                        "error": msg[:400]})
            print(f"[canary] N={N//1_000_000}M {rec['verdict']} | predicted {rec['predicted_total_gb']} GB vs free {rec['free_vram_gb']} GB", flush=True)
        finally:
            # CRITICAL: free the resident buffers before the next rung — a leaked model here poisons the
            # next rung's pre-check free-VRAM read (that contaminated the first run). del + empty_cache always.
            try: del model
            except Exception: pass
            gc.collect(); torch.cuda.empty_cache()
        results.append(rec)
        try: edges.unlink()
        except OSError: pass

    out = {"schema": "vram-canary-ladder-2026-09-07", "kind": "SYNTHETIC-TOPOLOGY memory/throughput canary — NO quality claims",
           "substrate": "all-zero sparse memmap (int8 quantizer floors zero rows); content irrelevant to resident-set size",
           "graph": "fabricated random k15-shaped pairs (15 targets/node)", "dim": DIM, "gpu_total_gb": round(total / 1e9, 1),
           "recipe": "champion md000/dose4/pos0.10/rankneg=25%N/bs16k, x_residency=device_int8", "bench_steps": BENCH_STEPS,
           "rungs": results,
           "note": "device_int8 pre-check (core.py): fail-closed if need_x(int8+scales)+resident_edges+3.5GB margin > free VRAM. "
                   "comfortable-max = largest N with verdict RAN and headroom. it/s is steady-state (warmup excluded)."}
    (OUT / "ladder.json").write_text(json.dumps(out, indent=1))
    try: Path(sub_path).unlink()          # reclaim the sparse substrate
    except OSError: pass
    print(f"[canary] DONE -> {OUT/'ladder.json'}", flush=True)
    for r in results:
        print(f"  N={r['N']//1_000_000:>2}M  {r['verdict']:<13} predicted {r['predicted_total_gb']:>5} GB  "
              f"alloc {r.get('max_alloc_gb','-')} reserved {r.get('max_reserved_gb','-')}  {r.get('steady_it_per_s','-')} it/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
