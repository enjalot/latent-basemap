#!/usr/bin/env python3
"""Deterministic micro-benchmark for the edge-list training input pipeline.

Compares the legacy per-batch sampler path (``gpu_resident_data=false``:
``EdgeListBalancedIterator`` -> list-of-tuples -> ``DataPrefetcher`` host->device
gather) against the GPU-resident fast path (``gpu_resident_data=true``:
``DeviceEdgeSampler`` -> on-device ``index_select``), with and without the
PaCMAP mid-near term. Reports samples/s (edge-pairs/s) for each.

Runs the *real* ``ParametricUMAP.fit`` loop for exactly ``--steps`` measured
training steps on synthetic data. Setup cost (X upload, edge load, model init)
is cancelled by subtracting a short warmup run, so the number reflects the
steady-state step throughput.

CPU is the default so it is safe to run while the GPU is reserved. Rerun on the
5090 with a single flag::

    python experiments/bench_input_pipeline.py --device cuda

The synthetic shape defaults to 500k x 768 fp32, batch 8192, pos_ratio 0.20 --
matching experiments/configs/jina_en_200k_k50.yaml at ~2.5x the rows.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_synthetic_edges(tmp_path, n_rows, k, seed):
    """Random directed edge list matching the build_*_index_modal npz schema."""
    rng = np.random.RandomState(seed)
    sources = np.repeat(np.arange(n_rows, dtype=np.int32), k)
    targets = rng.randint(0, n_rows, size=n_rows * k).astype(np.int32)
    weights = rng.uniform(0.01, 1.0, size=n_rows * k).astype(np.float32)
    path = Path(tmp_path) / "bench_edges.npz"
    np.savez(path, sources=sources, targets=targets, weights=weights,
             n_nodes=n_rows, k=k)
    return str(path)


def time_fit(X, edges_path, device, gpu_resident, midnear, batch_size,
             pos_ratio, steps, seed):
    """Return (seconds, fast_path) for exactly ``steps`` measured steps.

    Timing is taken *inside* fit's training loop (fit._bench_seconds) between the
    warmup boundary and the stop step, so one-time setup (X upload, edge load,
    model init, cudnn autotune) is excluded and the number reflects steady-state
    step throughput.
    """
    from basemap.pumap.parametric_umap import ParametricUMAP

    warmup = max(5, steps // 5)
    pumap = ParametricUMAP(
        architecture="residual_bottleneck", hidden_dim=1024, n_layers=3,
        n_components=2, a=1.0, b=1.0, correlation_weight=0.0,
        learning_rate=1e-3, n_epochs=100000, batch_size=batch_size,
        pos_ratio=pos_ratio, device=device,
        use_amp=("cuda" in str(device)),
        positive_target_mode="binary", lr_schedule="cosine",
        warmup_steps=200, total_steps_estimate=500000,
        clip_grad_norm=1.0,
        midnear_enabled=midnear, mn_pairs_per_batch=0, mn_weight_scale=1.0,
        gpu_resident_data=("true" if gpu_resident else "false"),
        gpu_resident_vram_budget_gb=1e6,  # never budget-limited in the bench
    )
    pumap._bench_warmup = warmup
    pumap._max_train_steps = warmup + steps
    pumap.fit(X, precomputed_edges_path=edges_path, random_state=seed,
              verbose=False)
    return max(pumap._bench_seconds, 1e-9), pumap._fast_device_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, GPU-safe) or cuda")
    ap.add_argument("--n-rows", type=int, default=500_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=10, help="edges per row")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--pos-ratio", type=float, default=0.20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-midnear", action="store_true",
                    help="only benchmark the no-midnear configs (faster)")
    ap.add_argument("--memmap", action="store_true",
                    help="back X with an on-disk fp16 .npy via "
                         "MemmapArrayConcatenator, matching the real "
                         "source: memmap configs (legacy path then does "
                         "per-batch disk reads instead of a one-time upload)")
    args = ap.parse_args()

    import tempfile
    tmp = tempfile.mkdtemp(prefix="bench_pipeline_")

    print(f"# Input-pipeline benchmark")
    print(f"# device={args.device}  X=({args.n_rows}, {args.dim}) fp32  "
          f"edges={args.n_rows * args.k:,}  batch={args.batch_size}  "
          f"pos_ratio={args.pos_ratio}  measured_steps={args.steps}")

    rng = np.random.RandomState(args.seed)
    edges_path = make_synthetic_edges(tmp, args.n_rows, args.k, args.seed)

    if args.memmap:
        # Write X as an on-disk fp16 shard and load it lazily, exactly like the
        # real source: memmap configs (fineweb jina embeddings are fp16 .npy).
        from basemap.data_loader import MemmapArrayConcatenator
        shard_dir = Path(tmp) / "X_shard"
        shard_dir.mkdir(parents=True, exist_ok=True)
        Xd = rng.standard_normal((args.n_rows, args.dim)).astype(np.float16)
        np.save(shard_dir / "data-00000-of-00001.npy", Xd)
        del Xd
        X = MemmapArrayConcatenator([str(shard_dir)], args.dim)
        print(f"# X backed by on-disk fp16 memmap ({shard_dir})")
    else:
        X = rng.standard_normal((args.n_rows, args.dim)).astype(np.float32)

    # optimized-first so a partial/timed-out GPU run still yields the fast
    # numbers before the slow legacy path.
    configs = [
        ("optimized(gpu_resident=true )", True, False),
        ("legacy   (gpu_resident=false)", False, False),
    ]
    if not args.skip_midnear:
        configs += [
            ("optimized + midnear", True, True),
            ("legacy    + midnear", False, True),
        ]

    print()
    hdr = f"{'config':32s} {'fast?':5s} {'samples/s':>12s} {'batches/s':>10s} {'ms/step':>8s}"
    print(hdr)
    print("-" * len(hdr))
    # Measure + print each row as we go so a timed-out run still yields partials.
    base = {}
    for label, gpu_resident, midnear in configs:
        secs, fast = time_fit(
            X, edges_path, args.device, gpu_resident, midnear,
            args.batch_size, args.pos_ratio, args.steps, args.seed)
        sps = (args.steps * args.batch_size) / secs
        bps = args.steps / secs
        ms = 1000.0 * secs / args.steps
        print(f"{label:32s} {str(fast):5s} {sps:12,.0f} {bps:10.1f} {ms:8.1f}",
              flush=True)
        base[(label.startswith('optimized'), 'midnear' in label)] = sps

    # Speedups (optimized vs legacy) for each midnear setting.
    print()
    for mn in (False, True):
        leg = base.get((False, mn))
        opt = base.get((True, mn))
        if leg and opt:
            tag = "with midnear" if mn else "no midnear "
            print(f"speedup {tag}: {opt / leg:6.2f}x  ({opt:,.0f} vs {leg:,.0f} samples/s)")


if __name__ == "__main__":
    main()
