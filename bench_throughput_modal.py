"""
Benchmark training throughput optimizations for CPU→GPU data transfer.
Tests: baseline, pinned memory, async prefetch, GPU-resident data.

Usage:
  modal run bench_throughput_modal.py
"""
from basemap.round0005_retirement import refuse_retired_launcher

refuse_retired_launcher("bench_throughput_modal.py")

import time
import logging
import numpy as np
from modal import App, Image, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.1.2", "numpy==1.26.3", "scipy", "tqdm")
    .add_local_python_source("basemap")
)

with st_image.imports():
    import torch as _torch

app = App("bench-throughput")
VOLUMES = {
    "/embeddings": Volume.from_name("embeddings"),
    "/checkpoints": Volume.from_name("checkpoints"),
}


@app.function(gpu="A10G", timeout=60*30, image=st_image, volumes=VOLUMES, memory=32768)
def bench_all():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from basemap.data_loader import MemmapArrayConcatenator
    from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

    device = "cuda"
    D_IN = 384
    HIDDEN = 512
    BS = 4096
    N_SAMPLES = 5_000_000  # 5M — fits on GPU (7.7 GB)
    SECONDS = 30  # each test runs for 30s

    # Load data
    loader = MemmapArrayConcatenator(
        ["/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train"], D_IN)
    X_np = np.ascontiguousarray(np.asarray(loader[:N_SAMPLES]).astype(np.float32))
    logging.info(f"Data: {X_np.shape} ({X_np.nbytes/1e9:.1f} GB)")

    # Load edges
    edges = np.load("/checkpoints/pumap/edges_15m_k15.npz")
    mask = (edges["sources"] < N_SAMPLES) & (edges["targets"] < N_SAMPLES)
    pos_src = edges["sources"][mask]
    pos_dst = edges["targets"][mask]
    pos_wt = edges["weights"][mask]
    logging.info(f"Edges: {len(pos_src):,} after filtering")

    # Model
    class UMAPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = nn.Linear(D_IN, HIDDEN)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
                for _ in range(3)
            ])
            self.out_norm = nn.LayerNorm(HIDDEN)
            self.proj_out = nn.Linear(HIDDEN, 2)
        def forward(self, x):
            x = F.relu(self.proj_in(x))
            for block in self.blocks:
                x = x + block(x)
            return self.proj_out(self.out_norm(x))

    A_PARAM, B_PARAM, CORR_WEIGHT = 0.1, 0.8951, 50.0
    loss_fn = nn.BCELoss()
    rng = np.random.RandomState(42)
    pos_per_batch = int(BS * 0.2)

    def make_batch():
        """Generate one training batch (edges + negatives)."""
        idx = rng.randint(0, len(pos_src), pos_per_batch)
        n_neg = BS - pos_per_batch
        all_src = np.concatenate([pos_src[idx], rng.randint(0, N_SAMPLES, n_neg).astype(np.int32)])
        all_dst = np.concatenate([pos_dst[idx], rng.randint(0, N_SAMPLES, n_neg).astype(np.int32)])
        targets = np.concatenate([pos_wt[idx], np.zeros(n_neg, dtype=np.float32)])
        return all_src, all_dst, targets

    def train_step(model, optimizer, src_values, dst_values, targets_t):
        """One training step — same for all methods."""
        optimizer.zero_grad(set_to_none=True)
        src_emb = model(src_values)
        dst_emb = model(dst_values)
        dists = torch.norm(src_emb - dst_emb, dim=1, p=2*B_PARAM)
        qs = torch.clamp(torch.pow(1 + A_PARAM * dists, -1), 1e-7, 1-1e-7)
        umap_loss = loss_fn(qs, targets_t)
        hd = torch.norm(src_values - dst_values, dim=1)
        ld = torch.norm(src_emb - dst_emb, dim=1)
        corr_loss = compute_correlation_loss(torch.log1p(hd), torch.log1p(ld))
        loss = umap_loss + CORR_WEIGHT * corr_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    results = {}

    # ════════════════════════════════════════════
    # Method 1: Baseline (CPU tensor, .to(device) per batch)
    # ════════════════════════════════════════════
    logging.info("\n=== Method 1: BASELINE (CPU tensor, .to() per batch) ===")
    X_cpu = torch.tensor(X_np, dtype=torch.float32)
    model = UMAPNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    model.train()
    torch.cuda.reset_peak_memory_stats()

    steps = 0
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        all_src, all_dst, targets = make_batch()
        src_idx = torch.from_numpy(all_src.astype(np.int64))
        dst_idx = torch.from_numpy(all_dst.astype(np.int64))
        src_values = X_cpu[src_idx].to(device)
        dst_values = X_cpu[dst_idx].to(device)
        targets_t = torch.from_numpy(targets).to(device)
        train_step(model, optimizer, src_values, dst_values, targets_t)
        steps += 1

    elapsed = time.time() - t0
    sps_baseline = steps * BS / elapsed
    mem_baseline = torch.cuda.max_memory_allocated() / 1e9
    logging.info(f"  {sps_baseline:,.0f} samp/s, {steps} steps, {mem_baseline:.2f} GB GPU")
    results["baseline_sps"] = sps_baseline
    results["baseline_mem"] = mem_baseline
    del X_cpu

    # ════════════════════════════════════════════
    # Method 2: Pinned memory
    # ════════════════════════════════════════════
    logging.info("\n=== Method 2: PINNED MEMORY ===")
    X_pinned = torch.tensor(X_np, dtype=torch.float32).pin_memory()
    model = UMAPNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    model.train()
    torch.cuda.reset_peak_memory_stats()

    steps = 0
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        all_src, all_dst, targets = make_batch()
        src_idx = torch.from_numpy(all_src.astype(np.int64))
        dst_idx = torch.from_numpy(all_dst.astype(np.int64))
        src_values = X_pinned[src_idx].to(device, non_blocking=True)
        dst_values = X_pinned[dst_idx].to(device, non_blocking=True)
        targets_t = torch.from_numpy(targets).to(device, non_blocking=True)
        train_step(model, optimizer, src_values, dst_values, targets_t)
        steps += 1

    elapsed = time.time() - t0
    sps_pinned = steps * BS / elapsed
    mem_pinned = torch.cuda.max_memory_allocated() / 1e9
    logging.info(f"  {sps_pinned:,.0f} samp/s, {steps} steps, {mem_pinned:.2f} GB GPU")
    results["pinned_sps"] = sps_pinned
    results["pinned_mem"] = mem_pinned
    del X_pinned

    # ════════════════════════════════════════════
    # Method 3: Async prefetch (prepare next batch on CUDA stream)
    # ════════════════════════════════════════════
    logging.info("\n=== Method 3: ASYNC PREFETCH (CUDA streams) ===")
    X_pinned2 = torch.tensor(X_np, dtype=torch.float32).pin_memory()
    model = UMAPNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    model.train()
    torch.cuda.reset_peak_memory_stats()

    transfer_stream = torch.cuda.Stream()

    # Pre-fetch first batch
    all_src, all_dst, targets = make_batch()
    src_idx = torch.from_numpy(all_src.astype(np.int64))
    dst_idx = torch.from_numpy(all_dst.astype(np.int64))
    with torch.cuda.stream(transfer_stream):
        next_src = X_pinned2[src_idx].to(device, non_blocking=True)
        next_dst = X_pinned2[dst_idx].to(device, non_blocking=True)
        next_tgt = torch.from_numpy(targets).to(device, non_blocking=True)

    steps = 0
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        # Wait for current batch transfer
        torch.cuda.current_stream().wait_stream(transfer_stream)
        src_values = next_src
        dst_values = next_dst
        targets_t = next_tgt

        # Start prefetching next batch while training
        all_src, all_dst, targets = make_batch()
        src_idx = torch.from_numpy(all_src.astype(np.int64))
        dst_idx = torch.from_numpy(all_dst.astype(np.int64))
        with torch.cuda.stream(transfer_stream):
            next_src = X_pinned2[src_idx].to(device, non_blocking=True)
            next_dst = X_pinned2[dst_idx].to(device, non_blocking=True)
            next_tgt = torch.from_numpy(targets).to(device, non_blocking=True)

        # Train on current batch
        train_step(model, optimizer, src_values, dst_values, targets_t)
        steps += 1

    elapsed = time.time() - t0
    sps_prefetch = steps * BS / elapsed
    mem_prefetch = torch.cuda.max_memory_allocated() / 1e9
    logging.info(f"  {sps_prefetch:,.0f} samp/s, {steps} steps, {mem_prefetch:.2f} GB GPU")
    results["prefetch_sps"] = sps_prefetch
    results["prefetch_mem"] = mem_prefetch
    del X_pinned2

    # ════════════════════════════════════════════
    # Method 4: GPU-resident data (5M fits on A10G)
    # ════════════════════════════════════════════
    logging.info("\n=== Method 4: GPU-RESIDENT DATA ===")
    X_gpu = torch.tensor(X_np, dtype=torch.float32, device=device)
    model = UMAPNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    model.train()
    torch.cuda.reset_peak_memory_stats()

    steps = 0
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        all_src, all_dst, targets = make_batch()
        src_idx = torch.from_numpy(all_src.astype(np.int64)).to(device)
        dst_idx = torch.from_numpy(all_dst.astype(np.int64)).to(device)
        src_values = X_gpu[src_idx]
        dst_values = X_gpu[dst_idx]
        targets_t = torch.from_numpy(targets).to(device)
        train_step(model, optimizer, src_values, dst_values, targets_t)
        steps += 1

    elapsed = time.time() - t0
    sps_gpu = steps * BS / elapsed
    mem_gpu = torch.cuda.max_memory_allocated() / 1e9
    logging.info(f"  {sps_gpu:,.0f} samp/s, {steps} steps, {mem_gpu:.2f} GB GPU")
    results["gpu_resident_sps"] = sps_gpu
    results["gpu_resident_mem"] = mem_gpu
    del X_gpu

    # ════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════
    logging.info(f"\n{'='*60}")
    logging.info(f"  THROUGHPUT OPTIMIZATION RESULTS (5M samples, A10G)")
    logging.info(f"{'='*60}")
    logging.info(f"  {'Method':<30} {'Samp/s':>10} {'Speedup':>8} {'GPU Mem':>8}")
    logging.info(f"  {'-'*58}")
    for name, key in [("Baseline (CPU .to())", "baseline"),
                       ("Pinned memory", "pinned"),
                       ("Async prefetch", "prefetch"),
                       ("GPU-resident", "gpu_resident")]:
        sps = results[f"{key}_sps"]
        mem = results[f"{key}_mem"]
        speedup = sps / results["baseline_sps"]
        logging.info(f"  {name:<30} {sps:>9,.0f} {speedup:>7.1f}x {mem:>7.2f}GB")

    # Project 15M costs
    logging.info(f"\n  15M × 10 epoch projections:")
    for name, key in [("Baseline", "baseline"), ("Pinned", "pinned"),
                       ("Prefetch", "prefetch"), ("GPU-resident*", "gpu_resident")]:
        sps = results[f"{key}_sps"]
        time_15m = 15_000_000 * 10 / sps
        cost = time_15m * 1.10 / 3600
        logging.info(f"    {name:<20} {time_15m/60:>6.1f} min  ${cost:.2f}")
    logging.info(f"  * GPU-resident only works for ≤5M on A10G (24GB)")

    return results


@app.local_entrypoint()
def run():
    results = bench_all.remote()
    print(f"\nResults: {results}")
