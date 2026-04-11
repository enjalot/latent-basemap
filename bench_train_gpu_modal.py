"""
Training throughput benchmark across GPU types.
Tests pure training speed (no k-NN) with synthetic on-the-fly edges.

Usage:
  modal run bench_train_gpu_modal.py
"""
import time
import logging
import numpy as np
from modal import App, Image, Secret, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2", "numpy==1.26.3", "scipy", "tqdm",
    )
    .add_local_python_source("basemap")
)

with st_image.imports():
    import torch as _torch

app = App("bench-train-gpu")

VOLUMES = {
    "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
}


def _do_train_bench(gpu_name, n_samples=1_000_000, d=384, hidden_dim=512,
                    n_layers=3, batch_size=4096, seconds=120):
    """Pure training throughput test — no k-NN, synthetic edges."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from basemap.data_loader import MemmapArrayConcatenator
    from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

    device = "cuda"
    results = {"gpu": gpu_name, "n_samples": n_samples, "hidden_dim": hidden_dim,
               "n_layers": n_layers, "batch_size": batch_size}

    # Load real data
    t0 = time.time()
    loader = MemmapArrayConcatenator(
        ["/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train"], d)
    actual_n = min(n_samples, loader.shape[0])
    X = np.ascontiguousarray(np.asarray(loader[:actual_n]).astype(np.float32))
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    load_time = time.time() - t0
    logging.info(f"Data: {X.shape} on {device} in {load_time:.1f}s")

    gpu_mem_data = torch.cuda.max_memory_allocated() / 1e9
    logging.info(f"Data GPU mem: {gpu_mem_data:.2f} GB")

    # Model
    class UMAPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = nn.Linear(d, hidden_dim)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(n_layers)
            ])
            self.out_norm = nn.LayerNorm(hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, 2)
        def forward(self, x):
            x = F.relu(self.proj_in(x))
            for block in self.blocks:
                x = x + block(x)
            return self.proj_out(self.out_norm(x))

    model = UMAPNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.BCELoss()
    logging.info(f"Model: {n_params:,} params")

    # Also test larger model
    for test_hidden in [hidden_dim, 1024, 2048]:
        if test_hidden != hidden_dim:
            model2 = type(model)()  # can't easily reconstruct, skip
            continue

        torch.cuda.reset_peak_memory_stats()
        model.train()

        # Training loop — synthetic on-the-fly edges
        rng = np.random.RandomState(42)
        total_samples = 0
        total_steps = 0
        A_PARAM, B_PARAM = 0.1, 0.8951

        t0 = time.time()
        while time.time() - t0 < seconds:
            # Random positive + negative edges (on-the-fly)
            n_pos = int(batch_size * 0.2)
            n_neg = batch_size - n_pos

            # Positive: random pairs from data
            pos_src = rng.randint(0, actual_n, n_pos)
            pos_dst = rng.randint(0, actual_n, n_pos)  # approximate neighbors
            # Negative: random pairs
            neg_src = rng.randint(0, actual_n, n_neg)
            neg_dst = rng.randint(0, actual_n, n_neg)

            src_idx = np.concatenate([pos_src, neg_src])
            dst_idx = np.concatenate([pos_dst, neg_dst])
            targets = np.concatenate([np.ones(n_pos, dtype=np.float32) / 15,
                                       np.zeros(n_neg, dtype=np.float32)])

            src_idx_t = torch.from_numpy(src_idx).to(device)
            dst_idx_t = torch.from_numpy(dst_idx).to(device)
            targets_t = torch.from_numpy(targets).to(device)

            src_values = X_tensor[src_idx_t]
            dst_values = X_tensor[dst_idx_t]

            optimizer.zero_grad(set_to_none=True)

            src_emb = model(src_values)
            dst_emb = model(dst_values)

            dists = torch.norm(src_emb - dst_emb, dim=1, p=2*B_PARAM)
            qs = torch.pow(1 + A_PARAM * dists, -1)
            qs = torch.clamp(qs, min=1e-7, max=1 - 1e-7)
            umap_loss = loss_fn(qs, targets_t)

            hd = torch.norm(src_values - dst_values, dim=1)
            ld = torch.norm(src_emb - dst_emb, dim=1)
            corr_loss = compute_correlation_loss(torch.log1p(hd), torch.log1p(ld))

            loss = umap_loss + 50.0 * corr_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_samples += batch_size
            total_steps += 1

        elapsed = time.time() - t0
        sps = total_samples / elapsed
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        rate = {"T4": 0.59, "A10G": 1.10, "L4": 0.73, "A100-40GB": 2.10,
                "A100-80GB": 2.50, "H100": 3.95, "L40S": 1.70}.get(gpu_name, 1.10)
        epoch_150m = 150_000_000 / sps
        cost_per_epoch = epoch_150m * rate / 3600
        cost_10ep = cost_per_epoch * 10

        logging.info(f"\n{'='*50}")
        logging.info(f"  {gpu_name} — hidden={hidden_dim}, bs={batch_size}")
        logging.info(f"{'='*50}")
        logging.info(f"  Throughput:     {sps:,.0f} samp/s")
        logging.info(f"  Steps:          {total_steps:,} in {elapsed:.0f}s")
        logging.info(f"  Peak GPU mem:   {peak_mem:.2f} GB")
        logging.info(f"  --- 150M projection ---")
        logging.info(f"  Epoch time:     {epoch_150m/3600:.1f} hr")
        logging.info(f"  Cost/epoch:     ${cost_per_epoch:.2f}")
        logging.info(f"  10 epochs:      ${cost_10ep:.2f}")
        logging.info(f"{'='*50}")

        results["sps"] = sps
        results["peak_mem_gb"] = peak_mem
        results["n_params"] = n_params
        results["epoch_150m_hr"] = epoch_150m / 3600
        results["cost_per_epoch"] = cost_per_epoch
        results["cost_10ep"] = cost_10ep
        results["rate_per_hr"] = rate

    return results


@app.function(gpu="T4", timeout=300, image=st_image, volumes=VOLUMES)
def bench_t4():
    return _do_train_bench("T4", seconds=60)

@app.function(gpu="A10G", timeout=300, image=st_image, volumes=VOLUMES)
def bench_a10g():
    return _do_train_bench("A10G", seconds=60)

@app.function(gpu="A100-40GB", timeout=300, image=st_image, volumes=VOLUMES)
def bench_a100():
    return _do_train_bench("A100-40GB", seconds=60)

@app.function(gpu="H100", timeout=300, image=st_image, volumes=VOLUMES)
def bench_h100():
    return _do_train_bench("H100", seconds=60)

@app.function(gpu="L40S", timeout=300, image=st_image, volumes=VOLUMES)
def bench_l40s():
    return _do_train_bench("L40S", seconds=60)


@app.local_entrypoint()
def run():
    # Spawn all in parallel
    handles = [
        ("T4", bench_t4.spawn()),
        ("A10G", bench_a10g.spawn()),
        ("A100-40GB", bench_a100.spawn()),
        ("H100", bench_h100.spawn()),
        ("L40S", bench_l40s.spawn()),
    ]

    results = []
    for name, h in handles:
        try:
            r = h.get()
            results.append(r)
            print(f"  {name}: {r['sps']:,.0f} samp/s, ${r['cost_10ep']:.2f}/10ep")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    print(f"\n{'='*70}")
    print(f"  TRAINING THROUGHPUT: 150M, 10 epochs, hidden=512, bs=4096")
    print(f"{'='*70}")
    print(f"  {'GPU':<12} {'$/hr':>6} {'Samp/s':>10} {'Epoch':>8} {'10ep Cost':>10} {'GPU Mem':>8}")
    print(f"  {'-'*58}")
    for r in sorted(results, key=lambda x: x.get('cost_10ep', 999)):
        print(f"  {r['gpu']:<12} ${r['rate_per_hr']:.2f}  {r['sps']:>9,.0f}  {r['epoch_150m_hr']:>6.1f}hr  ${r['cost_10ep']:>8.2f}  {r['peak_mem_gb']:>6.1f}GB")
