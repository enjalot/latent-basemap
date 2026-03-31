"""
Scale benchmarks on Modal: measure k-NN speed, training throughput, and memory
at 100K and 1M+ with FAISS + on-the-fly negatives.

Usage:
  modal run bench_scale_modal.py                    # default: 100k, 2 epochs
  modal run bench_scale_modal.py --n-samples 1000000 --n-epochs 1
  modal run bench_scale_modal.py --n-samples 100000 --n-neighbors 100
"""
import time
import logging
import numpy as np
from modal import App, Image, Secret, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2", "numpy==1.26.3", "scipy", "scikit-learn",
        "tqdm", "faiss-gpu-cu12", "wandb",
    )
    .add_local_python_source("basemap")
)

with st_image.imports():
    import torch as _torch

app = App("bench-pumap-scale")

VOLUMES = {
    "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
}
SECRETS = [Secret.from_name("enjalot-wandb-secret")]


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 2,
    scaledown_window=300,
    image=st_image,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def bench(n_samples: int = 100_000, n_neighbors: int = 15, n_epochs: int = 2,
          batch_size: int = 4096, hidden_dim: int = 512):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from basemap.data_loader import MemmapArrayConcatenator
    from basemap.pumap.parametric_umap.utils.graph import compute_knn_graph_fast
    from basemap.pumap.parametric_umap.datasets.edge_dataset import EdgeDataset
    from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

    device = "cuda"
    results = {"n_samples": n_samples, "n_neighbors": n_neighbors,
               "n_epochs": n_epochs, "batch_size": batch_size, "gpu": "a10g"}

    # ── Load data ──
    t0 = time.time()
    # Use the 384d MiniLM embeddings (already on Modal from SAE training)
    datasets = ["/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train"]
    loader = MemmapArrayConcatenator(datasets, 384)
    total_available = loader.shape[0]
    logging.info(f"Total available samples: {total_available:,}")

    actual_n = min(n_samples, total_available)
    X = np.asarray(loader[:actual_n]).astype(np.float32)
    load_time = time.time() - t0
    logging.info(f"Data loaded: {X.shape} in {load_time:.1f}s")
    results["data_load_s"] = load_time
    results["actual_n"] = actual_n
    results["input_dim"] = X.shape[1]

    # ── k-NN (fast, no sigma) ──
    t0 = time.time()
    P_sym = compute_knn_graph_fast(X, k=n_neighbors)
    knn_time = time.time() - t0
    logging.info(f"k-NN (fast): {knn_time:.1f}s ({P_sym.nnz:,} edges)")
    results["knn_time_s"] = knn_time
    results["n_edges"] = P_sym.nnz

    # ── Build edge dataset with on-the-fly negatives ──
    t0 = time.time()
    ed = EdgeDataset(P_sym)
    # Use on-the-fly loader — no precomputed negatives!
    otf_loader = ed.get_on_the_fly_loader(
        n_nodes=actual_n, batch_size=batch_size,
        pos_ratio=0.2, shuffle=True
    )
    edge_setup_time = time.time() - t0
    logging.info(f"Edge setup (on-the-fly): {edge_setup_time:.1f}s, {len(otf_loader)} batches/epoch")
    results["edge_setup_s"] = edge_setup_time
    results["batches_per_epoch"] = len(otf_loader)

    # ── Model ──
    class UMAPNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim=2, n_layers=3):
            super().__init__()
            self.proj_in = nn.Linear(input_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(n_layers)
            ])
            self.out_norm = nn.LayerNorm(hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            x = F.relu(self.proj_in(x))
            for block in self.blocks:
                x = x + block(x)
            x = self.out_norm(x)
            return self.proj_out(x)

    model = UMAPNet(X.shape[1], hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    results["n_params"] = n_params
    logging.info(f"Model: {n_params:,} params")

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.BCELoss()
    A_PARAM, B_PARAM = 0.1, 0.8951

    # ── Training ──
    logging.info(f"Training {n_epochs} epochs...")
    model.train()
    train_start = time.time()
    total_steps = 0

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for edges, labels in otf_loader:
            edges_arr = np.array(edges, dtype=np.int64)
            src_idx = torch.from_numpy(edges_arr[:, 0]).to(device)
            dst_idx = torch.from_numpy(edges_arr[:, 1]).to(device)
            targets = torch.from_numpy(labels.astype(np.float32)).to(device)

            src_values = X_tensor[src_idx]
            dst_values = X_tensor[dst_idx]

            optimizer.zero_grad(set_to_none=True)

            src_emb = model(src_values)
            dst_emb = model(dst_values)

            dists = torch.norm(src_emb - dst_emb, dim=1, p=2*B_PARAM)
            qs = torch.pow(1 + A_PARAM * dists, -1)
            qs = torch.clamp(qs, min=1e-7, max=1 - 1e-7)
            umap_loss = loss_fn(qs, targets)

            hd = torch.norm(src_values - dst_values, dim=1)
            ld = torch.norm(src_emb - dst_emb, dim=1)
            corr_loss = compute_correlation_loss(torch.log1p(hd), torch.log1p(ld))

            loss = umap_loss + 50.0 * corr_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            total_steps += 1

        elapsed = time.time() - train_start
        logging.info(f"Epoch {epoch}/{n_epochs}: loss={epoch_loss/n_batches:.4f} [{elapsed:.0f}s]")

    train_time = time.time() - train_start
    samples_per_sec = actual_n * n_epochs / train_time
    results["train_time_s"] = train_time
    results["samples_per_sec"] = samples_per_sec
    results["total_steps"] = total_steps

    # ── Memory ──
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        results["peak_gpu_mem_gb"] = peak_mem
        logging.info(f"Peak GPU memory: {peak_mem:.2f} GB")

    # ── Cost ──
    total_time = load_time + knn_time + edge_setup_time + train_time
    cost = total_time * 1.10 / 3600
    results["total_time_s"] = total_time
    results["cost_usd"] = cost

    # ── Summary ──
    logging.info(f"\n{'='*60}")
    logging.info(f"  SCALE BENCHMARK: {actual_n:,} samples")
    logging.info(f"{'='*60}")
    logging.info(f"  k-NN (FAISS):     {knn_time:.1f}s")
    logging.info(f"  Edge setup:       {edge_setup_time:.1f}s")
    logging.info(f"  Training ({n_epochs}ep): {train_time:.1f}s ({samples_per_sec:.0f} samp/s)")
    logging.info(f"  Peak GPU mem:     {results.get('peak_gpu_mem_gb', 0):.2f} GB")
    logging.info(f"  Total:            {total_time:.1f}s")
    logging.info(f"  Cost:             ${cost:.4f}")
    logging.info(f"{'='*60}")

    return results


@app.local_entrypoint()
def run(n_samples: int = 100_000, n_neighbors: int = 15, n_epochs: int = 2,
        batch_size: int = 4096, hidden_dim: int = 512):
    print(f"Benchmarking: {n_samples:,} samples, nn={n_neighbors}, {n_epochs} epochs, bs={batch_size}")
    results = bench.remote(n_samples, n_neighbors, n_epochs, batch_size, hidden_dim)
    print(f"\nResults: {results}")
