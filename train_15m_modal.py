"""
Train parametric UMAP on 15M samples (5M from each dataset) using
precomputed edge list from IVF_PQ index.

Usage:
  modal run train_15m_modal.py                    # default: L40S, 10 epochs
  modal run train_15m_modal.py --gpu a10g         # cheaper GPU
  modal run train_15m_modal.py --n-epochs 50      # longer training
"""
import time
import logging
import os
import numpy as np
from modal import App, Image, Secret, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2", "numpy==1.26.3", "scipy", "scikit-learn",
        "tqdm", "wandb",
    )
    .add_local_python_source("basemap")
)

with st_image.imports():
    import torch as _torch

app = App("train-pumap-15m")

VOLUMES = {
    "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
    "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
}
SECRETS = [Secret.from_name("enjalot-wandb-secret")]

DATASETS = {
    "fineweb": "/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "redpajama": "/embeddings/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "pile": "/embeddings/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
}
D_IN = 384
N_PER_DATASET = 5_000_000
EDGES_PATH = "/checkpoints/pumap/edges_15m_k15.npz"


def _do_train(gpu_name, n_epochs=10, batch_size=4096, hidden_dim=512, lr=1e-3):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from basemap.data_loader import MemmapArrayConcatenator
    from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

    device = "cuda"
    rate = {"T4": 0.59, "A10G": 1.10, "L40S": 1.70, "A100-40GB": 2.10,
            "H100": 3.95}.get(gpu_name, 1.10)

    # ── Load embeddings (stream from memmap) ──
    t0 = time.time()
    all_X = []
    for name, path in DATASETS.items():
        loader = MemmapArrayConcatenator([path], D_IN)
        X_part = np.asarray(loader[:N_PER_DATASET]).astype(np.float32)
        all_X.append(X_part)
        logging.info(f"  {name}: {X_part.shape}")
    X = np.concatenate(all_X, axis=0)
    X = np.ascontiguousarray(X)
    actual_n = len(X)
    del all_X
    load_time = time.time() - t0
    logging.info(f"Data: {X.shape} in {load_time:.1f}s")

    # Keep data on CPU — 15M x 384 x 4B = 23GB doesn't fit on A10G (24GB)
    # Transfer per-batch to GPU instead
    X_tensor = torch.tensor(X, dtype=torch.float32)  # CPU
    logging.info(f"Data on CPU: {X_tensor.shape} ({X_tensor.nbytes/1e9:.1f} GB)")

    # ── Load edge list ──
    t0 = time.time()
    edges = np.load(EDGES_PATH)
    pos_sources = edges["sources"]
    pos_targets = edges["targets"]
    pos_weights = edges["weights"]
    n_pos_edges = len(pos_sources)
    logging.info(f"Edges loaded: {n_pos_edges:,} in {time.time()-t0:.1f}s")

    # ── Model ──
    class UMAPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = nn.Linear(D_IN, hidden_dim)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(3)
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
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine schedule
    pos_per_batch = int(batch_size * 0.2)
    batches_per_epoch = n_pos_edges // pos_per_batch
    total_steps = batches_per_epoch * n_epochs
    warmup = min(500, total_steps // 10)
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + np.cos(np.pi * min((step - warmup) / max(1, total_steps - warmup), 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.BCELoss()
    A_PARAM, B_PARAM = 0.1, 0.8951
    CORR_WEIGHT = 50.0

    logging.info(f"Model: {n_params:,} params")
    logging.info(f"Training: {n_epochs} epochs, {batches_per_epoch:,} batches/ep, bs={batch_size}")

    # ── Wandb ──
    try:
        import wandb
        wandb.init(project="pumap-15m-training", name=f"15m-{gpu_name}-{n_epochs}ep",
                   config={"n_samples": actual_n, "n_epochs": n_epochs, "batch_size": batch_size,
                           "hidden_dim": hidden_dim, "lr": lr, "gpu": gpu_name, "n_params": n_params})
    except:
        wandb = None

    # ── Training ──
    model.train()
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.RandomState(42)

    train_start = time.time()
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        # Shuffle positive edges each epoch
        perm = rng.permutation(n_pos_edges)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n_pos_edges, pos_per_batch):
            batch_end = min(batch_start + pos_per_batch, n_pos_edges)
            idx = perm[batch_start:batch_end]

            # Positive edges from precomputed list
            p_src = pos_sources[idx]
            p_dst = pos_targets[idx]
            p_wt = pos_weights[idx]
            n_pos = len(p_src)

            # Random negative edges (on-the-fly)
            n_neg = batch_size - n_pos
            neg_src = rng.randint(0, actual_n, n_neg).astype(np.int32)
            neg_dst = rng.randint(0, actual_n, n_neg).astype(np.int32)

            # Combine
            all_src = np.concatenate([p_src, neg_src])
            all_dst = np.concatenate([p_dst, neg_dst])
            targets = np.concatenate([p_wt, np.zeros(n_neg, dtype=np.float32)])

            src_idx = torch.from_numpy(all_src.astype(np.int64))
            dst_idx = torch.from_numpy(all_dst.astype(np.int64))
            targets_t = torch.from_numpy(targets).to(device)

            # Index on CPU, transfer to GPU per batch
            src_values = X_tensor[src_idx.cpu()].to(device)
            dst_values = X_tensor[dst_idx.cpu()].to(device)

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

            loss = umap_loss + CORR_WEIGHT * corr_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if wandb and global_step % 100 == 0:
                wandb.log({"batch_loss": loss.item(), "umap_loss": umap_loss.item(),
                           "corr_loss": corr_loss.item(), "lr": scheduler.get_last_lr()[0]},
                          step=global_step)

        elapsed = time.time() - train_start
        avg_loss = epoch_loss / n_batches
        sps = actual_n * epoch / elapsed
        logging.info(f"Epoch {epoch}/{n_epochs}: loss={avg_loss:.4f} [{elapsed:.0f}s, {sps:.0f} samp/s]")

        if wandb:
            wandb.log({"epoch": epoch, "epoch_loss": avg_loss, "sps": sps}, step=global_step)

    train_time = time.time() - train_start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    overall_sps = actual_n * n_epochs / train_time
    cost = train_time * rate / 3600

    # ── Save model ──
    model_path = f"/checkpoints/pumap/model_15m_{gpu_name}_{n_epochs}ep.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"hidden_dim": hidden_dim, "n_layers": 3, "d_in": D_IN, "d_out": 2},
        "n_params": n_params,
        "n_samples": actual_n,
        "n_epochs": n_epochs,
    }, model_path)
    VOLUMES["/checkpoints"].commit()

    # ── Summary ──
    logging.info(f"\n{'='*60}")
    logging.info(f"  15M TRAINING COMPLETE — {gpu_name}")
    logging.info(f"{'='*60}")
    logging.info(f"  Samples:      {actual_n:,}")
    logging.info(f"  Epochs:       {n_epochs}")
    logging.info(f"  Train time:   {train_time:.1f}s ({train_time/60:.1f} min)")
    logging.info(f"  Throughput:   {overall_sps:,.0f} samp/s")
    logging.info(f"  Peak GPU mem: {peak_mem:.2f} GB")
    logging.info(f"  Cost:         ${cost:.4f}")
    logging.info(f"  Model saved:  {model_path}")
    logging.info(f"{'='*60}")

    if wandb:
        wandb.log({"train_time_s": train_time, "overall_sps": overall_sps,
                    "peak_mem_gb": peak_mem, "cost_usd": cost})
        wandb.summary.update({"train_time_s": train_time, "overall_sps": overall_sps,
                               "peak_mem_gb": peak_mem, "cost_usd": cost})
        wandb.finish()

    return {"gpu": gpu_name, "n_samples": actual_n, "n_epochs": n_epochs,
            "train_time_s": train_time, "sps": overall_sps,
            "peak_mem_gb": peak_mem, "cost_usd": cost, "model_path": model_path}


@app.function(gpu="L40S", timeout=60*60*2, scaledown_window=300,
              image=st_image, volumes=VOLUMES, secrets=SECRETS)
def train_l40s(n_epochs=10, batch_size=4096, hidden_dim=512, lr=1e-3):
    return _do_train("L40S", n_epochs, batch_size, hidden_dim, lr)

@app.function(gpu="A10G", timeout=60*60*2, scaledown_window=300,
              image=st_image, volumes=VOLUMES, secrets=SECRETS)
def train_a10g(n_epochs=10, batch_size=4096, hidden_dim=512, lr=1e-3):
    return _do_train("A10G", n_epochs, batch_size, hidden_dim, lr)


@app.local_entrypoint()
def run(n_epochs: int = 10, batch_size: int = 4096, hidden_dim: int = 512,
        lr: float = 1e-3, gpu: str = "l40s"):
    print(f"Training 15M on {gpu}, {n_epochs} epochs")
    if gpu == "l40s":
        results = train_l40s.remote(n_epochs, batch_size, hidden_dim, lr)
    else:
        results = train_a10g.remote(n_epochs, batch_size, hidden_dim, lr)
    print(f"\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")
