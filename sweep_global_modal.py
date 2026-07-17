"""
Global distance optimization sweep — 10 experiments at 15M on L40S.
Each experiment tests a different approach to improving distance correlation.

Usage:
  modal run sweep_global_modal.py
"""
from basemap.round0005_retirement import refuse_retired_launcher

refuse_retired_launcher("sweep_global_modal.py")

import time
import logging
import numpy as np
from modal import App, Image, Secret, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.1.2", "numpy==1.26.3", "scipy", "scikit-learn",
                 "tqdm", "wandb")
    .add_local_python_source("basemap")
)

with st_image.imports():
    import torch as _torch

app = App("sweep-global")

VOLUMES = {
    "/embeddings": Volume.from_name("embeddings"),
    "/checkpoints": Volume.from_name("checkpoints"),
}
SECRETS = [Secret.from_name("enjalot-wandb-secret")]

DATASETS = {
    "fineweb": "/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "redpajama": "/embeddings/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "pile": "/embeddings/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
}
D_IN = 384
N_PER_DS = 5_000_000
EDGES_PATH = "/checkpoints/pumap/edges_15m_k15.npz"


EXPERIMENTS = [
    {
        "name": "baseline-edge-only",
        "desc": "Baseline: UMAP BCE + log corr on edges only (no global)",
        "global_weight": 0.0,
        "umap_weight": 1.0,
        "edge_corr_weight": 50.0,
        "global_loss_type": "none",
    },
    {
        "name": "global-random-pairs",
        "desc": "Add global random-pair distance MSE loss",
        "global_weight": 10.0,
        "umap_weight": 1.0,
        "edge_corr_weight": 50.0,
        "global_loss_type": "mse",
    },
    {
        "name": "global-random-log-corr",
        "desc": "Global random-pair log-space Pearson correlation",
        "global_weight": 50.0,
        "umap_weight": 1.0,
        "edge_corr_weight": 50.0,
        "global_loss_type": "log_corr",
    },
    {
        "name": "global-rank-corr",
        "desc": "Global random-pair soft rank correlation",
        "global_weight": 5.0,
        "umap_weight": 1.0,
        "edge_corr_weight": 0.0,
        "global_loss_type": "rank",
    },
    {
        "name": "global-only-no-umap",
        "desc": "Pure global: no UMAP loss, only distance matching",
        "global_weight": 100.0,
        "umap_weight": 0.0,
        "edge_corr_weight": 0.0,
        "global_loss_type": "log_corr",
    },
    {
        "name": "global-heavy-50-50",
        "desc": "50/50 balance: edge UMAP + global distance",
        "global_weight": 50.0,
        "umap_weight": 1.0,
        "edge_corr_weight": 0.0,
        "global_loss_type": "log_corr",
    },
    {
        "name": "global-mse-normalized",
        "desc": "Global MSE on normalized distances (d/max_d)",
        "global_weight": 50.0,
        "umap_weight": 1.0,
        "edge_corr_weight": 25.0,
        "global_loss_type": "mse_normalized",
    },
    {
        "name": "global-large-batch-8k",
        "desc": "Larger batch (8192) for better global distance estimates",
        "global_weight": 50.0,
        "umap_weight": 1.0,
        "edge_corr_weight": 50.0,
        "global_loss_type": "log_corr",
        "batch_size": 8192,
    },
    {
        "name": "hierarchical-2phase",
        "desc": "Phase 1 (5ep): global only. Phase 2 (5ep): edge UMAP + global",
        "global_weight": 100.0,
        "umap_weight": 0.0,
        "edge_corr_weight": 0.0,
        "global_loss_type": "log_corr",
        "hierarchical": True,
    },
    {
        "name": "global-cosine-dist",
        "desc": "Global correlation on cosine distance (not L2)",
        "global_weight": 50.0,
        "umap_weight": 1.0,
        "edge_corr_weight": 25.0,
        "global_loss_type": "cosine_corr",
    },
]


@app.function(
    gpu="L40S", timeout=60*60*2, scaledown_window=300,
    image=st_image, volumes=VOLUMES, secrets=SECRETS,
)
def run_experiment(exp_config: dict, n_epochs: int = 10):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from sklearn.neighbors import NearestNeighbors
    from basemap.data_loader import MemmapArrayConcatenator
    from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

    device = "cuda"
    name = exp_config["name"]
    batch_size = exp_config.get("batch_size", 4096)
    hierarchical = exp_config.get("hierarchical", False)

    # ── Load data ──
    t0 = time.time()
    all_X = []
    for ds_name, path in DATASETS.items():
        loader = MemmapArrayConcatenator([path], D_IN)
        all_X.append(np.asarray(loader[:N_PER_DS]).astype(np.float32))
    X = np.ascontiguousarray(np.concatenate(all_X))
    del all_X
    actual_n = len(X)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    logging.info(f"[{name}] Data: {X.shape} on GPU in {time.time()-t0:.0f}s")

    # ── Load edges ──
    edges = np.load(EDGES_PATH)
    mask = (edges["sources"] < actual_n) & (edges["targets"] < actual_n)
    pos_src = edges["sources"][mask]
    pos_dst = edges["targets"][mask]
    pos_wt = edges["weights"][mask]
    n_pos = len(pos_src)
    logging.info(f"[{name}] Edges: {n_pos:,}")

    # ── Model ──
    class UMAPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = nn.Linear(D_IN, 512)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512), nn.ReLU())
                for _ in range(3)
            ])
            self.out_norm = nn.LayerNorm(512)
            self.proj_out = nn.Linear(512, 2)
        def forward(self, x):
            x = F.relu(self.proj_in(x))
            for block in self.blocks:
                x = x + block(x)
            return self.proj_out(self.out_norm(x))

    model = UMAPNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.BCELoss()
    A_PARAM, B_PARAM = 0.1, 0.8951
    rng = np.random.RandomState(42)
    pos_per_batch = int(batch_size * 0.2)
    batches_per_epoch = n_pos // pos_per_batch

    # Cosine schedule
    total_steps = batches_per_epoch * n_epochs
    warmup = min(500, total_steps // 10)
    def lr_lambda(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + np.cos(np.pi * min((step - warmup) / max(1, total_steps - warmup), 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Soft rank helper
    def soft_rank(x, temp=0.1):
        return torch.sigmoid((x.unsqueeze(1) - x.unsqueeze(0)) / temp).sum(dim=1)

    # ── Wandb ──
    try:
        import wandb
        wandb.init(project="pumap-global-sweep", name=name,
                   config={**exp_config, "n_samples": actual_n, "n_epochs": n_epochs})
    except:
        wandb = None

    # ── Training ──
    model.train()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        # Hierarchical: switch loss at midpoint
        if hierarchical and epoch > n_epochs // 2:
            cur_umap_w = 1.0
            cur_global_w = 50.0
            cur_edge_corr_w = 25.0
        else:
            cur_umap_w = exp_config["umap_weight"]
            cur_global_w = exp_config["global_weight"]
            cur_edge_corr_w = exp_config["edge_corr_weight"]

        perm = rng.permutation(n_pos)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n_pos, pos_per_batch):
            idx = perm[batch_start:batch_start + pos_per_batch]
            n_p = len(idx)
            n_neg = batch_size - n_p

            # Edge batch
            all_src = np.concatenate([pos_src[idx], rng.randint(0, actual_n, n_neg).astype(np.int32)])
            all_dst = np.concatenate([pos_dst[idx], rng.randint(0, actual_n, n_neg).astype(np.int32)])
            targets = np.concatenate([pos_wt[idx], np.zeros(n_neg, dtype=np.float32)])

            src_idx = torch.from_numpy(all_src.astype(np.int64)).to(device)
            dst_idx = torch.from_numpy(all_dst.astype(np.int64)).to(device)
            targets_t = torch.from_numpy(targets).to(device)

            src_values = X_tensor[src_idx]
            dst_values = X_tensor[dst_idx]

            optimizer.zero_grad(set_to_none=True)
            src_emb = model(src_values)
            dst_emb = model(dst_values)

            loss = torch.tensor(0.0, device=device)

            # UMAP BCE loss on edges
            if cur_umap_w > 0:
                dists = torch.norm(src_emb - dst_emb, dim=1, p=2*B_PARAM)
                qs = torch.clamp(torch.pow(1 + A_PARAM * dists, -1), 1e-7, 1-1e-7)
                loss = loss + cur_umap_w * loss_fn(qs, targets_t)

            # Edge-based distance correlation
            if cur_edge_corr_w > 0:
                hd = torch.norm(src_values - dst_values, dim=1)
                ld = torch.norm(src_emb - dst_emb, dim=1)
                loss = loss + cur_edge_corr_w * compute_correlation_loss(torch.log1p(hd), torch.log1p(ld))

            # ── Global distance loss (random pairs from full dataset) ──
            if cur_global_w > 0:
                n_global = min(1024, batch_size)
                g_src = torch.randint(0, actual_n, (n_global,), device=device)
                g_dst = torch.randint(0, actual_n, (n_global,), device=device)
                g_src_val = X_tensor[g_src]
                g_dst_val = X_tensor[g_dst]
                g_src_emb = model(g_src_val)
                g_dst_emb = model(g_dst_val)

                g_hd = torch.norm(g_src_val - g_dst_val, dim=1)
                g_ld = torch.norm(g_src_emb - g_dst_emb, dim=1)

                glt = exp_config["global_loss_type"]
                if glt == "mse":
                    loss = loss + cur_global_w * F.mse_loss(g_ld, g_hd / g_hd.max().detach() * g_ld.max().detach())
                elif glt == "mse_normalized":
                    loss = loss + cur_global_w * F.mse_loss(g_ld / (g_ld.max().detach() + 1e-8), g_hd / (g_hd.max().detach() + 1e-8))
                elif glt == "log_corr":
                    loss = loss + cur_global_w * compute_correlation_loss(torch.log1p(g_hd), torch.log1p(g_ld))
                elif glt == "rank":
                    n_sub = min(512, n_global)
                    loss = loss + cur_global_w * compute_correlation_loss(soft_rank(g_hd[:n_sub]), soft_rank(g_ld[:n_sub]))
                elif glt == "cosine_corr":
                    g_cos_hd = 1 - F.cosine_similarity(g_src_val, g_dst_val)
                    g_cos_ld = torch.norm(g_src_emb - g_dst_emb, dim=1)
                    loss = loss + cur_global_w * compute_correlation_loss(torch.log1p(g_cos_hd), torch.log1p(g_cos_ld))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if wandb and global_step % 200 == 0:
                wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=global_step)

        elapsed = time.time() - train_start
        sps = actual_n * epoch / elapsed
        logging.info(f"[{name}] Epoch {epoch}/{n_epochs}: loss={epoch_loss/n_batches:.4f} [{elapsed:.0f}s, {sps:.0f} samp/s]")

    train_time = time.time() - train_start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # ── Evaluate (batched to avoid OOM) ──
    model.eval()
    Z_parts = []
    with torch.no_grad():
        for i in range(0, actual_n, 100_000):
            Z_parts.append(model(X_tensor[i:i+100_000]).cpu().numpy())
    Z = np.concatenate(Z_parts)

    # Sampled distance correlation (global metric)
    rng2 = np.random.RandomState(42)
    n_eval = min(10000, actual_n)
    eval_idx = rng2.choice(actual_n, n_eval, replace=False)
    X_eval, Z_eval = X[eval_idx], Z[eval_idx]

    n_pairs = 10000
    i = rng2.randint(0, n_eval, n_pairs)
    j = rng2.randint(0, n_eval, n_pairs)
    mask_ij = i != j; i, j = i[mask_ij], j[mask_ij]
    dh = np.linalg.norm(X_eval[i] - X_eval[j], axis=1)
    dl = np.linalg.norm(Z_eval[i] - Z_eval[j], axis=1)
    dist_corr = float(np.corrcoef(dh, dl)[0, 1])

    # KNN preservation
    nn_h = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(X_eval)
    nn_l = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(Z_eval)
    _, idx_h = nn_h.kneighbors(X_eval)
    _, idx_l = nn_l.kneighbors(Z_eval)
    knn_10 = sum(len(set(idx_h[ii,1:]) & set(idx_l[ii,1:])) for ii in range(n_eval)) / (n_eval * 10)

    cost = train_time * 1.70 / 3600

    result = {
        "name": name, "desc": exp_config["desc"],
        "dist_corr": dist_corr, "knn_10": knn_10,
        "train_time_s": train_time, "sps": actual_n * n_epochs / train_time,
        "peak_mem_gb": peak_mem, "cost_usd": cost,
    }

    logging.info(f"\n[{name}] dist_corr={dist_corr:.4f}, knn_10={knn_10:.4f}, cost=${cost:.2f}")

    if wandb:
        wandb.log(result)
        wandb.summary.update(result)
        wandb.finish()

    return result


@app.local_entrypoint()
def run():
    # Spawn all 10 experiments in parallel
    handles = []
    for exp in EXPERIMENTS:
        logging.info(f"Spawning: {exp['name']} — {exp['desc']}")
        h = run_experiment.spawn(exp, n_epochs=10)
        handles.append((exp["name"], h))

    print(f"\nAll {len(handles)} experiments spawned. Waiting...")

    results = []
    for name, h in handles:
        try:
            r = h.get()
            results.append(r)
            print(f"  {name}: dist_corr={r['dist_corr']:.4f}, knn_10={r['knn_10']:.4f}, ${r['cost_usd']:.2f}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    # Summary table
    results.sort(key=lambda x: x.get("dist_corr", 0), reverse=True)
    print(f"\n{'='*80}")
    print(f"  GLOBAL DISTANCE SWEEP (15M × 10ep, L40S)")
    print(f"{'='*80}")
    print(f"  {'Name':<30} {'DistCorr':>9} {'KNN10':>7} {'Cost':>6} {'Samp/s':>9}")
    print(f"  {'-'*65}")
    for r in results:
        print(f"  {r['name']:<30} {r['dist_corr']:>8.4f} {r['knn_10']:>7.4f} ${r['cost_usd']:>5.2f} {r['sps']:>8,.0f}")
