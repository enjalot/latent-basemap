"""
Combined best autoresearch findings — run on Modal A10G for 50 epochs.

Combines:
  - Agent 2: rank correlation + hard positive mining + weighted BCE + LayerNorm
  - Agent 3: large batch (4096), log-space correlation, low a=0.1
  - Agent 1: high correlation weight, pos_ratio=0.2

Usage:
  modal run train_combined_modal.py
  modal run train_combined_modal.py --n-epochs 100
"""
import time
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from modal import App, Image, Secret, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2", "numpy==1.26.3", "scipy", "scikit-learn",
        "tqdm", "h5py", "pandas", "pyarrow", "wandb",
    )
    .add_local_python_source("basemap")
    .add_local_python_source("autoresearch")
)

with st_image.imports():
    import torch as _torch

app = App("train-pumap-combined")

VOLUMES = {
    "/data": Volume.from_name("pumap-data", create_if_missing=True),
}
SECRETS = [Secret.from_name("enjalot-wandb-secret")]


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 6,
    scaledown_window=300,
    image=st_image,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def train_combined(n_epochs: int = 50):
    import h5py
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold import trustworthiness as sk_trustworthiness
    from scipy.spatial import procrustes

    device = "cuda"

    # ── Load data ──
    logging.info("Loading data...")
    t0 = time.time()
    with h5py.File("/data/ls-squad/embedding-003.h5", "r") as f:
        X = f["embeddings"][:].astype(np.float32)
    Z_ref = pd.read_parquet("/data/ls-squad/umap-001.parquet")[['x', 'y']].values.astype(np.float32)
    with open("/data/precomputed/ls-squad_nn100_psym.pkl", "rb") as f:
        P_sym = pickle.load(f)
    with open("/data/precomputed/ls-squad_nn100_negatives.pkl", "rb") as f:
        neg_edges = pickle.load(f)
    logging.info(f"Data loaded in {time.time()-t0:.1f}s: X={X.shape}, edges={P_sym.nnz}")

    # ── Eval split ──
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X))
    eval_idx = perm[int(len(X) * 0.8):]

    # ── Model (Agent 2: residual MLP with LayerNorm) ──
    class UMAPNet(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=512, output_dim=2, n_layers=3):
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

    model = UMAPNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model params: {n_params:,}")

    # ── Edge dataset ──
    from basemap.pumap.parametric_umap.datasets.edge_dataset import EdgeDataset, BalancedEdgeBatchIterator
    from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

    ed = EdgeDataset(P_sym)
    ed.neg_edges = neg_edges

    # ── Hyperparams (combined best from all agents) ──
    BATCH_SIZE = 4096
    LR = 1e-3
    POS_RATIO = 0.2       # Agent 1+3: more negatives
    A_PARAM = 0.1         # Agent 2: flat kernel
    B_PARAM = 0.8951
    CORR_WEIGHT = 5.0     # Agent 2: rank correlation weight
    KNN_LOSS_WEIGHT = 1.0 # Agent 2: hard positive mining

    # Data on GPU
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    loader = BalancedEdgeBatchIterator(
        ed.pos_edges, ed.neg_edges,
        pos_weights=ed.pos_weights,
        pos_ratio=POS_RATIO,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ── Optimizer ──
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps_est = len(loader) * n_epochs
    warmup_steps = min(500, total_steps_est // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps_est - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Wandb ──
    try:
        import wandb
        wandb.init(project="pumap-combined-best", name=f"combined-{n_epochs}ep", config={
            "n_epochs": n_epochs, "batch_size": BATCH_SIZE, "lr": LR,
            "pos_ratio": POS_RATIO, "a": A_PARAM, "corr_weight": CORR_WEIGHT,
            "knn_loss_weight": KNN_LOSS_WEIGHT, "n_params": n_params,
        })
    except Exception:
        wandb = None

    # ── Soft rank helper (Agent 2) ──
    def soft_rank(x, temp=0.1):
        x_unsq = x.unsqueeze(1)
        x_ref = x.unsqueeze(0)
        return torch.sigmoid((x_unsq - x_ref) / temp).sum(dim=1)

    # ── Training ──
    logging.info(f"Training {n_epochs} epochs, {len(loader)} batches/epoch, batch={BATCH_SIZE}")
    model.train()
    train_start = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        epoch_umap = 0.0
        n_batches = 0

        for edges, labels in loader:
            edges_arr = np.array(edges, dtype=np.int64)
            src_idx = torch.from_numpy(edges_arr[:, 0]).to(device)
            dst_idx = torch.from_numpy(edges_arr[:, 1]).to(device)
            targets = torch.from_numpy(labels.astype(np.float32)).to(device)

            src_values = X_tensor[src_idx]
            dst_values = X_tensor[dst_idx]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=False):
                src_emb = model(src_values)
                dst_emb = model(dst_values)

                # UMAP similarity (flat kernel, Agent 2)
                dists = torch.norm(src_emb - dst_emb, dim=1, p=2*B_PARAM)
                qs = torch.pow(1 + A_PARAM * dists, -1)
                qs = torch.clamp(qs, min=1e-7, max=1 - 1e-7)

                # Weighted BCE (Agent 2: 2x on positives)
                bce = F.binary_cross_entropy(qs, targets, reduction='none')
                weights = torch.where(targets > 0.5, 2.0, 1.0)
                umap_loss = (weights * bce).mean()

                # Soft rank correlation (Agent 2)
                hd = torch.norm(src_values - dst_values, dim=1)
                ld = torch.norm(src_emb - dst_emb, dim=1)
                n_sub = min(1024, len(hd))
                idx_sub = torch.randperm(len(hd), device=device)[:n_sub]
                rank_hd = soft_rank(hd[idx_sub])
                rank_ld = soft_rank(ld[idx_sub])
                rank_loss = compute_correlation_loss(rank_hd, rank_ld)

                # Hard positive mining KNN loss (Agent 2)
                is_pos = targets > 0.5
                if is_pos.any() and (~is_pos).any():
                    pos_ld = ld[is_pos]
                    pos_log_probs = -pos_ld / (pos_ld.mean().detach() + 1e-8)
                    knn_loss = (F.softmax(pos_log_probs, dim=0).detach() * pos_ld).sum()
                else:
                    knn_loss = torch.tensor(0.0, device=device)

                loss = umap_loss + CORR_WEIGHT * rank_loss + KNN_LOSS_WEIGHT * knn_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_umap += umap_loss.item()
            n_batches += 1

            if wandb and n_batches % 50 == 0:
                wandb.log({"batch_loss": loss.item(), "umap_loss": umap_loss.item(),
                           "rank_loss": rank_loss.item(), "knn_loss": knn_loss.item(),
                           "lr": scheduler.get_last_lr()[0]})

        elapsed = time.time() - train_start
        avg_loss = epoch_loss / n_batches
        avg_umap = epoch_umap / n_batches
        logging.info(f"Epoch {epoch}/{n_epochs}: loss={avg_loss:.4f} umap={avg_umap:.4f} [{elapsed:.0f}s]")

        if wandb:
            wandb.log({"epoch": epoch, "epoch_loss": avg_loss, "epoch_umap": avg_umap})

    train_time = time.time() - train_start
    logging.info(f"Training complete: {train_time:.1f}s")

    # ── Evaluate ──
    logging.info("Evaluating...")
    model.eval()
    with torch.no_grad():
        Z = model(X_tensor).cpu().numpy()

    X_eval = X[eval_idx]
    Z_eval = Z[eval_idx]
    Z_ref_eval = Z_ref[eval_idx]

    results = {}
    n = len(X_eval)

    for k in [10, 25, 50]:
        nn_h = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(X_eval)
        nn_l = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(Z_eval)
        _, idx_h = nn_h.kneighbors(X_eval)
        _, idx_l = nn_l.kneighbors(Z_eval)
        preserved = sum(len(set(idx_h[i, 1:]) & set(idx_l[i, 1:])) for i in range(n))
        results[f"knn_{k}"] = preserved / (n * k)

    results["trustworthiness"] = float(sk_trustworthiness(X_eval, Z_eval, n_neighbors=10))

    # Distance correlation
    rng2 = np.random.RandomState(42)
    n_pairs = min(10000, n * (n-1) // 2)
    i, j = rng2.randint(0, n, n_pairs), rng2.randint(0, n, n_pairs)
    mask = i != j; i, j = i[mask], j[mask]
    dh = np.linalg.norm(X_eval[i] - X_eval[j], axis=1)
    dl = np.linalg.norm(Z_eval[i] - Z_eval[j], axis=1)
    results["dist_corr"] = float(np.corrcoef(dh, dl)[0, 1])

    # Reference UMAP comparison
    _, _, disp = procrustes(Z_ref_eval[:2000], Z_eval[:2000])
    results["ref_procrustes"] = float(disp)
    nn_ref = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(Z_ref_eval)
    nn_par = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(Z_eval)
    _, ir = nn_ref.kneighbors(Z_ref_eval); _, ip = nn_par.kneighbors(Z_eval)
    results["ref_knn_overlap"] = sum(len(set(ir[i,1:])&set(ip[i,1:])) for i in range(n))/(n*10)

    cost = train_time * 1.10 / 3600

    logging.info(f"\n{'='*60}")
    logging.info(f"  RESULTS: {n_epochs} epochs, {train_time:.0f}s, ${cost:.2f}")
    logging.info(f"{'='*60}")
    for k, v in results.items():
        logging.info(f"  {k}: {v:.4f}")

    if wandb:
        wandb.log(results)
        wandb.log({"train_time_s": train_time, "cost_usd": cost})
        wandb.summary.update(results)
        wandb.finish()

    return results


@app.local_entrypoint()
def run(n_epochs: int = 50):
    print(f"Launching combined best training: {n_epochs} epochs on A10G")
    results = train_combined.remote(n_epochs)
    print(f"\nResults: {results}")
