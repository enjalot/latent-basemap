"""
Train on dedicated 15M and 30M indexes (100% usable edges).
4 experiments: 2 configs × 2 scales.

Usage:
  modal run train_dedicated_modal.py
"""
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

app = App("train-dedicated")

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

EXPERIMENTS = [
    {"name": "ded-15m-more-global", "n_per_ds": 5_000_000,
     "hidden_dim": 512, "umap_weight": 5.0, "edge_corr_weight": 50.0,
     "global_weight": 30.0},
    {"name": "ded-15m-bigger-model", "n_per_ds": 5_000_000,
     "hidden_dim": 1024, "umap_weight": 5.0, "edge_corr_weight": 50.0,
     "global_weight": 10.0},
    {"name": "ded-30m-more-global", "n_per_ds": 10_000_000,
     "hidden_dim": 512, "umap_weight": 5.0, "edge_corr_weight": 50.0,
     "global_weight": 30.0},
    {"name": "ded-30m-bigger-model", "n_per_ds": 10_000_000,
     "hidden_dim": 1024, "umap_weight": 5.0, "edge_corr_weight": 50.0,
     "global_weight": 10.0},
]


@app.function(
    gpu="L40S", timeout=60*60*3, scaledown_window=300,
    image=st_image, volumes=VOLUMES, secrets=SECRETS,
)
def run_experiment(exp: dict, n_epochs: int = 10):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from sklearn.neighbors import NearestNeighbors
    from basemap.data_loader import MemmapArrayConcatenator
    from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

    device = "cuda"
    name = exp["name"]
    n_per_ds = exp["n_per_ds"]
    hidden_dim = exp["hidden_dim"]
    batch_size = 4096

    # Determine edge file
    total_n_target = n_per_ds * len(DATASETS)
    tag = f"{total_n_target // 1_000_000}m"
    edges_path = f"/checkpoints/pumap/edges_{tag}_k15.npz"

    # Load data
    all_X = []
    for ds_name, path in DATASETS.items():
        loader = MemmapArrayConcatenator([path], D_IN)
        all_X.append(np.asarray(loader[:n_per_ds]).astype(np.float32))
    X = np.ascontiguousarray(np.concatenate(all_X))
    actual_n = len(X)
    del all_X

    # GPU-resident if fits (15M=23GB, 30M=46GB, L40S=48GB)
    data_gb = actual_n * D_IN * 4 / 1e9
    props = torch.cuda.get_device_properties(0)
    gpu_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    data_on_gpu = data_gb < (gpu_gb - 6.0)

    if data_on_gpu:
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        logging.info(f"[{name}] Data on GPU: {X.shape} ({data_gb:.1f}/{gpu_gb:.0f} GB)")
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32).pin_memory()
        logging.info(f"[{name}] Data on CPU pinned: {X.shape} ({data_gb:.1f}/{gpu_gb:.0f} GB)")

    # Load edges — dedicated index, no filtering needed
    edges = np.load(edges_path)
    pos_src = edges["sources"]
    pos_dst = edges["targets"]
    pos_wt = edges["weights"]
    # Safety filter in case any indices are out of range
    mask = (pos_src < actual_n) & (pos_dst < actual_n)
    if mask.sum() < len(mask):
        logging.info(f"[{name}] Filtered {len(mask)-mask.sum():,} out-of-range edges")
        pos_src, pos_dst, pos_wt = pos_src[mask], pos_dst[mask], pos_wt[mask]
    n_pos = len(pos_src)
    logging.info(f"[{name}] Edges: {n_pos:,} (100% usable from dedicated {tag} index)")

    # Model
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
            for b in self.blocks: x = x + b(x)
            return self.proj_out(self.out_norm(x))

    model = UMAPNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.BCELoss()
    A_PARAM, B_PARAM = 0.1, 0.8951
    rng = np.random.RandomState(42)
    pos_per_batch = int(batch_size * 0.2)
    # Cap batches per epoch — with dedicated index we have many more edges,
    # but we don't need to iterate all of them each epoch. Sample instead.
    max_batches_per_epoch = 35_000  # ~same as 28M filtered edges
    batches_per_epoch = min(n_pos // pos_per_batch, max_batches_per_epoch)
    logging.info(f"[{name}] {batches_per_epoch:,} batches/ep (capped from {n_pos // pos_per_batch:,})")

    total_steps = batches_per_epoch * n_epochs
    warmup = min(500, total_steps // 10)
    def lr_lambda(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + np.cos(np.pi * min((step - warmup) / max(1, total_steps - warmup), 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    try:
        import wandb
        wandb.init(project="pumap-dedicated-idx", name=name,
                   config={**exp, "n_samples": actual_n, "n_params": n_params,
                           "n_edges": n_pos, "edges_path": edges_path})
    except:
        wandb = None

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    global_step = 0

    uw, ecw, gw = exp["umap_weight"], exp["edge_corr_weight"], exp["global_weight"]

    for epoch in range(1, n_epochs + 1):
        perm = rng.permutation(n_pos)
        epoch_loss, nb = 0.0, 0
        edges_this_epoch = min(batches_per_epoch * pos_per_batch, n_pos)

        for bs_start in range(0, edges_this_epoch, pos_per_batch):
            idx = perm[bs_start:bs_start + pos_per_batch]
            n_p = len(idx)
            n_neg = batch_size - n_p
            neg_src_np = rng.randint(0, actual_n, n_neg).astype(np.int32)
            neg_dst_np = rng.randint(0, actual_n, n_neg).astype(np.int32)

            all_src = np.concatenate([pos_src[idx], neg_src_np])
            all_dst = np.concatenate([pos_dst[idx], neg_dst_np])
            targets = np.concatenate([pos_wt[idx], np.zeros(n_neg, dtype=np.float32)])

            if data_on_gpu:
                si = torch.from_numpy(all_src.astype(np.int64)).to(device)
                di = torch.from_numpy(all_dst.astype(np.int64)).to(device)
                sv, dv = X_tensor[si], X_tensor[di]
            else:
                si = torch.from_numpy(all_src.astype(np.int64))
                di = torch.from_numpy(all_dst.astype(np.int64))
                sv = X_tensor[si].to(device, non_blocking=True)
                dv = X_tensor[di].to(device, non_blocking=True)
            tgt = torch.from_numpy(targets).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            se, de = model(sv), model(dv)
            loss = torch.tensor(0.0, device=device)

            if uw > 0:
                dists = torch.norm(se - de, dim=1, p=2*B_PARAM)
                qs = torch.clamp(torch.pow(1 + A_PARAM * dists, -1), 1e-7, 1-1e-7)
                loss = loss + uw * loss_fn(qs, tgt)
            if ecw > 0:
                hd = torch.norm(sv - dv, dim=1)
                ld = torch.norm(se - de, dim=1)
                loss = loss + ecw * compute_correlation_loss(torch.log1p(hd), torch.log1p(ld))
            if gw > 0:
                ng = 1024
                gs = torch.randint(0, actual_n, (ng,), device=device)
                gd = torch.randint(0, actual_n, (ng,), device=device)
                if data_on_gpu:
                    gsv, gdv = X_tensor[gs], X_tensor[gd]
                else:
                    gsv = X_tensor[gs.cpu()].to(device, non_blocking=True)
                    gdv = X_tensor[gd.cpu()].to(device, non_blocking=True)
                gse, gde = model(gsv), model(gdv)
                ghd = torch.norm(gsv - gdv, dim=1)
                gld = torch.norm(gse - gde, dim=1)
                loss = loss + gw * compute_correlation_loss(torch.log1p(ghd), torch.log1p(gld))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item(); nb += 1; global_step += 1

            if wandb and global_step % 200 == 0:
                wandb.log({"loss": loss.item()}, step=global_step)

        sps = actual_n * epoch / (time.time() - t0)
        logging.info(f"[{name}] Epoch {epoch}/{n_epochs}: loss={epoch_loss/nb:.4f} [{time.time()-t0:.0f}s, {sps:.0f} sps]")

    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Save model
    save_path = f"/checkpoints/pumap/model_ded_{name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"d_in": D_IN, "hidden_dim": hidden_dim, "n_layers": 3, "d_out": 2},
        "name": name,
    }, save_path)

    # Evaluate (batched)
    model.eval()
    Z_parts = []
    with torch.no_grad():
        for i in range(0, actual_n, 100_000):
            if data_on_gpu:
                Z_parts.append(model(X_tensor[i:i+100_000]).cpu().numpy())
            else:
                Z_parts.append(model(X_tensor[i:i+100_000].to(device)).cpu().numpy())
    Z = np.concatenate(Z_parts)

    rng2 = np.random.RandomState(42)
    n_eval = min(10000, actual_n)
    eval_idx = rng2.choice(actual_n, n_eval, replace=False)
    X_eval, Z_eval = X[eval_idx], Z[eval_idx]

    n_pairs = 10000
    i_p = rng2.randint(0, n_eval, n_pairs)
    j_p = rng2.randint(0, n_eval, n_pairs)
    m = i_p != j_p; i_p, j_p = i_p[m], j_p[m]
    dh = np.linalg.norm(X_eval[i_p] - X_eval[j_p], axis=1)
    dl = np.linalg.norm(Z_eval[i_p] - Z_eval[j_p], axis=1)
    dist_corr = float(np.corrcoef(dh, dl)[0, 1])

    nn_h = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(X_eval)
    nn_l = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(Z_eval)
    _, idx_h = nn_h.kneighbors(X_eval)
    _, idx_l = nn_l.kneighbors(Z_eval)
    knn_10 = sum(len(set(idx_h[ii,1:]) & set(idx_l[ii,1:])) for ii in range(n_eval)) / (n_eval * 10)

    VOLUMES["/checkpoints"].commit()
    cost = train_time * 1.70 / 3600

    result = {
        "name": name, "dist_corr": dist_corr, "knn_10": knn_10,
        "n_edges": n_pos, "n_samples": actual_n, "n_params": n_params,
        "train_time_s": train_time, "peak_mem_gb": peak_mem, "cost_usd": cost,
        "save_path": save_path,
    }
    logging.info(f"[{name}] dc={dist_corr:.4f} knn={knn_10:.4f} edges={n_pos:,} ${cost:.2f}")

    if wandb:
        wandb.log(result); wandb.summary.update(result); wandb.finish()
    return result


@app.local_entrypoint()
def run():
    handles = [(e["name"], run_experiment.spawn(e)) for e in EXPERIMENTS]
    print(f"Spawned {len(handles)} experiments")

    results = []
    for name, h in handles:
        try:
            r = h.get(); results.append(r)
            print(f"  {name}: dc={r['dist_corr']:.4f} knn={r['knn_10']:.4f} edges={r['n_edges']:,} ${r['cost_usd']:.2f}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    results.sort(key=lambda x: x.get("dist_corr", 0) + x.get("knn_10", 0), reverse=True)
    print(f"\n{'='*80}")
    print(f"  DEDICATED INDEX TRAINING")
    print(f"{'='*80}")
    print(f"  {'Name':<30} {'DC':>7} {'KNN':>7} {'Edges':>12} {'Cost':>6}")
    print(f"  {'-'*65}")
    total = 0
    for r in results:
        total += r['cost_usd']
        print(f"  {r['name']:<30} {r['dist_corr']:>6.4f} {r['knn_10']:>6.4f} {r['n_edges']:>11,} ${r['cost_usd']:>5.2f}")
    print(f"\n  Total: ${total:.2f}")
