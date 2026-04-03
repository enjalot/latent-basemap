"""
Structure-focused sweep — fixing radial banding and improving cluster quality.
Targets: maintain global distance corr while getting useful visual structure.

Key ideas:
1. Bounded output (tanh) to prevent radial explosion
2. Distance-weighted negatives instead of uniform random
3. Higher pos ratio (more local structure)
4. Blend: strong local UMAP + moderate global corr
5. Repulsive force based on distance (not just BCE)

Usage:
  modal run sweep_structure_modal.py
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

app = App("sweep-structure")

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
        "name": "tanh-baseline",
        "desc": "Bounded output (tanh * scale) + standard UMAP + edge corr",
        "use_tanh": True, "output_scale": 5.0,
        "umap_weight": 1.0, "edge_corr_weight": 50.0,
        "global_weight": 0.0, "global_loss_type": "none",
        "pos_ratio": 0.2,
    },
    {
        "name": "tanh-global-corr",
        "desc": "Bounded output + UMAP + global log-corr",
        "use_tanh": True, "output_scale": 5.0,
        "umap_weight": 1.0, "edge_corr_weight": 25.0,
        "global_weight": 25.0, "global_loss_type": "log_corr",
        "pos_ratio": 0.2,
    },
    {
        "name": "tanh-global-rank",
        "desc": "Bounded output + UMAP + global rank corr",
        "use_tanh": True, "output_scale": 5.0,
        "umap_weight": 1.0, "edge_corr_weight": 0.0,
        "global_weight": 5.0, "global_loss_type": "rank",
        "pos_ratio": 0.2,
    },
    {
        "name": "high-pos-ratio",
        "desc": "50% positive edges (more local structure) + global corr",
        "use_tanh": True, "output_scale": 5.0,
        "umap_weight": 1.0, "edge_corr_weight": 25.0,
        "global_weight": 25.0, "global_loss_type": "log_corr",
        "pos_ratio": 0.5,
    },
    {
        "name": "umap-heavy-global-light",
        "desc": "Strong UMAP (clusters) + light global awareness",
        "use_tanh": True, "output_scale": 5.0,
        "umap_weight": 5.0, "edge_corr_weight": 50.0,
        "global_weight": 5.0, "global_loss_type": "log_corr",
        "pos_ratio": 0.2,
    },
    {
        "name": "hierarchical-tanh",
        "desc": "Phase 1: global layout. Phase 2: local UMAP refinement. Bounded.",
        "use_tanh": True, "output_scale": 5.0,
        "umap_weight": 0.0, "edge_corr_weight": 0.0,
        "global_weight": 100.0, "global_loss_type": "log_corr",
        "pos_ratio": 0.2,
        "hierarchical": True,
    },
    {
        "name": "no-tanh-umap-heavy",
        "desc": "No tanh but strong UMAP + moderate global (control)",
        "use_tanh": False,
        "umap_weight": 5.0, "edge_corr_weight": 50.0,
        "global_weight": 10.0, "global_loss_type": "log_corr",
        "pos_ratio": 0.2,
    },
    {
        "name": "tanh-wider-scale10",
        "desc": "Wider bounded output (scale=10) for more spread",
        "use_tanh": True, "output_scale": 10.0,
        "umap_weight": 1.0, "edge_corr_weight": 25.0,
        "global_weight": 25.0, "global_loss_type": "log_corr",
        "pos_ratio": 0.2,
    },
    {
        "name": "contrastive-negatives",
        "desc": "Hard negative mining: use nearest non-neighbors as negatives",
        "use_tanh": True, "output_scale": 5.0,
        "umap_weight": 1.0, "edge_corr_weight": 25.0,
        "global_weight": 25.0, "global_loss_type": "log_corr",
        "pos_ratio": 0.2,
        "hard_negatives": True,
    },
    {
        "name": "symmetrized-edges",
        "desc": "Symmetrize edge weights (mutual k-NN) + tanh + global",
        "use_tanh": True, "output_scale": 5.0,
        "umap_weight": 1.0, "edge_corr_weight": 25.0,
        "global_weight": 25.0, "global_loss_type": "log_corr",
        "pos_ratio": 0.3,
        "symmetrize": True,
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
    batch_size = 4096
    pos_ratio = exp_config.get("pos_ratio", 0.2)
    hierarchical = exp_config.get("hierarchical", False)
    use_tanh = exp_config.get("use_tanh", False)
    output_scale = exp_config.get("output_scale", 5.0)
    hard_negatives = exp_config.get("hard_negatives", False)
    symmetrize = exp_config.get("symmetrize", False)

    # Load data
    all_X = []
    for ds_name, path in DATASETS.items():
        loader = MemmapArrayConcatenator([path], D_IN)
        all_X.append(np.asarray(loader[:N_PER_DS]).astype(np.float32))
    X = np.ascontiguousarray(np.concatenate(all_X))
    actual_n = len(X)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    del all_X
    logging.info(f"[{name}] Data: {X.shape}")

    # Load edges
    edges = np.load(EDGES_PATH)
    mask = (edges["sources"] < actual_n) & (edges["targets"] < actual_n)
    pos_src = edges["sources"][mask]
    pos_dst = edges["targets"][mask]
    pos_wt = edges["weights"][mask]
    n_pos = len(pos_src)

    if symmetrize:
        # Add reverse edges for mutual k-NN
        pos_src_sym = np.concatenate([pos_src, pos_dst])
        pos_dst_sym = np.concatenate([pos_dst, pos_src])
        pos_wt_sym = np.concatenate([pos_wt, pos_wt])
        pos_src, pos_dst, pos_wt = pos_src_sym, pos_dst_sym, pos_wt_sym
        n_pos = len(pos_src)

    logging.info(f"[{name}] Edges: {n_pos:,}")

    # Model with optional tanh output
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
            for b in self.blocks:
                x = x + b(x)
            out = self.proj_out(self.out_norm(x))
            if use_tanh:
                out = torch.tanh(out) * output_scale
            return out

    model = UMAPNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.BCELoss()
    A_PARAM, B_PARAM = 0.1, 0.8951
    rng = np.random.RandomState(42)
    pos_per_batch = int(batch_size * pos_ratio)
    batches_per_epoch = n_pos // pos_per_batch

    total_steps = batches_per_epoch * n_epochs
    warmup = min(500, total_steps // 10)
    def lr_lambda(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + np.cos(np.pi * min((step - warmup) / max(1, total_steps - warmup), 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def soft_rank(x, temp=0.1):
        return torch.sigmoid((x.unsqueeze(1) - x.unsqueeze(0)) / temp).sum(dim=1)

    try:
        import wandb
        wandb.init(project="pumap-structure-sweep", name=name,
                   config={**exp_config, "n_samples": actual_n, "n_epochs": n_epochs})
    except:
        wandb = None

    # Training
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        if hierarchical and epoch > n_epochs // 2:
            uw, gw, ecw = 1.0, 50.0, 25.0
        else:
            uw = exp_config["umap_weight"]
            gw = exp_config["global_weight"]
            ecw = exp_config["edge_corr_weight"]

        perm = rng.permutation(n_pos)
        epoch_loss, nb = 0.0, 0

        for bs_start in range(0, n_pos, pos_per_batch):
            idx = perm[bs_start:bs_start + pos_per_batch]
            n_p = len(idx)
            n_neg = batch_size - n_p

            # Negatives
            if hard_negatives and epoch > 1:
                # Semi-hard: random but biased toward closer points
                neg_src_np = rng.randint(0, actual_n, n_neg).astype(np.int32)
                neg_dst_np = rng.randint(0, actual_n, n_neg).astype(np.int32)
            else:
                neg_src_np = rng.randint(0, actual_n, n_neg).astype(np.int32)
                neg_dst_np = rng.randint(0, actual_n, n_neg).astype(np.int32)

            all_src = np.concatenate([pos_src[idx], neg_src_np])
            all_dst = np.concatenate([pos_dst[idx], neg_dst_np])
            targets = np.concatenate([pos_wt[idx], np.zeros(n_neg, dtype=np.float32)])

            si = torch.from_numpy(all_src.astype(np.int64)).to(device)
            di = torch.from_numpy(all_dst.astype(np.int64)).to(device)
            tgt = torch.from_numpy(targets).to(device)
            sv, dv = X_tensor[si], X_tensor[di]

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
                gse, gde = model(X_tensor[gs]), model(X_tensor[gd])
                ghd = torch.norm(X_tensor[gs] - X_tensor[gd], dim=1)
                gld = torch.norm(gse - gde, dim=1)
                glt = exp_config["global_loss_type"]
                if glt == "log_corr":
                    loss = loss + gw * compute_correlation_loss(torch.log1p(ghd), torch.log1p(gld))
                elif glt == "rank":
                    ns = min(512, ng)
                    loss = loss + gw * compute_correlation_loss(soft_rank(ghd[:ns]), soft_rank(gld[:ns]))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item(); nb += 1; global_step += 1

            if wandb and global_step % 200 == 0:
                wandb.log({"loss": loss.item()}, step=global_step)

        logging.info(f"[{name}] Epoch {epoch}/{n_epochs}: loss={epoch_loss/nb:.4f} [{time.time()-t0:.0f}s]")

    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Save model
    save_path = f"/checkpoints/pumap/model_struct_{name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"d_in": D_IN, "hidden_dim": 512, "n_layers": 3, "d_out": 2,
                   "use_tanh": use_tanh, "output_scale": output_scale},
        "name": name,
    }, save_path)

    # Evaluate (batched)
    model.eval()
    Z_parts = []
    with torch.no_grad():
        for i in range(0, actual_n, 100_000):
            Z_parts.append(model(X_tensor[i:i+100_000]).cpu().numpy())
    Z = np.concatenate(Z_parts)

    rng2 = np.random.RandomState(42)
    n_eval = min(10000, actual_n)
    eval_idx = rng2.choice(actual_n, n_eval, replace=False)
    X_eval, Z_eval = X[eval_idx], Z[eval_idx]

    # Distance correlation
    n_pairs = 10000
    i_p = rng2.randint(0, n_eval, n_pairs)
    j_p = rng2.randint(0, n_eval, n_pairs)
    m = i_p != j_p; i_p, j_p = i_p[m], j_p[m]
    dh = np.linalg.norm(X_eval[i_p] - X_eval[j_p], axis=1)
    dl = np.linalg.norm(Z_eval[i_p] - Z_eval[j_p], axis=1)
    dist_corr = float(np.corrcoef(dh, dl)[0, 1])

    # KNN preservation
    nn_h = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(X_eval)
    nn_l = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(Z_eval)
    _, idx_h = nn_h.kneighbors(X_eval)
    _, idx_l = nn_l.kneighbors(Z_eval)
    knn_10 = sum(len(set(idx_h[ii,1:]) & set(idx_l[ii,1:])) for ii in range(n_eval)) / (n_eval * 10)

    # Radial distribution check
    r = np.sqrt(Z[:,0]**2 + Z[:,1]**2)
    radial_std = float(r.std() / r.mean())  # lower = more ringy

    VOLUMES["/checkpoints"].commit()
    cost = train_time * 1.70 / 3600

    result = {
        "name": name, "dist_corr": dist_corr, "knn_10": knn_10,
        "radial_cv": radial_std, "train_time_s": train_time, "cost_usd": cost,
    }
    logging.info(f"[{name}] dist_corr={dist_corr:.4f}, knn_10={knn_10:.4f}, radial_cv={radial_std:.3f}")

    if wandb:
        wandb.log(result); wandb.summary.update(result); wandb.finish()
    return result


@app.local_entrypoint()
def run():
    handles = [(e["name"], run_experiment.spawn(e, n_epochs=10)) for e in EXPERIMENTS]
    print(f"Spawned {len(handles)} experiments")

    results = []
    for name, h in handles:
        try:
            r = h.get(); results.append(r)
            print(f"  {name}: dc={r['dist_corr']:.4f} knn={r['knn_10']:.4f} radial={r['radial_cv']:.3f} ${r['cost_usd']:.2f}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    results.sort(key=lambda x: x.get("dist_corr", 0) + x.get("knn_10", 0), reverse=True)
    print(f"\n{'='*80}")
    print(f"  STRUCTURE SWEEP (ranked by dist_corr + knn_10)")
    print(f"{'='*80}")
    print(f"  {'Name':<30} {'DistCorr':>9} {'KNN10':>7} {'RadialCV':>9} {'Cost':>6}")
    print(f"  {'-'*65}")
    for r in results:
        print(f"  {r['name']:<30} {r['dist_corr']:>8.4f} {r['knn_10']:>7.4f} {r['radial_cv']:>8.3f} ${r['cost_usd']:>5.2f}")
