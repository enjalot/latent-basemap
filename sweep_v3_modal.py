"""
Sweep v3: Building on umap-007 (baseline-edge-only), umap-010 (hierarchical),
and umap-012 (no-tanh-umap-heavy). All unbounded. Budget: ~$5 for 7 experiments.

Key insights from prior rounds:
- tanh causes square boundaries → drop it
- Strong UMAP BCE creates clusters (umap-012 best structure)
- Hierarchical (global→local) creates meaningful regions (umap-010)
- baseline edge-only has structure but stippled rings from uniform negatives
- Need: clusters + global distance awareness + no ring artifacts

Usage:
  modal run sweep_v3_modal.py
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

app = App("sweep-v3")

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
        # Build on umap-012: the best visual structure.
        # Original: umap=5, edge_corr=50, global=10 log_corr
        # Try: more epochs with same config to see if it improves
        "name": "v3-umap-heavy-20ep",
        "desc": "umap-012 config but 20 epochs for better convergence",
        "umap_weight": 5.0, "edge_corr_weight": 50.0,
        "global_weight": 10.0, "global_loss_type": "log_corr",
        "n_epochs": 20,
    },
    {
        # Build on umap-012 but increase global weight
        "name": "v3-umap-heavy-more-global",
        "desc": "Strong UMAP + stronger global correlation",
        "umap_weight": 5.0, "edge_corr_weight": 50.0,
        "global_weight": 30.0, "global_loss_type": "log_corr",
        "n_epochs": 10,
    },
    {
        # Hierarchical but with stronger phase 2 (more UMAP epochs)
        # Original umap-010: 5ep global, 5ep global+local
        # Try: 3ep global, 7ep local-heavy
        "name": "v3-hierarchical-7030",
        "desc": "30% global warmup, 70% local-heavy refinement",
        "phase1_epochs": 3, "phase2_epochs": 7,
        "phase1": {"umap_weight": 0.0, "edge_corr_weight": 0.0,
                   "global_weight": 100.0, "global_loss_type": "log_corr"},
        "phase2": {"umap_weight": 5.0, "edge_corr_weight": 50.0,
                   "global_weight": 10.0, "global_loss_type": "log_corr"},
        "n_epochs": 10,
        "hierarchical_custom": True,
    },
    {
        # Distance-weighted negatives: instead of uniform random, sample negatives
        # proportional to their distance. Nearby non-neighbors are harder negatives.
        "name": "v3-weighted-neg-sampling",
        "desc": "Negative weight based on high-d distance (nearby non-neighbors matter more)",
        "umap_weight": 5.0, "edge_corr_weight": 50.0,
        "global_weight": 10.0, "global_loss_type": "log_corr",
        "neg_distance_weight": True,
        "n_epochs": 10,
    },
    {
        # Higher positive ratio: 40% positives for denser local structure
        "name": "v3-pos40-strong-umap",
        "desc": "40% positive ratio + strong UMAP for denser clusters",
        "umap_weight": 5.0, "edge_corr_weight": 50.0,
        "global_weight": 10.0, "global_loss_type": "log_corr",
        "pos_ratio": 0.4,
        "n_epochs": 10,
    },
    {
        # Use a/b params from standard UMAP (a=1.577, b=0.895, min_dist=0.1)
        # instead of our current a=0.1, b=0.8951
        "name": "v3-standard-ab-params",
        "desc": "Standard UMAP a/b params (a=1.577, b=0.895 for min_dist=0.1)",
        "umap_weight": 5.0, "edge_corr_weight": 50.0,
        "global_weight": 10.0, "global_loss_type": "log_corr",
        "a_param": 1.577, "b_param": 0.8951,
        "n_epochs": 10,
    },
    {
        # Bigger model: more capacity might help structure
        "name": "v3-bigger-model-1024",
        "desc": "Hidden dim 1024 (4x params) for more capacity",
        "umap_weight": 5.0, "edge_corr_weight": 50.0,
        "global_weight": 10.0, "global_loss_type": "log_corr",
        "hidden_dim": 1024,
        "n_epochs": 10,
    },
]


@app.function(
    gpu="L40S", timeout=60*60*3, scaledown_window=300,
    image=st_image, volumes=VOLUMES, secrets=SECRETS,
)
def run_experiment(exp_config: dict):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from sklearn.neighbors import NearestNeighbors
    from basemap.data_loader import MemmapArrayConcatenator
    from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

    device = "cuda"
    name = exp_config["name"]
    n_epochs = exp_config.get("n_epochs", 10)
    batch_size = 4096
    pos_ratio = exp_config.get("pos_ratio", 0.2)
    hidden_dim = exp_config.get("hidden_dim", 512)
    a_param = exp_config.get("a_param", 0.1)
    b_param = exp_config.get("b_param", 0.8951)
    hierarchical_custom = exp_config.get("hierarchical_custom", False)
    neg_distance_weight = exp_config.get("neg_distance_weight", False)

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
    logging.info(f"[{name}] Edges: {n_pos:,}")

    # Precompute norms for distance-weighted negative sampling
    if neg_distance_weight:
        # Precompute L2 norms for fast approximate distance
        X_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        logging.info(f"[{name}] Precomputed norms for distance-weighted negatives")

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
    rng = np.random.RandomState(42)
    pos_per_batch = int(batch_size * pos_ratio)
    batches_per_epoch = n_pos // pos_per_batch

    total_steps = batches_per_epoch * n_epochs
    warmup = min(500, total_steps // 10)
    def lr_lambda(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + np.cos(np.pi * min((step - warmup) / max(1, total_steps - warmup), 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    try:
        import wandb
        wandb.init(project="pumap-v3-sweep", name=name,
                   config={**exp_config, "n_samples": actual_n, "n_params": n_params})
    except:
        wandb = None

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        # Determine loss weights for this epoch
        if hierarchical_custom:
            phase1_ep = exp_config["phase1_epochs"]
            if epoch <= phase1_ep:
                cfg = exp_config["phase1"]
            else:
                cfg = exp_config["phase2"]
            uw, ecw, gw = cfg["umap_weight"], cfg["edge_corr_weight"], cfg["global_weight"]
            glt = cfg["global_loss_type"]
        else:
            uw = exp_config["umap_weight"]
            ecw = exp_config["edge_corr_weight"]
            gw = exp_config["global_weight"]
            glt = exp_config["global_loss_type"]

        perm = rng.permutation(n_pos)
        epoch_loss, nb = 0.0, 0

        for bs_start in range(0, n_pos, pos_per_batch):
            idx = perm[bs_start:bs_start + pos_per_batch]
            n_p = len(idx)
            n_neg = batch_size - n_p

            # Negative sampling
            if neg_distance_weight:
                # Sample source uniformly, then pick dst weighted by distance
                neg_src_np = rng.randint(0, actual_n, n_neg).astype(np.int32)
                # Approximate: weight by |norm_src - norm_dst| (fast proxy for L2 dist)
                # Closer points get higher weight as negatives (harder negatives)
                src_norms = X_norms[neg_src_np]
                # Create weights inversely proportional to norm difference
                candidates = rng.randint(0, actual_n, (n_neg, 4)).astype(np.int32)
                cand_norms = X_norms[candidates]
                norm_diffs = np.abs(cand_norms - src_norms[:, None])
                # Pick the candidate with smallest norm diff (closest = hardest negative)
                best = norm_diffs.argmin(axis=1)
                neg_dst_np = candidates[np.arange(n_neg), best]
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
                dists = torch.norm(se - de, dim=1, p=2*b_param)
                qs = torch.clamp(torch.pow(1 + a_param * dists, -1), 1e-7, 1-1e-7)
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
                if glt == "log_corr":
                    loss = loss + gw * compute_correlation_loss(torch.log1p(ghd), torch.log1p(gld))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item(); nb += 1; global_step += 1

            if wandb and global_step % 200 == 0:
                wandb.log({"loss": loss.item()}, step=global_step)

        logging.info(f"[{name}] Epoch {epoch}/{n_epochs}: loss={epoch_loss/nb:.4f} [{time.time()-t0:.0f}s]")

    train_time = time.time() - t0

    # Save model
    save_path = f"/checkpoints/pumap/model_v3_{name}.pt"
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
            Z_parts.append(model(X_tensor[i:i+100_000]).cpu().numpy())
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

    r = np.sqrt(Z[:,0]**2 + Z[:,1]**2)
    radial_cv = float(r.std() / r.mean())

    VOLUMES["/checkpoints"].commit()
    cost = train_time * 1.70 / 3600

    result = {
        "name": name, "dist_corr": dist_corr, "knn_10": knn_10,
        "radial_cv": radial_cv, "train_time_s": train_time, "cost_usd": cost,
        "n_params": n_params, "save_path": save_path,
    }
    logging.info(f"[{name}] dc={dist_corr:.4f} knn={knn_10:.4f} rcv={radial_cv:.3f} ${cost:.2f}")

    if wandb:
        wandb.log(result); wandb.summary.update(result); wandb.finish()
    return result


@app.local_entrypoint()
def run():
    handles = [(e["name"], run_experiment.spawn(e)) for e in EXPERIMENTS]
    print(f"Spawned {len(handles)} experiments (~${len(handles)*0.75:.2f} estimated)")

    results = []
    for name, h in handles:
        try:
            r = h.get(); results.append(r)
            print(f"  {name}: dc={r['dist_corr']:.4f} knn={r['knn_10']:.4f} ${r['cost_usd']:.2f}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    results.sort(key=lambda x: x.get("dist_corr", 0) + x.get("knn_10", 0), reverse=True)
    print(f"\n{'='*80}")
    print(f"  SWEEP v3 (ranked by dist_corr + knn_10)")
    print(f"{'='*80}")
    print(f"  {'Name':<35} {'DC':>7} {'KNN':>7} {'RCV':>6} {'Cost':>6} {'Params':>8}")
    print(f"  {'-'*75}")
    total_cost = 0
    for r in results:
        total_cost += r['cost_usd']
        print(f"  {r['name']:<35} {r['dist_corr']:>6.4f} {r['knn_10']:>6.4f} {r['radial_cv']:>5.3f} ${r['cost_usd']:>5.2f} {r['n_params']:>7,}")
    print(f"\n  Total cost: ${total_cost:.2f}")
