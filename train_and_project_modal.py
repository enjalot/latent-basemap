"""
Train 4 global distance configs and save models to checkpoint volume.
Then we download and project locally.

Usage:
  modal run train_and_project_modal.py
"""
from basemap.round0005_retirement import refuse_retired_launcher

refuse_retired_launcher("train_and_project_modal.py")

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

app = App("train-global-models")

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

CONFIGS = [
    {
        "name": "baseline-edge-only",
        "umap_weight": 1.0, "edge_corr_weight": 50.0,
        "global_weight": 0.0, "global_loss_type": "none",
    },
    {
        "name": "global-rank-corr",
        "umap_weight": 1.0, "edge_corr_weight": 0.0,
        "global_weight": 5.0, "global_loss_type": "rank",
    },
    {
        "name": "global-only-no-umap",
        "umap_weight": 0.0, "edge_corr_weight": 0.0,
        "global_weight": 100.0, "global_loss_type": "log_corr",
    },
    {
        "name": "hierarchical-2phase",
        "umap_weight": 0.0, "edge_corr_weight": 0.0,
        "global_weight": 100.0, "global_loss_type": "log_corr",
        "hierarchical": True,
    },
]


@app.function(
    gpu="L40S", timeout=60*60*2, scaledown_window=300,
    image=st_image, volumes=VOLUMES, secrets=SECRETS,
)
def train_and_save(config: dict, n_epochs: int = 10):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from basemap.data_loader import MemmapArrayConcatenator
    from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

    device = "cuda"
    name = config["name"]
    hierarchical = config.get("hierarchical", False)

    # Load data
    all_X = []
    for ds_name, path in DATASETS.items():
        loader = MemmapArrayConcatenator([path], D_IN)
        all_X.append(np.asarray(loader[:N_PER_DS]).astype(np.float32))
    X = np.ascontiguousarray(np.concatenate(all_X))
    actual_n = len(X)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    del all_X

    # Load edges
    edges = np.load(EDGES_PATH)
    mask = (edges["sources"] < actual_n) & (edges["targets"] < actual_n)
    pos_src, pos_dst, pos_wt = edges["sources"][mask], edges["targets"][mask], edges["weights"][mask]
    n_pos = len(pos_src)
    logging.info(f"[{name}] Data: {X.shape}, Edges: {n_pos:,}")

    # Model
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
            for b in self.blocks: x = x + b(x)
            return self.proj_out(self.out_norm(x))

    model = UMAPNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.BCELoss()
    A_PARAM, B_PARAM = 0.1, 0.8951
    rng = np.random.RandomState(42)
    batch_size = 4096
    pos_per_batch = int(batch_size * 0.2)

    # Schedule
    total_steps = (n_pos // pos_per_batch) * n_epochs
    warmup = min(500, total_steps // 10)
    def lr_lambda(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + np.cos(np.pi * min((step - warmup) / max(1, total_steps - warmup), 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def soft_rank(x, temp=0.1):
        return torch.sigmoid((x.unsqueeze(1) - x.unsqueeze(0)) / temp).sum(dim=1)

    # Training
    model.train()
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        if hierarchical and epoch > n_epochs // 2:
            uw, gw, ecw = 1.0, 50.0, 25.0
        else:
            uw = config["umap_weight"]
            gw = config["global_weight"]
            ecw = config["edge_corr_weight"]

        perm = rng.permutation(n_pos)
        epoch_loss, nb = 0.0, 0
        for bs in range(0, n_pos, pos_per_batch):
            idx = perm[bs:bs+pos_per_batch]
            n_neg = batch_size - len(idx)
            all_src = np.concatenate([pos_src[idx], rng.randint(0, actual_n, n_neg).astype(np.int32)])
            all_dst = np.concatenate([pos_dst[idx], rng.randint(0, actual_n, n_neg).astype(np.int32)])
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
                loss = loss + ecw * compute_correlation_loss(
                    torch.log1p(torch.norm(sv-dv, dim=1)), torch.log1p(torch.norm(se-de, dim=1)))
            if gw > 0:
                ng = 1024
                gs, gd = torch.randint(0, actual_n, (ng,), device=device), torch.randint(0, actual_n, (ng,), device=device)
                gse, gde = model(X_tensor[gs]), model(X_tensor[gd])
                ghd, gld = torch.norm(X_tensor[gs]-X_tensor[gd], dim=1), torch.norm(gse-gde, dim=1)
                glt = config["global_loss_type"]
                if glt == "log_corr":
                    loss = loss + gw * compute_correlation_loss(torch.log1p(ghd), torch.log1p(gld))
                elif glt == "rank":
                    ns = min(512, ng)
                    loss = loss + gw * compute_correlation_loss(soft_rank(ghd[:ns]), soft_rank(gld[:ns]))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item(); nb += 1

        logging.info(f"[{name}] Epoch {epoch}/{n_epochs}: loss={epoch_loss/nb:.4f} [{time.time()-t0:.0f}s]")

    # Save model
    save_path = f"/checkpoints/pumap/model_global_{name}.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"d_in": D_IN, "hidden_dim": 512, "n_layers": 3, "d_out": 2},
                "name": name}, save_path)
    VOLUMES["/checkpoints"].commit()
    logging.info(f"[{name}] Saved to {save_path}")

    return {"name": name, "train_time_s": time.time()-t0, "save_path": save_path}


@app.local_entrypoint()
def run():
    handles = [(c["name"], train_and_save.spawn(c, n_epochs=10)) for c in CONFIGS]
    print(f"Spawned {len(handles)} training runs")
    for name, h in handles:
        try:
            r = h.get()
            print(f"  {name}: {r['train_time_s']:.0f}s, saved to {r['save_path']}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
