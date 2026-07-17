#!/usr/bin/env python3
"""
Download trained models from Modal and project datasets locally.
Creates UMAP parquet files in latent-scope format.

Usage:
  # Download models from Modal volume
  python project_local.py --download

  # Project ls-squad with all models
  python project_local.py --dataset ls-squad

  # Project ls-fineweb-edu-100k (needs MiniLM embedding first)
  python project_local.py --dataset ls-fineweb-edu-100k
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

MODEL_DIR = Path("models/global")
LS_DIR = Path(os.path.expanduser("~/latent-scope-demo"))

DATASETS = {
    "ls-squad": {
        "h5_path": LS_DIR / "ls-squad" / "embeddings" / "embedding-002.h5",
        "umap_dir": LS_DIR / "ls-squad" / "umaps",
        "d_in": 384,
    },
    "ls-fineweb-edu-100k": {
        # Will need MiniLM embedding — check for embedding-002.h5
        "h5_path": LS_DIR / "ls-fineweb-edu-100k" / "embeddings" / "embedding-002.h5",
        "umap_dir": LS_DIR / "ls-fineweb-edu-100k" / "umaps",
        "d_in": 384,
    },
}

MODEL_NAMES = [
    "baseline-edge-only",
    "global-rank-corr",
    "global-only-no-umap",
    "hierarchical-2phase",
]


class UMAPNet(nn.Module):
    def __init__(self, d_in=384, hidden_dim=512, d_out=2, n_layers=3,
                 use_tanh=False, output_scale=5.0):
        super().__init__()
        self.use_tanh = use_tanh
        self.output_scale = output_scale
        self.proj_in = nn.Linear(d_in, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, d_out)

    def forward(self, x):
        x = F.relu(self.proj_in(x))
        for b in self.blocks:
            x = x + b(x)
        out = self.proj_out(self.out_norm(x))
        if self.use_tanh:
            out = torch.tanh(out) * self.output_scale
        return out


def download_models():
    """Download models from Modal checkpoint volume."""
    import subprocess
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name in MODEL_NAMES:
        remote = f"pumap/model_global_{name}.pt"
        local = MODEL_DIR / f"{name}.pt"
        if local.exists():
            print(f"  Already exists: {local}")
            continue
        print(f"  Downloading {remote} -> {local}")
        # Use modal volume get
        # Find modal binary
        import shutil
        modal_bin = shutil.which("modal") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "modal")
        result = subprocess.run(
            [modal_bin, "volume", "get", "checkpoints", remote, str(local)],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr.strip()}")
        else:
            size = local.stat().st_size / 1024
            print(f"    OK ({size:.0f} KB)")


def load_model(model_path, device="cpu"):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model = UMAPNet(
        d_in=cfg.get("d_in", 384),
        hidden_dim=cfg.get("hidden_dim", 512),
        d_out=cfg.get("d_out", 2),
        n_layers=cfg.get("n_layers", 3),
        use_tanh=cfg.get("use_tanh", False),
        output_scale=cfg.get("output_scale", 5.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


def project_dataset(dataset_name, device="cpu", model_dir=None):
    """Project a dataset with all models, save as UMAP parquets."""
    ds = DATASETS[dataset_name]

    if not ds["h5_path"].exists():
        print(f"  ERROR: {ds['h5_path']} not found!")
        if dataset_name == "ls-fineweb-edu-100k":
            print("  Run: cd ~/code/latent-scope && ls embed ls-fineweb-edu-100k --model_id sentence-transformers/all-MiniLM-L6-v2")
        return

    # Load embeddings
    print(f"  Loading {ds['h5_path']}...")
    with h5py.File(ds["h5_path"], "r") as f:
        X = f["embeddings"][:].astype(np.float32)
    print(f"  Data: {X.shape}")

    umap_dir = ds["umap_dir"]
    umap_dir.mkdir(parents=True, exist_ok=True)

    # Find next available UMAP ID
    existing = [f.stem for f in umap_dir.glob("umap-*.json")]
    if existing:
        max_id = max(int(f.split("-")[1]) for f in existing)
    else:
        max_id = 0

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    # Determine which models to project
    if model_dir:
        model_names_and_paths = [(p.stem, p) for p in sorted(Path(model_dir).glob("*.pt"))]
    else:
        model_names_and_paths = [(n, MODEL_DIR / f"{n}.pt") for n in MODEL_NAMES]

    for model_name, model_path in model_names_and_paths:
        if not model_path.exists():
            print(f"  Skipping {model_name} — model not found")
            continue

        max_id += 1
        umap_id = f"umap-{max_id:03d}"

        print(f"\n  Projecting with {model_name} -> {umap_id}")
        model = load_model(model_path, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Model: {n_params:,} params")

        # Project in batches
        t0 = time.time()
        Z_parts = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 4096):
                Z_parts.append(model(X_tensor[i:i+4096]).cpu().numpy())
        Z = np.concatenate(Z_parts)
        proj_time = time.time() - t0
        print(f"    Projected {len(X):,} rows in {proj_time:.2f}s")

        # Normalize to [-1, 1] (latent-scope convention) and use float32
        x_raw, y_raw = Z[:, 0], Z[:, 1]
        x_min, x_max = x_raw.min(), x_raw.max()
        y_min, y_max = y_raw.min(), y_raw.max()
        x_norm = 2 * (x_raw - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (y_raw - y_min) / (y_max - y_min) - 1
        df = pd.DataFrame({
            "x": x_norm.astype(np.float32),
            "y": y_norm.astype(np.float32),
        })

        # Save parquet
        parquet_path = umap_dir / f"{umap_id}.parquet"
        df.to_parquet(parquet_path)
        print(f"    Saved: {parquet_path}")

        # Save metadata JSON — match latent-scope format exactly
        import json
        meta = {
            "id": umap_id,
            "embedding_id": "embedding-002",
            "neighbors": 15,
            "min_dist": 0.1,
            "min_values": [float(x_min), float(y_min)],
            "max_values": [float(x_max), float(y_max)],
            "note": f"parametric-umap: {model_name}",
        }
        json_path = umap_dir / f"{umap_id}.json"
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"    Metadata: {json_path}")


if __name__ == "__main__":
    from basemap.round0005_retirement import refuse_retired_launcher
    refuse_retired_launcher("project_local.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download models from Modal")
    parser.add_argument("--dataset", type=str, help="Dataset to project")
    parser.add_argument("--device", type=str, default="cpu", help="Device (mps/cpu)")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory of .pt model files")
    args = parser.parse_args()

    if args.download:
        print("Downloading models from Modal...")
        download_models()

    if args.dataset:
        print(f"\nProjecting {args.dataset}...")
        if args.model_dir:
            project_dataset(args.dataset, device=args.device, model_dir=Path(args.model_dir))
        else:
            project_dataset(args.dataset, device=args.device)

    if not args.download and not args.dataset:
        print("Usage:")
        print("  python project_local.py --download")
        print("  python project_local.py --dataset ls-squad")
        print("  python project_local.py --dataset ls-fineweb-edu-100k")
