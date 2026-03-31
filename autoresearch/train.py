"""
Autoresearch training script for Parametric UMAP.
This is the ONLY file the agent modifies.
Usage: cd autoresearch && python train.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Add parent dir so we can import from basemap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from autoresearch.prepare import (
    load_data, evaluate, print_results,
    TIME_BUDGET, INPUT_DIM, OUTPUT_DIM
)
from basemap.pumap.parametric_umap.datasets.edge_dataset import EdgeDataset
from basemap.pumap.parametric_umap.datasets.covariates_datasets import VariableDataset
from basemap.pumap.parametric_umap.utils.losses import compute_correlation_loss

# ─── Model Architecture ────────────────────────────────────────────────────
# MODIFY THIS: Try different architectures, activations, skip connections, etc.

class UMAPNet(nn.Module):
    """MLP for parametric UMAP. Modify this architecture freely."""
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=512, output_dim=OUTPUT_DIM, n_layers=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Hyperparameters ────────────────────────────────────────────────────────
# MODIFY THESE: Tune freely.

HIDDEN_DIM = 512
N_LAYERS = 3
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
POS_RATIO = 0.5
CLIP_GRAD_NORM = 1.0

# UMAP curve parameters: q_ij = (1 + a * ||z_i - z_j||^{2b})^{-1}
A_PARAM = 1.9
B_PARAM = 0.8951

# Loss weights
CORRELATION_WEIGHT = 0.1     # weight for distance correlation loss


# ─── Training Loop ──────────────────────────────────────────────────────────
# MODIFY THIS: Change optimizer, scheduler, loss computation, etc.

def train():
    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    # Load data (immutable)
    print("Loading data...")
    data = load_data()
    X = data["X"]
    P_sym = data["P_sym"]
    neg_edges = data["neg_edges"]

    # Build edge dataset
    ed = EdgeDataset(P_sym)
    ed.neg_edges = neg_edges

    # Build model
    model = UMAPNet(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Data
    dataset = VariableDataset(X)
    if device != 'mps':
        dataset = dataset.to(device)

    # Edge loader
    from basemap.pumap.parametric_umap.datasets.edge_dataset import BalancedEdgeBatchIterator
    loader = BalancedEdgeBatchIterator(
        ed.pos_edges, ed.neg_edges,
        pos_weights=ed.pos_weights,
        pos_ratio=POS_RATIO,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Optimizer + cosine schedule with warmup
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    warmup_steps = 200
    total_steps_est = 12000  # ~5 epochs at batch 2048
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps_est - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.BCELoss()

    # Training
    model.train()
    print(f"Training for {TIME_BUDGET}s...")
    print(f"Batches per epoch: {len(loader)}")

    start_time = time.time()
    epoch = 0
    total_steps = 0

    while True:
        epoch += 1
        epoch_loss = 0.0
        epoch_umap = 0.0
        epoch_corr = 0.0
        n_batches = 0

        for edges, labels in loader:
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed >= TIME_BUDGET:
                break

            # edges is list of (src, dst) tuples, labels is np array of weights
            src_idx = torch.tensor([e[0] for e in edges], dtype=torch.long)
            dst_idx = torch.tensor([e[1] for e in edges], dtype=torch.long)
            targets = torch.tensor(labels, dtype=torch.float32, device=device)

            # Index on CPU, then move to device
            src_values = dataset[src_idx].to(device)
            dst_values = dataset[dst_idx].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward
            src_emb = model(src_values)
            dst_emb = model(dst_values)

            # UMAP similarity: q_ij = (1 + a * ||z_i - z_j||^{2b})^{-1}
            dists = torch.norm(src_emb - dst_emb, dim=1, p=2*B_PARAM)
            qs = torch.pow(1 + A_PARAM * dists, -1)
            qs = torch.clamp(qs, min=1e-7, max=1 - 1e-7)

            umap_loss = loss_fn(qs.float(), targets.float())

            # Distance correlation loss
            corr_loss = compute_correlation_loss(
                torch.norm(src_values - dst_values, dim=1),
                torch.norm(src_emb.float() - dst_emb.float(), dim=1)
            )

            loss = umap_loss + CORRELATION_WEIGHT * corr_loss

            # Backward
            loss.backward()
            if CLIP_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_umap += umap_loss.item()
            epoch_corr += corr_loss.item()
            n_batches += 1
            total_steps += 1

        elapsed = time.time() - start_time
        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_umap = epoch_umap / n_batches
            avg_corr = epoch_corr / n_batches
            print(f"Epoch {epoch}: loss={avg_loss:.4f} (umap={avg_umap:.4f}, corr={avg_corr:.4f}) [{elapsed:.0f}s]")

        if elapsed >= TIME_BUDGET:
            break

    training_seconds = time.time() - start_time
    print(f"\nTraining complete: {epoch} epochs, {total_steps} steps, {training_seconds:.1f}s")

    # Evaluate
    print("Evaluating...")
    eval_start = time.time()
    results = evaluate(model, X, data["Z_ref"], data["eval_idx"], device=device)
    eval_seconds = time.time() - eval_start

    # Print summary
    print_results(results)
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"eval_seconds: {eval_seconds:.1f}")
    print(f"total_seconds: {training_seconds + eval_seconds:.1f}")
    print(f"num_epochs: {epoch}")
    print(f"num_steps: {total_steps}")
    print(f"num_params: {n_params}")
    print(f"hidden_dim: {HIDDEN_DIM}")
    print(f"n_layers: {N_LAYERS}")


if __name__ == "__main__":
    train()
