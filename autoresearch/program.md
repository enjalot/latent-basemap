# Parametric UMAP Autoresearch

Autonomous ML research for improving parametric UMAP's neighborhood preservation.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar30`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The autoresearch directory is small. Read these for full context:
   - `autoresearch/prepare.py` — fixed constants, data loading, evaluation harness. Do not modify.
   - `autoresearch/train.py` — the file you modify. Model architecture, optimizer, loss, training loop.
4. **Verify data exists**: Run `cd /Users/enjalot/code/latent-basemap && python autoresearch/prepare.py` to check all data files are present.
5. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row.
6. **Confirm and go**.

## Context: What We Know

This is a parametric UMAP project. The model maps 768d text embeddings (nomic-embed-text-v1.5) to 2D coordinates, trying to replicate what standard UMAP does. The key challenge:

**KNN preservation is stuck at 5-9%.** We've already swept:
- n_neighbors: 100 is best (the precomputed graph uses nn=100)
- Architecture: 512 hidden × 3 layers is the sweet spot (920K params)
- UMAP curve params: a=1.9, b=0.895 is best
- Loss balance: pos_ratio=0.5, correlation_weight=0.1 are optimal
- Higher pos_ratio hurts catastrophically

**What hasn't been tried:**
- Residual/skip connections in the architecture
- Different activation functions (GELU, SiLU, etc.)
- Layer normalization vs batch normalization
- Learning rate scheduling (warmup, cosine decay)
- Different optimizers (SGD with momentum, Lion, etc.)
- Weight initialization strategies
- Output normalization/scaling
- Multi-scale loss (combining different neighborhood sizes)
- Contrastive losses alongside or instead of BCE
- Bottleneck architectures (wider early layers, narrower later)
- Embedding normalization before the model
- Gradient accumulation for larger effective batch size
- Temperature scaling on the similarity function

## Experimentation

Each experiment runs locally on MPS (Apple M2 Max). Training runs for a **fixed time budget of 3 minutes** wall clock. Launch: `cd /Users/enjalot/code/latent-basemap && python autoresearch/train.py`.

**What you CAN do:**
- Modify `autoresearch/train.py` — everything is fair game: model architecture, optimizer, hyperparameters, loss function, training loop.

**What you CANNOT do:**
- Modify `autoresearch/prepare.py`. It is read-only. Contains the fixed evaluation, data loading, and constants.
- Install new packages. Only use what's already available (torch, numpy, scipy, sklearn).
- Modify the evaluation functions.

**The goal: maximize `knn_10`.** This is KNN preservation at k=10 — the fraction of each point's 10 nearest neighbors in high-dim that remain neighbors in the 2D output. Current best: ~0.087. Higher is better. You also get `knn_25`, `knn_50`, `trustworthiness`, `dist_corr`, `ref_procrustes`, and `ref_knn_overlap` — use these as secondary signals, but optimize primarily for `knn_10`.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement from cleaner code is always a win.

## Output format

The script prints a summary like:

```
---
knn_10: 0.087000
knn_25: 0.120000
knn_50: 0.180000
trustworthiness: 0.760000
dist_corr: 0.430000
ref_procrustes: 0.390000
ref_knn_overlap: 0.085000
training_seconds: 180.1
eval_seconds: 12.3
total_seconds: 192.4
num_epochs: 3
num_steps: 4500
num_params: 920066
hidden_dim: 512
n_layers: 3
```

Extract the key metric: `grep "^knn_10:" run.log`

## Logging results

When an experiment is done, log it to `autoresearch/results.tsv` (tab-separated).

Header and columns:

```
commit	knn_10	knn_25	knn_50	status	description
```

1. git commit hash (short, 7 chars)
2. knn_10 achieved (e.g. 0.087000) — use 0.000000 for crashes
3. knn_25 achieved
4. knn_50 achieved
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Modify `autoresearch/train.py` with an experimental idea
3. git commit
4. Run: `cd /Users/enjalot/code/latent-basemap && python autoresearch/train.py > autoresearch/run.log 2>&1`
5. Read results: `grep "^knn_10:\|^knn_25:\|^knn_50:" autoresearch/run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 autoresearch/run.log` for the traceback.
7. Record in results.tsv (do NOT commit results.tsv — leave it untracked)
8. If knn_10 improved (higher), keep the commit
9. If knn_10 is equal or worse, `git reset --hard HEAD~1` to revert
10. **Loop back to step 1**

**Timeout**: Each run should take ~3 minutes for training + ~15 seconds for evaluation. If a run exceeds 5 minutes total, kill it and treat as failure.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. They may be asleep. Continue indefinitely until manually stopped. If you run out of ideas, think harder — re-read prepare.py, try combining near-misses, try radical changes.

## Research strategy hints

Think about the problem deeply. Standard UMAP directly optimizes point positions in 2D. We're asking a neural network to *learn a function* that maps *any* 768d point to the right 2D position. This is fundamentally harder because:

1. The network must generalize — it can't memorize individual positions
2. The training signal comes from edges (pairs), not direct coordinate supervision
3. Local neighborhood structure requires the function to be locally smooth but globally flexible
4. 768d → 2D is extreme compression — the model needs to learn which dimensions matter most

Ideas worth exploring:
- **Residual connections** — let the model learn incremental refinements
- **Input normalization** — L2-normalize embeddings before feeding to the model
- **Cosine similarity** — the embeddings may live on a hypersphere
- **Multi-head output** — predict 2D coords from multiple "views" and average
- **Feature bottleneck** — compress to intermediate dim before final 2D projection
- **Triplet/contrastive loss** — directly optimize neighborhood ranking
- **Hard negative mining** — focus on the hardest-to-separate negatives
- **Warmup + decay** — start slow, ramp up, then decay learning rate
- **Larger batch for better gradients** — with 3 min budget, fewer bigger batches may help
- **Gradient accumulation** — simulate larger batch without memory increase
