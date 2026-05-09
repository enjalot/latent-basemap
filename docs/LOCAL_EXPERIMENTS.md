# Local Experiment Setup

This repo was mostly exercised on Modal. On GSV, treat local experimentation as
two stages: first make runs reproducible and auditable, then use the RTX 5090 for
actual training once it is free.

## Current Local Assets

Run:

```bash
python3 -m experiments.inventory_assets \
  --local /data/checkpoints/pumap \
  --local /data/embeddings \
  --modal-volume pumap-data \
  --modal-volume checkpoints \
  --modal-volume embeddings \
  --modal-path / \
  --modal-path /precomputed \
  --modal-path /pumap \
  --output experiments/assets_inventory.json
```

As of the first GSV inventory:

- Local `/data/checkpoints/pumap` has `edges_150m_k15.npz` and
  `faiss_ivf_pq_150m.index`.
- Local `/data/embeddings` has the MiniLM 150M slice plus Jina and ColBERT
  assets.
- Modal `pumap-data:/precomputed` has LS-SQuAD `nn15`, `nn50`, and `nn100`
  `P_sym` and negative-edge pickles.
- Modal `checkpoints:/pumap` has 3M, 15M, 30M, and 150M edge/index artifacts
  plus prior trained model checkpoints.

## Environment

The system Python on GSV currently has no ML dependencies installed. Use a
repo-local virtual environment before running experiments:

```bash
cd /home/enjalot/code/latent-basemap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA training, replace the plain `torch` wheel with the CUDA wheel matching
the installed driver/CUDA stack before long runs. Do not launch training while
the GPU is occupied by SAE work.

## Run Bookkeeping

`experiments/run_experiment.py` writes both:

- `results.json`: metrics, config, timing, embedding stats.
- `manifest.json`: effective config, git commit/branch/dirty state, host,
  selected device, data asset paths/sizes, and evaluation contract.

The manifest explicitly labels evaluation mode:

- `holdout_rows`: graph/model trained on train rows; metrics can use held-out
  rows.
- `transductive_full_graph`: training uses all rows because precomputed graph
  indices must align; sampled metrics are from rows seen during training.

Do not compare these two modes as if they measure the same thing.

## Next Safe Steps Before GPU Training

1. Pull the small LS-SQuAD H5/reference UMAP assets from Modal or reconstruct
   them locally.
2. Run `smoke_test.yaml` on CPU to verify the environment.
3. Rerun the current autoresearch baseline and record whether `knn_10 ~= 0.272`
   reproduces.
4. Add a local edge-list trainer for `/data/checkpoints/pumap/edges_150m_k15.npz`
   before attempting 1M/5M/15M scale runs.
