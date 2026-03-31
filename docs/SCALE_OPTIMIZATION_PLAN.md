# Optimize Neighbor Computation for Scale

## Context
The current pipeline has an O(N²) bottleneck in negative sampling and materializes
a full N×N sparse P_sym matrix. These changes remove both bottlenecks to enable
training on 10M+ points, with a path to 1B+.

## Codebase Orientation
- `basemap/pumap/parametric_umap/utils/graph.py` — k-NN search, sigma/rho binary search, P_sym computation
- `basemap/pumap/parametric_umap/datasets/edge_dataset.py` — positive/negative edge sampling, batch iterators
- `basemap/pumap/parametric_umap/core.py` — ParametricUMAP training loop
- `edges_modal.py` — distributed negative edge precomputation (Modal)
- `psym_modal.py` — distributed P_sym precomputation (Modal)
- `scale_experiment.py` — scaling benchmarks
- `train_modal.py`, `train_local.py` — training entrypoints

## Step 1: Random Negative Sampling (removes O(N²) bottleneck)

**File:** `edge_dataset.py`, function `sample_negative_edges_worker` (line 130)

The current code does this per node:
```python
all_nodes = np.arange(total)           # O(N) alloc
mask = ~np.isin(all_nodes, connected)  # O(N) scan
candidates = all_nodes[mask]           # O(N) copy
targets = rng.choice(candidates, ...)  # sample from filtered set
```

Replace with simple random sampling — no exclusion:
```python
targets = rng.randint(0, total, size=k)
# Optional: exclude self
targets = targets[targets != node_int]
```

**Why this is correct:** With k=15 neighbors out of N nodes, collision probability
is k/N. At N=100K that's 0.015%. At N=1M it's 0.0015%. UMAP's reference
implementation, word2vec, and cuML UMAP all do this.

**Validation:** Run `validate_negative_edges()` before and after — invalid edge
count should be ~0 at any reasonable scale.

## Step 2: On-the-fly Negative Sampling During Training

**Goal:** Eliminate the entire negative edge precomputation step. Sample negatives
inside the training loop instead of precomputing and storing them.

**Files to modify:**
1. `edge_dataset.py` — Add a new `OnTheFlyBalancedIterator` class:
   - Takes only positive edges + weights (from P_sym nonzeros)
   - Each `__next__` call: yields a batch of positive edges AND randomly sampled
     negative pairs (just `rng.randint(0, n_nodes, size=n_neg)` for destinations)
   - No precomputed neg_edges storage needed

2. `core.py` — In the `fit()` method, add a code path that uses the new iterator
   when `precomputed_negatives_path` is None and dataset is large enough (e.g. >50K).

**Keep backward compat:** The existing precomputed path should still work for
small datasets and reproducibility testing.

## Step 3: Edge-list Representation Instead of P_sym Matrix

**Goal:** Replace the N×N sparse CSR matrix with flat arrays that use O(nnz) memory
with small constants.

**File:** `graph.py`

After computing `(sigma, rho, distances, neighbors)`, instead of building
a sparse matrix and calling `compute_p_umap_symmetric()`:

1. Compute edge weights directly:
   ```python
   # From compute_p_umap: p_ji = exp(-max(d - rho, 0) / sigma)
   sources = np.repeat(np.arange(N), k)  # shape: (N*k,)
   targets = neighbors.flatten()          # shape: (N*k,)
   weights = p_j_i.flatten()              # shape: (N*k,)
   ```

2. Symmetrize without matrix ops:
   - Create a dict mapping `(min(i,j), max(i,j))` to list of weights from each direction
   - Apply: `p_sym = p_ij + p_ji - p_ij * p_ji` for bidirectional edges
   - For unidirectional edges, weight = the one-sided probability
   - Output: three flat arrays `(sources, targets, sym_weights)`

3. Save as a lightweight dict:
   ```python
   {"sources": np.array, "targets": np.array, "weights": np.array}
   ```

4. Update `EdgeDataset` to accept this format, or create a new `EdgeListDataset`.

**Memory comparison at 1B points, k=15:**
- Current P_sym CSR: ~360 GB (int64 indices + float32 values + indptr)
- Edge lists: ~30B edges x 12 bytes = ~360 GB ... but after symmetrization and
  dedup, ~20B edges x 12 bytes = ~240 GB, and you can use int32 indices up to
  2B points which gives ~20B x 8 bytes = ~160 GB
- More importantly: edge lists are trivially shardable. P_sym CSR is not.

## Step 4: Replace Annoy with FAISS for k-NN

**File:** `graph.py` — Add `compute_sigma_i_faiss()` alongside existing functions.

```python
import faiss

def compute_sigma_i_faiss(X, k, tol=1e-5, max_iter=100, use_gpu=True):
    n, d = X.shape
    X = X.astype(np.float32)

    # Normalize for cosine similarity (inner product on unit vectors)
    faiss.normalize_L2(X)

    if use_gpu and faiss.get_num_gpus() > 0:
        # GPU IVF index — handles up to ~1B vectors
        nlist = min(int(np.sqrt(n)), 16384)
        index = faiss.index_factory(d, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        # Train on subset if dataset is huge
        train_size = min(n, nlist * 40)
        index.train(X[:train_size])
        index.add(X)
        index.nprobe = min(nlist // 4, 64)  # recall vs speed tradeoff
    else:
        # CPU HNSW — good up to ~100M
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.add(X)

    distances, neighbors = index.search(X, k + 1)
    # Remove self-matches, then proceed with existing sigma binary search...
```

Update `compute_all_p_umap()` to prefer FAISS when available:
```python
try:
    import faiss
    sigma, rho, distances, neighbors = compute_sigma_i_faiss(X, k, tol, max_iter)
except ImportError:
    # fall back to annoy, then sklearn
```

`faiss-cpu` is already in `basemap/pumap/requirements.txt`.

## Step 5: Sharded k-NN for 100M+ Scale

**New file:** `psym_sharded.py` (or modify `psym_modal.py`)

The idea: split the dataset into S shards. Each shard queries its vectors against
a shared FAISS index of ALL vectors. Each shard outputs its own edge list file.

```
Shard 0: vectors[0:N/S]       -> edges_shard_0.npz
Shard 1: vectors[N/S:2N/S]    -> edges_shard_1.npz
...
Shard S: vectors[(S-1)N/S:N]  -> edges_shard_S.npz
```

Each shard computes sigma, rho, and symmetric edge weights for its nodes, then
saves `(sources, targets, weights)`. Training loads shards round-robin or
concatenates the edge arrays.

For datasets that don't fit in a single FAISS index (~100M+ on GPU), use
FAISS's `IndexShards` or `IndexIVF` with `index.add_with_ids()` across machines.

## Validation Plan

After each step, run `scale_experiment.py` to verify:

1. **Correctness:** `distance_correlation` and `knn_preservation` should not degrade
   vs. baseline at 10K and 100K
2. **Speed:** Training throughput (samples/sec) should improve or stay constant
3. **Memory:** Peak memory should not grow faster than O(N)

Specific checks:
- After Step 1: Compare neg edge quality — `validate_negative_edges()` should show
  ~0 collisions at 100K+
- After Step 2: Training loss curves should match precomputed-negatives baseline
- After Step 3: Verify edge weights match P_sym values on a small (1K) test case
- After Step 4: Compare FAISS neighbors vs Annoy neighbors — recall@k should be >95%

## Implementation Order

Do Steps 1-2 first (biggest impact, least risk), validate, then proceed to 3-4.
Step 5 is only needed if you actually have 100M+ data ready to process.
