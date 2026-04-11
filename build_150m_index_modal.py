"""
Build IVF_PQ index for the full 150M training mix (50M from each dataset).
Saves index + edge list to Modal checkpoint volume.

Usage:
  modal run build_150m_index_modal.py
  modal run build_150m_index_modal.py --n-per-dataset 10000000  # 10M per dataset = 30M total
  modal run build_150m_index_modal.py --query-only  # skip build, just query existing index
"""
import time
import logging
import os
import numpy as np
from modal import App, Image, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.3", "faiss-gpu-cu12", "scipy", "tqdm",
    )
    .add_local_python_source("basemap")
)

app = App("build-150m-index")

VOLUMES = {
    "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
    "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
}

DATASETS = {
    "fineweb": "/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "redpajama": "/embeddings/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "pile": "/embeddings/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
}
D_IN = 384


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 8,  # 8 hours for full 150M
    scaledown_window=600,
    image=st_image,
    volumes=VOLUMES,
    memory=65536,  # 64GB RAM — we stream from memmap, not full load
)
def build_and_query(n_per_dataset: int = 50_000_000, k: int = 15,
                    nprobe: int = 128, query_only: bool = False,
                    index_name: str = ""):
    import faiss
    from scipy import sparse
    from basemap.data_loader import MemmapArrayConcatenator

    total_n = n_per_dataset * len(DATASETS)
    tag = f"{total_n // 1_000_000}m"
    # Allow using a different (larger) index for query-only mode
    if index_name:
        index_path = f"/checkpoints/pumap/faiss_ivf_pq_{index_name}.index"
    else:
        index_path = f"/checkpoints/pumap/faiss_ivf_pq_{tag}.index"
    edges_path = f"/checkpoints/pumap/edges_{tag}_k{k}.npz"

    results = {"n_per_dataset": n_per_dataset, "total_n": total_n, "k": k}

    # ── Setup data loaders (memmap, not full load) ──
    t0 = time.time()
    loaders = []
    dataset_sizes = []
    for name, path in DATASETS.items():
        loader = MemmapArrayConcatenator([path], D_IN)
        n_available = loader.shape[0]
        n_take = min(n_per_dataset, n_available)
        loaders.append((name, loader, n_take))
        dataset_sizes.append(n_take)
        logging.info(f"  {name}: {n_available:,} available, will use {n_take:,}")

    actual_n = sum(dataset_sizes)
    setup_time = time.time() - t0
    logging.info(f"Total: {actual_n:,} vectors across {len(DATASETS)} datasets (setup: {setup_time:.1f}s)")
    results["actual_n"] = actual_n
    results["setup_s"] = setup_time
    load_time = setup_time  # for cost calculation

    def stream_batches(batch_size=500_000):
        """Yield contiguous float32 batches from all datasets without loading all into RAM."""
        for name, loader, n_take in loaders:
            for s in range(0, n_take, batch_size):
                e = min(s + batch_size, n_take)
                batch = np.asarray(loader[s:e]).astype(np.float32)
                yield np.ascontiguousarray(batch)

    def load_training_sample(n_train):
        """Load a small training sample (fits in RAM) for IVF centroid training."""
        per_ds = n_train // len(DATASETS) + 1
        parts = []
        for name, loader, n_take in loaders:
            take = min(per_ds, n_take)
            parts.append(np.asarray(loader[:take]).astype(np.float32))
        return np.ascontiguousarray(np.concatenate(parts)[:n_train])

    # ── Build index ──
    if not query_only:
        nlist = min(int(np.sqrt(actual_n)), 8192)
        num_sub = min(D_IN // 8, 48)
        train_size = min(actual_n, nlist * 40)

        logging.info(f"Building IVF_PQ: nlist={nlist}, PQ{num_sub}x8, train_size={train_size:,}")

        t0 = time.time()
        index = faiss.index_factory(D_IN, f"IVF{nlist},PQ{num_sub}x8")

        # Train on GPU
        gpu_res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index)

        logging.info("Loading training sample...")
        X_train = load_training_sample(train_size)
        logging.info(f"Training on GPU ({len(X_train):,} vectors)...")
        index_gpu.train(X_train)
        del X_train
        train_time = time.time() - t0
        logging.info(f"Training: {train_time:.1f}s")

        # Add vectors by streaming from memmap
        logging.info("Adding vectors (streaming from memmap)...")
        t0 = time.time()
        added = 0
        for batch in stream_batches(500_000):
            index_gpu.add(batch)
            added += len(batch)
            if added % 5_000_000 < 500_000:
                logging.info(f"  Added {added:,}/{actual_n:,}")
        add_time = time.time() - t0
        logging.info(f"Addition: {add_time:.1f}s ({added:,} vectors)")

        # Transfer to CPU and save
        index_cpu = faiss.index_gpu_to_cpu(index_gpu)
        del index_gpu

        t0 = time.time()
        faiss.write_index(index_cpu, index_path)
        save_time = time.time() - t0
        file_size = os.path.getsize(index_path) / (1024**3)
        logging.info(f"Saved: {index_path} ({file_size:.2f} GB) in {save_time:.1f}s")

        VOLUMES["/checkpoints"].commit()

        results["nlist"] = nlist
        results["train_s"] = train_time
        results["add_s"] = add_time
        results["save_s"] = save_time
        results["index_gb"] = file_size

    else:
        logging.info(f"Loading existing index from {index_path}...")
        t0 = time.time()
        index_cpu = faiss.read_index(index_path)
        logging.info(f"Index loaded in {time.time()-t0:.1f}s")

    # ── Query full dataset for k-NN ──
    logging.info(f"\nQuerying {actual_n:,} vectors (nprobe={nprobe})...")

    # Transfer index to GPU for fast queries
    gpu_res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
    index_gpu.nprobe = nprobe

    t0 = time.time()
    # Allocate output arrays (150M * 16 * 4B = ~10GB — fits in 64GB RAM)
    all_neighbors = np.empty((actual_n, k + 1), dtype=np.int64)
    all_distances = np.empty((actual_n, k + 1), dtype=np.float32)

    # Stream queries from memmap
    row_offset = 0
    for batch in stream_batches(100_000):
        bs = len(batch)
        all_distances[row_offset:row_offset+bs], all_neighbors[row_offset:row_offset+bs] = index_gpu.search(batch, k + 1)
        row_offset += bs
        elapsed = time.time() - t0
        if row_offset % 5_000_000 < 100_000 and row_offset > 0:
            qps = row_offset / elapsed
            eta = (actual_n - row_offset) / qps / 60
            logging.info(f"  Queried {row_offset:,}/{actual_n:,} ({qps:.0f} QPS, ETA {eta:.0f}min)")

    query_time = time.time() - t0
    qps = actual_n / query_time
    logging.info(f"Query complete: {query_time:.1f}s ({qps:.0f} QPS)")

    del index_gpu

    results["query_s"] = query_time
    results["qps"] = qps

    # ── Build edge list (no adjacency sets needed!) ──
    logging.info("Building edge list...")
    t0 = time.time()

    # Remove self-matches by ID (IVF_PQ doesn't guarantee self at column 0)
    all_sources = []
    all_targets = []
    for i in range(actual_n):
        nbrs = all_neighbors[i]
        mask = nbrs != i
        valid = nbrs[mask][:k]
        all_sources.extend([i] * len(valid))
        all_targets.extend(valid)
    sources = np.array(all_sources, dtype=np.int32)
    targets = np.array(all_targets, dtype=np.int32)
    weights = np.ones(len(sources), dtype=np.float32) / k

    # Save as compressed npz
    np.savez_compressed(edges_path,
                        sources=sources, targets=targets, weights=weights,
                        n_nodes=actual_n, k=k, nprobe=nprobe)
    edge_time = time.time() - t0
    edge_size = os.path.getsize(edges_path) / (1024**3)
    logging.info(f"Edge list saved: {edges_path} ({edge_size:.2f} GB) in {edge_time:.1f}s")
    logging.info(f"  {len(sources):,} directed edges ({actual_n:,} * {k} = {actual_n*k:,})")

    VOLUMES["/checkpoints"].commit()

    results["edge_s"] = edge_time
    results["edge_gb"] = edge_size

    # ── Summary ──
    build_cost = (results.get("train_s", 0) + results.get("add_s", 0)) * 1.10 / 3600
    query_cost = query_time * 1.10 / 3600
    total_cost = (load_time + results.get("train_s", 0) + results.get("add_s", 0) + query_time + edge_time) * 1.10 / 3600

    logging.info(f"\n{'='*60}")
    logging.info(f"  150M INDEX BUILD + QUERY")
    logging.info(f"{'='*60}")
    logging.info(f"  Data: {actual_n:,} vectors ({actual_n/1e6:.0f}M)")
    logging.info(f"  Load: {load_time:.1f}s")
    if not query_only:
        logging.info(f"  Build: {results.get('train_s',0) + results.get('add_s',0):.1f}s (index: {results.get('index_gb',0):.2f} GB)")
    logging.info(f"  Query: {query_time:.1f}s ({qps:.0f} QPS)")
    logging.info(f"  Edges: {edge_time:.1f}s ({edge_size:.2f} GB)")
    logging.info(f"  Cost: ${total_cost:.2f} (build ${build_cost:.2f} + query ${query_cost:.2f})")
    logging.info(f"{'='*60}")

    results["total_cost_usd"] = total_cost

    return results


@app.function(
    gpu="A100-40GB",
    timeout=60 * 60 * 8,
    scaledown_window=600,
    image=st_image,
    volumes=VOLUMES,
    memory=65536,
)
def build_and_query_a100(n_per_dataset: int = 50_000_000, k: int = 15,
                         nprobe: int = 128, query_only: bool = False,
                         index_name: str = ""):
    return build_and_query.local(n_per_dataset, k, nprobe, query_only, index_name)


@app.local_entrypoint()
def run(n_per_dataset: int = 50_000_000, k: int = 15, nprobe: int = 128,
        query_only: bool = False, gpu: str = "a10g", index_name: str = ""):
    total = n_per_dataset * 3
    print(f"Building IVF_PQ for {total/1e6:.0f}M vectors ({n_per_dataset/1e6:.0f}M x 3 datasets) on {gpu}")
    if gpu == "a100":
        results = build_and_query_a100.remote(n_per_dataset, k, nprobe, query_only, index_name)
    else:
        results = build_and_query.remote(n_per_dataset, k, nprobe, query_only, index_name)
    print(f"\nResults:")
    for key, val in results.items():
        print(f"  {key}: {val}")
