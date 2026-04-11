"""
Benchmark k-NN methods at different scales: FAISS GPU, FAISS CPU, LanceDB.
Tests both index build time and query time separately.

Usage:
  modal run bench_knn_modal.py --n-samples 10000
  modal run bench_knn_modal.py --n-samples 100000
  modal run bench_knn_modal.py --n-samples 1000000
"""
import time
import json
import logging
import tempfile
import numpy as np
from modal import App, Image, Secret, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2", "numpy==1.26.3", "scipy", "scikit-learn",
        "tqdm", "faiss-gpu-cu12", "lancedb", "pylance", "pyarrow",
    )
    .add_local_python_source("basemap")
)

with st_image.imports():
    import torch as _torch

app = App("bench-knn")

VOLUMES = {
    "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
}


@app.function(
    gpu="A10G",
    timeout=60 * 30,
    scaledown_window=120,
    image=st_image,
    volumes=VOLUMES,
)
def bench_knn(n_samples: int = 10000, k: int = 15, d: int = 384):
    import faiss
    import lancedb
    import pyarrow as pa

    from basemap.data_loader import MemmapArrayConcatenator

    results = {"n_samples": n_samples, "k": k, "d": d}

    # ── Load real data ──
    t0 = time.time()
    datasets = ["/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train"]
    loader = MemmapArrayConcatenator(datasets, d)
    X = np.asarray(loader[:n_samples]).astype(np.float32)
    X = np.ascontiguousarray(X)
    load_time = time.time() - t0
    logging.info(f"Loaded {X.shape} in {load_time:.1f}s")
    results["load_s"] = load_time

    # ═══════════════════════════════════════════════════════
    # FAISS GPU (GpuIndexFlatL2 or IVF)
    # ═══════════════════════════════════════════════════════
    logging.info(f"\n{'='*50}")
    logging.info(f"FAISS GPU — {n_samples:,} samples")
    logging.info(f"{'='*50}")

    gpu_res = faiss.StandardGpuResources()

    # Build
    t0 = time.time()
    if n_samples > 500_000:
        nlist = min(int(np.sqrt(n_samples)), 4096)
        index_cpu = faiss.index_factory(d, f"IVF{nlist},Flat")
        faiss_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
        faiss_gpu.train(X[:min(n_samples, nlist * 40)])
        faiss_gpu.add(X)
        faiss_gpu.nprobe = min(nlist // 4, 64)
        faiss_type = f"IVF{nlist}"
    else:
        faiss_gpu = faiss.GpuIndexFlatL2(gpu_res, d)
        faiss_gpu.add(X)
        faiss_type = "Flat"
    faiss_gpu_build = time.time() - t0
    logging.info(f"FAISS GPU build ({faiss_type}): {faiss_gpu_build:.2f}s")

    # Query (batch, all rows)
    t0 = time.time()
    batch_sz = 50_000
    all_dists = np.empty((n_samples, k + 1), dtype=np.float32)
    all_nbrs = np.empty((n_samples, k + 1), dtype=np.int64)
    for s in range(0, n_samples, batch_sz):
        e = min(s + batch_sz, n_samples)
        all_dists[s:e], all_nbrs[s:e] = faiss_gpu.search(X[s:e], k + 1)
    faiss_gpu_query = time.time() - t0
    logging.info(f"FAISS GPU query: {faiss_gpu_query:.2f}s")

    results["faiss_gpu_type"] = faiss_type
    results["faiss_gpu_build_s"] = faiss_gpu_build
    results["faiss_gpu_query_s"] = faiss_gpu_query
    results["faiss_gpu_total_s"] = faiss_gpu_build + faiss_gpu_query

    # Cleanup GPU memory
    del faiss_gpu
    import gc; gc.collect()

    # ═══════════════════════════════════════════════════════
    # FAISS CPU (Flat or HNSW)
    # ═══════════════════════════════════════════════════════
    logging.info(f"\n{'='*50}")
    logging.info(f"FAISS CPU — {n_samples:,} samples")
    logging.info(f"{'='*50}")

    t0 = time.time()
    if n_samples > 200_000:
        faiss_cpu = faiss.IndexHNSWFlat(d, 32)
        faiss_cpu.hnsw.efSearch = max(k * 4, 64)
        faiss_cpu.add(X)
        cpu_type = "HNSW"
    else:
        faiss_cpu = faiss.IndexFlatL2(d)
        faiss_cpu.add(X)
        cpu_type = "Flat"
    faiss_cpu_build = time.time() - t0
    logging.info(f"FAISS CPU build ({cpu_type}): {faiss_cpu_build:.2f}s")

    t0 = time.time()
    for s in range(0, n_samples, batch_sz):
        e = min(s + batch_sz, n_samples)
        all_dists[s:e], all_nbrs[s:e] = faiss_cpu.search(X[s:e], k + 1)
    faiss_cpu_query = time.time() - t0
    logging.info(f"FAISS CPU query: {faiss_cpu_query:.2f}s")

    results["faiss_cpu_type"] = cpu_type
    results["faiss_cpu_build_s"] = faiss_cpu_build
    results["faiss_cpu_query_s"] = faiss_cpu_query
    results["faiss_cpu_total_s"] = faiss_cpu_build + faiss_cpu_query

    del faiss_cpu; gc.collect()

    # ═══════════════════════════════════════════════════════
    # LanceDB
    # ═══════════════════════════════════════════════════════
    logging.info(f"\n{'='*50}")
    logging.info(f"LanceDB — {n_samples:,} samples")
    logging.info(f"{'='*50}")

    # Create a LanceDB table
    with tempfile.TemporaryDirectory() as tmpdir:
        db = lancedb.connect(tmpdir)

        # Build table
        t0 = time.time()
        table = db.create_table("bench", data=[
            {"id": i, "vector": X[i].tolist()}
            for i in range(n_samples)
        ])
        lance_ingest = time.time() - t0
        logging.info(f"LanceDB ingest: {lance_ingest:.2f}s")

        # Create vector index (GPU-accelerated if available)
        t0 = time.time()
        try:
            if n_samples >= 256:
                num_partitions = max(2, min(int(np.sqrt(n_samples)), 256))
                num_sub_vectors = min(d // 8, 48)
                table.create_index(
                    vector_column_name="vector",
                    index_type="IVF_PQ",
                    metric="L2",
                    num_partitions=num_partitions,
                    num_sub_vectors=num_sub_vectors,
                    accelerator="cuda",
                )
                lance_idx_type = f"IVF_PQ(p={num_partitions},sv={num_sub_vectors},gpu)"
            else:
                lance_idx_type = "none (too small)"
        except Exception as e:
            logging.warning(f"GPU index failed ({e}), trying CPU")
            try:
                num_partitions = max(2, min(int(np.sqrt(n_samples)), 256))
                num_sub_vectors = min(d // 8, 48)
                table.create_index(
                    vector_column_name="vector",
                    index_type="IVF_PQ",
                    metric="L2",
                    num_partitions=num_partitions,
                    num_sub_vectors=num_sub_vectors,
                )
                lance_idx_type = f"IVF_PQ(p={num_partitions},sv={num_sub_vectors},cpu)"
            except Exception as e2:
                logging.warning(f"CPU index also failed ({e2})")
                lance_idx_type = "none (failed)"

        lance_index = time.time() - t0
        logging.info(f"LanceDB index ({lance_idx_type}): {lance_index:.2f}s")

        # Query: k-NN for a SAMPLE of rows (full dataset is too slow for sequential API)
        t0 = time.time()
        sample_size = min(1000, n_samples)
        sample_idx = np.random.RandomState(42).choice(n_samples, sample_size, replace=False)
        for idx in sample_idx:
            table.search(X[idx].tolist()).limit(k + 1).to_list()
        lance_query_sample = time.time() - t0
        lance_qps = sample_size / lance_query_sample
        lance_query_est = n_samples / lance_qps  # estimated full query time
        logging.info(f"LanceDB query: {sample_size} samples in {lance_query_sample:.2f}s ({lance_qps:.0f} QPS)")
        logging.info(f"LanceDB estimated full query: {lance_query_est:.1f}s")
        lance_query = lance_query_est  # use estimate for comparison

        results["lance_idx_type"] = lance_idx_type
        results["lance_ingest_s"] = lance_ingest
        results["lance_index_s"] = lance_index
        results["lance_query_s"] = lance_query
        results["lance_total_s"] = lance_ingest + lance_index + lance_query

    # ═══════════════════════════════════════════════════════
    # Edge construction (for training pipeline comparison)
    # ═══════════════════════════════════════════════════════
    from basemap.pumap.parametric_umap.utils.graph import compute_knn_graph_fast

    logging.info(f"\n{'='*50}")
    logging.info(f"Full pipeline: compute_knn_graph_fast — {n_samples:,}")
    logging.info(f"{'='*50}")

    t0 = time.time()
    P_sym = compute_knn_graph_fast(X, k=k, use_gpu=True)
    pipeline_knn = time.time() - t0
    logging.info(f"Pipeline k-NN + P_sym: {pipeline_knn:.2f}s ({P_sym.nnz:,} edges)")
    results["pipeline_knn_s"] = pipeline_knn
    results["pipeline_edges"] = P_sym.nnz

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    cost = (load_time + max(faiss_gpu_build + faiss_gpu_query,
                            faiss_cpu_build + faiss_cpu_query,
                            lance_ingest + lance_index + lance_query)) * 1.10 / 3600
    results["est_cost_usd"] = cost

    logging.info(f"\n{'='*60}")
    logging.info(f"  k-NN BENCHMARK: {n_samples:,} samples, k={k}, d={d}")
    logging.info(f"{'='*60}")
    logging.info(f"  {'Method':<30} {'Build':>8} {'Query':>8} {'Total':>8}")
    logging.info(f"  {'-'*56}")
    logging.info(f"  {'FAISS GPU (' + faiss_type + ')':<30} {faiss_gpu_build:>7.1f}s {faiss_gpu_query:>7.1f}s {faiss_gpu_build+faiss_gpu_query:>7.1f}s")
    logging.info(f"  {'FAISS CPU (' + cpu_type + ')':<30} {faiss_cpu_build:>7.1f}s {faiss_cpu_query:>7.1f}s {faiss_cpu_build+faiss_cpu_query:>7.1f}s")
    logging.info(f"  {'LanceDB (' + lance_idx_type[:20] + ')':<30} {lance_ingest+lance_index:>7.1f}s {lance_query:>7.1f}s {lance_ingest+lance_index+lance_query:>7.1f}s")
    logging.info(f"  {'Pipeline (GPU + P_sym)':<30} {'':>8} {'':>8} {pipeline_knn:>7.1f}s")
    logging.info(f"{'='*60}")

    return results


@app.local_entrypoint()
def run(n_samples: int = 10000, k: int = 15):
    print(f"k-NN Benchmark: {n_samples:,} samples, k={k}")
    results = bench_knn.remote(n_samples, k)
    print(f"\n{json.dumps(results, indent=2)}")


@app.function(
    gpu="A100-40GB",
    timeout=60 * 30,
    scaledown_window=120,
    image=st_image,
    volumes=VOLUMES,
)
def bench_knn_a100(n_samples: int = 10000, k: int = 15, d: int = 384):
    """Same benchmark but on A100-40GB."""
    return bench_knn.local(n_samples, k, d)
