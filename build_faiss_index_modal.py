"""
Build a FAISS IVF_PQ or IVF_HNSW_SQ index offline and save to Modal volume.
The index is reusable across all training runs.

Usage:
  # Build IVF_PQ index for 1M samples (quick test)
  modal run build_faiss_index_modal.py --n-samples 1000000 --index-type ivf_pq

  # Build IVF_HNSW_SQ index for 100M
  modal run build_faiss_index_modal.py --n-samples 100000000 --index-type ivf_hnsw_sq

  # Build and also benchmark k-NN query speed
  modal run build_faiss_index_modal.py --n-samples 1000000 --benchmark
"""
import time
import logging
import json
import numpy as np
from modal import App, Image, Secret, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.3", "faiss-gpu-cu12", "tqdm",
    )
    .add_local_python_source("basemap")
)

app = App("build-faiss-index")

VOLUMES = {
    "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
    "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
}


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 4,
    scaledown_window=300,
    image=st_image,
    volumes=VOLUMES,
)
def build_index(n_samples: int = 1_000_000, index_type: str = "ivf_pq",
                d: int = 384, k: int = 15, benchmark: bool = True):
    import faiss
    from basemap.data_loader import MemmapArrayConcatenator

    results = {"n_samples": n_samples, "index_type": index_type, "d": d, "k": k}

    # ── Load data ──
    t0 = time.time()
    datasets = ["/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train"]
    loader = MemmapArrayConcatenator(datasets, d)
    total_available = loader.shape[0]
    actual_n = min(n_samples, total_available)
    X = np.asarray(loader[:actual_n]).astype(np.float32)
    X = np.ascontiguousarray(X)
    load_time = time.time() - t0
    logging.info(f"Loaded {X.shape} in {load_time:.1f}s ({total_available:,} available)")
    results["load_s"] = load_time
    results["actual_n"] = actual_n

    # ── Build index ──
    nlist = min(int(np.sqrt(actual_n)), 8192)
    train_size = min(actual_n, nlist * 40)

    logging.info(f"Building {index_type} index: nlist={nlist}, train_size={train_size:,}")

    t0 = time.time()

    if index_type == "ivf_pq":
        # IVF with Product Quantization — best compression
        num_sub = min(d // 8, 48)  # 48 sub-vectors for 384d
        nbits = 8
        index = faiss.index_factory(d, f"IVF{nlist},PQ{num_sub}x{nbits}")
        logging.info(f"IVF_PQ: nlist={nlist}, PQ{num_sub}x{nbits}")

    elif index_type == "ivf_sq":
        # IVF with Scalar Quantization — 4x compression, higher recall than PQ
        index = faiss.index_factory(d, f"IVF{nlist},SQ8")
        logging.info(f"IVF_SQ8: nlist={nlist}")

    elif index_type == "ivf_flat":
        # IVF with flat storage — no compression, exact within partition
        index = faiss.index_factory(d, f"IVF{nlist},Flat")
        logging.info(f"IVF_Flat: nlist={nlist}")

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Train on GPU
    gpu_res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    logging.info("Training index on GPU...")
    t_train_start = time.time()
    index_gpu.train(X[:train_size])
    train_time = time.time() - t_train_start
    logging.info(f"Index training: {train_time:.1f}s")

    # Add vectors (on GPU)
    logging.info("Adding vectors to index...")
    t_add_start = time.time()
    batch_add = 100_000
    for s in range(0, actual_n, batch_add):
        e = min(s + batch_add, actual_n)
        index_gpu.add(X[s:e])
        if s > 0 and s % (batch_add * 5) == 0:
            logging.info(f"  Added {s:,}/{actual_n:,}")
    add_time = time.time() - t_add_start
    logging.info(f"Vector addition: {add_time:.1f}s")

    build_time = time.time() - t0
    logging.info(f"Total build: {build_time:.1f}s")

    results["train_s"] = train_time
    results["add_s"] = add_time
    results["build_s"] = build_time
    results["nlist"] = nlist

    # Transfer back to CPU for saving
    index_cpu = faiss.index_gpu_to_cpu(index_gpu)
    del index_gpu

    # ── Save index ──
    save_path = f"/checkpoints/pumap/faiss_{index_type}_{actual_n}.index"
    t0 = time.time()
    faiss.write_index(index_cpu, save_path)
    save_time = time.time() - t0

    import os
    file_size = os.path.getsize(save_path) / (1024**3)
    logging.info(f"Index saved: {save_path} ({file_size:.2f} GB) in {save_time:.1f}s")
    results["save_s"] = save_time
    results["index_gb"] = file_size
    results["index_path"] = save_path

    VOLUMES["/checkpoints"].commit()

    # ── Benchmark queries ──
    if benchmark:
        logging.info("\nBenchmarking query performance...")

        for nprobe in [16, 64, 128, 256]:
            # Query on CPU
            index_cpu.nprobe = nprobe
            sample_n = min(10000, actual_n)
            sample_idx = np.random.RandomState(42).choice(actual_n, sample_n, replace=False)
            query_vecs = X[sample_idx]

            t0 = time.time()
            dists, neighbors = index_cpu.search(query_vecs, k + 1)
            cpu_time = time.time() - t0
            cpu_qps = sample_n / cpu_time

            # Query on GPU
            index_gpu2 = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
            index_gpu2.nprobe = nprobe

            t0 = time.time()
            dists_gpu, neighbors_gpu = index_gpu2.search(query_vecs, k + 1)
            gpu_time = time.time() - t0
            gpu_qps = sample_n / gpu_time

            del index_gpu2

            # Recall vs flat (on a small subset)
            if sample_n <= 10000:
                flat = faiss.IndexFlatL2(d)
                flat.add(X[:min(actual_n, 100000)])
                _, true_nbrs = flat.search(query_vecs[:1000], k + 1)
                _, approx_nbrs = index_cpu.search(query_vecs[:1000], k + 1)
                recall = np.mean([
                    len(set(true_nbrs[i, 1:]) & set(approx_nbrs[i, 1:])) / k
                    for i in range(1000)
                ])
                del flat
            else:
                recall = -1

            logging.info(f"  nprobe={nprobe:>3}: CPU {cpu_qps:>8,.0f} QPS ({cpu_time:.2f}s) | GPU {gpu_qps:>8,.0f} QPS ({gpu_time:.2f}s) | recall={recall:.3f}")

            results[f"nprobe{nprobe}_cpu_qps"] = cpu_qps
            results[f"nprobe{nprobe}_gpu_qps"] = gpu_qps
            results[f"nprobe{nprobe}_recall"] = recall

        # Project full-dataset k-NN time
        logging.info("\n  Projected full-dataset k-NN time (nprobe=128):")
        cpu_qps_128 = results.get("nprobe128_cpu_qps", 1)
        gpu_qps_128 = results.get("nprobe128_gpu_qps", 1)
        for n in [1_000_000, 10_000_000, 50_000_000, 100_000_000]:
            cpu_t = n / cpu_qps_128
            gpu_t = n / gpu_qps_128
            logging.info(f"    {n/1e6:.0f}M: CPU {cpu_t/60:.1f}min | GPU {gpu_t/60:.1f}min")
            results[f"proj_{n}_cpu_min"] = cpu_t / 60
            results[f"proj_{n}_gpu_min"] = gpu_t / 60

    # ── Summary ──
    cost = (build_time + load_time) * 1.10 / 3600
    results["cost_usd"] = cost

    logging.info(f"\n{'='*60}")
    logging.info(f"  INDEX BUILD: {index_type} for {actual_n:,} vectors")
    logging.info(f"{'='*60}")
    logging.info(f"  Train:   {train_time:.1f}s")
    logging.info(f"  Add:     {add_time:.1f}s")
    logging.info(f"  Save:    {save_time:.1f}s ({file_size:.2f} GB)")
    logging.info(f"  Total:   {build_time:.1f}s")
    logging.info(f"  Cost:    ${cost:.4f}")
    logging.info(f"{'='*60}")

    return results


@app.local_entrypoint()
def run(n_samples: int = 1_000_000, index_type: str = "ivf_pq",
        benchmark: bool = True):
    print(f"Building {index_type} index for {n_samples:,} vectors")
    results = build_index.remote(n_samples, index_type, benchmark=benchmark)
    print(f"\n{json.dumps(results, indent=2)}")
