"""Quick query speed comparison: A10G vs A100 on existing 1M IVF_PQ index."""
import time, logging, numpy as np
from modal import App, Image, Volume

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.3", "faiss-gpu-cu12")
    .add_local_python_source("basemap")
)

app = App("bench-query-gpu")
VOLUMES = {
    "/embeddings": Volume.from_name("embeddings"),
    "/checkpoints": Volume.from_name("checkpoints"),
}

def _do_query(gpu_name):
    import faiss
    from basemap.data_loader import MemmapArrayConcatenator

    # Load 1M vectors
    loader = MemmapArrayConcatenator(["/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train"], 384)
    X = np.ascontiguousarray(np.asarray(loader[:1_000_000]).astype(np.float32))
    logging.info(f"Data: {X.shape}")

    # Load existing index
    index = faiss.read_index("/checkpoints/pumap/faiss_ivf_pq_1000000.index")
    logging.info(f"Index loaded: {index.ntotal} vectors")

    # Transfer to GPU
    gpu_res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    results = {"gpu": gpu_name}
    for nprobe in [64, 128, 256]:
        index_gpu.nprobe = nprobe
        t0 = time.time()
        d, n = index_gpu.search(X[:100_000], 16)
        elapsed = time.time() - t0
        qps = 100_000 / elapsed
        logging.info(f"  nprobe={nprobe}: {qps:,.0f} QPS ({elapsed:.2f}s for 100K queries)")
        results[f"nprobe{nprobe}_qps"] = qps

    # Project 150M
    qps_128 = results["nprobe128_qps"]
    est_150m = 150_000_000 / qps_128 / 3600
    rate = 2.10 if "100" in gpu_name else 1.10
    cost = est_150m * rate
    logging.info(f"\n  150M projection (nprobe=128): {est_150m:.1f} hr, ${cost:.2f}")
    results["proj_150m_hr"] = est_150m
    results["proj_150m_cost"] = cost
    return results

@app.function(gpu="A10G", timeout=300, image=st_image, volumes=VOLUMES)
def query_a10g():
    return _do_query("A10G")

@app.function(gpu="A100-40GB", timeout=300, image=st_image, volumes=VOLUMES)
def query_a100():
    return _do_query("A100-40GB")

@app.local_entrypoint()
def run():
    import concurrent.futures
    # Run both in parallel
    h1 = query_a10g.spawn()
    h2 = query_a100.spawn()
    r1 = h1.get()
    r2 = h2.get()
    print(f"\n{'='*50}")
    print(f"  QUERY SPEED: A10G vs A100 (1M IVF_PQ)")
    print(f"{'='*50}")
    for nprobe in [64, 128, 256]:
        q1 = r1[f"nprobe{nprobe}_qps"]
        q2 = r2[f"nprobe{nprobe}_qps"]
        print(f"  nprobe={nprobe}: A10G {q1:>8,.0f} QPS | A100 {q2:>8,.0f} QPS | speedup {q2/q1:.1f}x")
    print(f"\n  150M projection (nprobe=128):")
    print(f"    A10G: {r1['proj_150m_hr']:.1f} hr, ${r1['proj_150m_cost']:.2f}")
    print(f"    A100: {r2['proj_150m_hr']:.1f} hr, ${r2['proj_150m_cost']:.2f}")
