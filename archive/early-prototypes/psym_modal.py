#!/usr/bin/env python
"""
Modal script to precompute the symmetric probability matrix (P_sym)
and save it to a file for later use in negative edge computation.
"""

from modal import App, Image, Volume, method, enter, gpu
import logging

from basemap.data_loader import MemmapArrayConcatenator
from basemap.pumap.parametric_umap.utils.graph import compute_and_save_all_p_umap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Constants and defaults for P_sym precomputation
# DATASET = ["/embeddings/wikipedia-en-chunked-120-all-MiniLM-L6-v2/train"]
DATASET = ["/embeddings/wikipedia-en-chunked-500-all-MiniLM-L6-v2/train"]
D_IN = 384
N_NEIGHBORS = 15
# PSYM_RESULTS_FILE = "/checkpoints/pumap/wikipedia-en-chunked-120-all-MiniLM-L6-v2/precomputed_psym-0.pkl"
PSYM_RESULTS_FILE = "/checkpoints/pumap/wikipedia-en-chunked-500-all-MiniLM-L6-v2/precomputed_psym.pkl"
TESTING = False

# GPU_CONCURRENCY = 1
# CPU_CONCURRENCY = 32
# GPU_CONFIG = gpu.A100(size="80GB")
# GPU_CONFIG = gpu.A100(size="40GB")
# GPU_CONFIG = gpu.A10G()
# GPU_CONFIG = gpu.T4()
# GPU_CONFIG = gpu.H100()

# Create an image with the required dependencies
st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.3",
        "torch==2.1.2",
        "scipy",
        "tqdm",
        "parametric_umap",
        "annoy",
        "lancedb"
    )
)

app = App("precompute-psym")

@app.cls(
    # cpu=CPU_CONCURRENCY,
    timeout=60 * 60 * 4,  # 4 hours timeout (adjust as needed)
    container_idle_timeout=1200,
    image=st_image,
    volumes={
        "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
        "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
    },
)
class RemotePSymPrecomputer:
    @enter()
    def start_engine(self):
        print("Starting PSym precomputation engine.")

    @method()
    def precompute_psym(self):
        print("Loading training data from:", DATASET)
        X_train = MemmapArrayConcatenator(DATASET, D_IN, testing=TESTING)
        print("Loaded training data memmap with shape:", X_train.shape)

        print("Computing symmetric probability matrix (P_sym) with k =", N_NEIGHBORS)
        # This call computes P_sym and saves it to PSYM_RESULTS_FILE.
        compute_and_save_all_p_umap(X_train, k=N_NEIGHBORS, file_path=PSYM_RESULTS_FILE)
        print("P_sym computed and saved to:", PSYM_RESULTS_FILE)

@app.local_entrypoint()
def run():
    print("Running PSym precomputation.")
    precomputer = RemotePSymPrecomputer()
    precomputer.precompute_psym.remote() 