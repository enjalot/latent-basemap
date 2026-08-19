"""
Modal script to load a precomputed symmetric probability matrix (P_sym)
and compute negative edges from it.
"""

from modal import App, Image, Volume, method, enter
import logging
import pickle
import os
import numpy as np  # used for splitting data into chunks

from basemap.pumap.parametric_umap.datasets.edge_dataset import EdgeDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

N_CHUNKS = 20
K = 300
RANDOM_STATE = 0

# Constants and defaults for negative edge precomputation
# PSYM_RESULTS_FILE = "/checkpoints/pumap/wikipedia-en-chunked-120-all-MiniLM-L6-v2/precomputed_psym-0.pkl"
# NEGATIVE_EDGES_FILE = "/checkpoints/pumap/wikipedia-en-chunked-120-all-MiniLM-L6-v2/precomputed_negatives-0.pkl"
# PSYM_RESULTS_FILE = "/checkpoints/pumap/wikipedia-en-chunked-500-all-MiniLM-L6-v2/precomputed_psym.pkl"
# NEGATIVE_EDGES_FILE = "/checkpoints/pumap/wikipedia-en-chunked-500-all-MiniLM-L6-v2/precomputed_negatives.pkl"
# PSYM_RESULTS_FILE = "/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_psym.pkl"
PSYM_RESULTS_FILE = "/checkpoints/pumap/ls-dataisplural/precomputed_psym-15.pkl"
# PSYM_RESULTS_FILE = "/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_psym-45.pkl"
# NEGATIVE_EDGES_FILE = "/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_negatives.pkl"
# NEGATIVE_EDGES_FILE = f"/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_negatives-{K}.pkl"
NEGATIVE_EDGES_FILE = f"/checkpoints/pumap/ls-dataisplural/precomputed_negatives-{K}.pkl"


# CPU_CONCURRENCY = 2

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

app = App("precompute-negatives")

@app.cls(
    # cpu=CPU_CONCURRENCY,
    timeout=60 * 60 * 10,  # 10 hours timeout
    container_idle_timeout=1200,
    image=st_image,
    volumes={
        "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
    },
)
class RemoteNegativesPrecomputer:
    @enter()
    def start_engine(self):
        print("Starting negative edge precomputation engine.")

    @method()
    def precompute_edges(self):
        """
        Single–machine version: load the full P_sym matrix,
        build the EdgeDataset, and compute negatives using multi-threading.
        """
        print("Loading precomputed P_sym from:", PSYM_RESULTS_FILE)
        if not os.path.exists(PSYM_RESULTS_FILE):
            raise FileNotFoundError(f"P_sym file not found at {PSYM_RESULTS_FILE}")

        with open(PSYM_RESULTS_FILE, "rb") as f:
            P_sym = pickle.load(f)["P_sym"]
        print("P_sym loaded successfully.")

        # Create the EdgeDataset instance with the loaded P_sym
        ed = EdgeDataset(P_sym)
        print("EdgeDataset created with", len(ed.pos_edges), "positive edges.")

        print("Precomputing negative edges and saving to", NEGATIVE_EDGES_FILE)
        ed.precompute_and_save_negative_edges(
            NEGATIVE_EDGES_FILE,
            random_state=RANDOM_STATE,
            # n_processes=CPU_CONCURRENCY,
            verbose=True
        )
        print("Negative edge precomputation complete.")

    @method()
    def precompute_edges_distributed(self):
        """
        Distributed negative edge precomputation using Modal's map.
        Instead of partitioning the nodes in the coordinator, we dispatch tasks that
        compute their own chunk based on the provided chunk index.
        """
        print("Starting distributed negative edge precomputation (coordinator).")
        # Instead of loading and partitioning here, dispatch tasks with the chunk indices.
        chunk_indices = list(range(N_CHUNKS))
        print(f"Dispatching {N_CHUNKS} distributed tasks for negative sampling.")

        distributed_results = self.precompute_edges_distributed_subtask.map(
            chunk_indices, order_outputs=True
        )
        all_negatives = []
        for negs in distributed_results:
            all_negatives.extend(negs)

        print(f"Distributed negative edge sampling complete. Total negatives sampled: {len(all_negatives)}")
        print("Saving distributed negative edges to", NEGATIVE_EDGES_FILE)
        os.makedirs(os.path.dirname(NEGATIVE_EDGES_FILE), exist_ok=True)
        with open(NEGATIVE_EDGES_FILE, "wb") as f:
            pickle.dump(all_negatives, f)
        print("Distributed negative edge precomputation and saving complete.")

    @method()
    def precompute_edges_distributed_subtask(self, chunk_idx: int):
        """
        Distributed subtask for negative edge sampling.
        Each invocation loads the P_sym file, creates the EdgeDataset, determines its portion
        of positive nodes based on the provided chunk index and total number of chunks, and then computes negative edges.
        """
        print(f"Distributed subtask started for chunk index {chunk_idx} out of {N_CHUNKS} chunks.")
        # Delegate the computation to the EdgeDataset class method.
        from basemap.pumap.parametric_umap.datasets.edge_dataset import distributed_sample_negative_edges_subtask
        neg_edges = distributed_sample_negative_edges_subtask(
            psym_filepath=PSYM_RESULTS_FILE,
            chunk_idx=chunk_idx,
            n_chunks=N_CHUNKS,
            random_state=RANDOM_STATE,
            k=K,
            verbose=True
        )
        print(f"Distributed subtask completed for chunk index {chunk_idx}, sampled {len(neg_edges)} negative edges.")
        return neg_edges

@app.local_entrypoint()
def run(distributed: bool = False):
    """
    Local entrypoint allowing you to select between the single–machine precomputation run
    or the distributed version (using Modal's map).
    
    Parameters:
      distributed: bool – if set to True, use distributed negative edge sampling.
      n_chunks: int – number of data chunks (i.e. remote tasks) to spawn in distributed mode.
    """
    if distributed:
        print("Running distributed negative edge precomputation with random_state:", RANDOM_STATE)
        precomputer = RemoteNegativesPrecomputer()
        precomputer.precompute_edges_distributed.remote()
    else:
        print("Running single–machine negative edge precomputation with random_state:", RANDOM_STATE)
        precomputer = RemoteNegativesPrecomputer()
        precomputer.precompute_edges.remote()