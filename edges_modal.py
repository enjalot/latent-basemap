"""
modal run edges_modal.py --n-processes 32 --random-state 0
"""

from modal import App, Image, Volume, method, enter
import numpy as np
import logging
from basemap.data_loader import MemmapArrayConcatenator
from basemap.pumap.parametric_umap.core import compute_all_p_umap
from basemap.pumap.parametric_umap.datasets.edge_dataset import EdgeDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants and defaults (you may parameterize these further as needed)
DATASET = [
    "/embeddings/wikipedia-en-chunked-120-all-MiniLM-L6-v2/train",
]
D_IN = 384
N_NEIGHBORS = 15

CPU_CONCURRENCY = 64

NEGATIVE_EDGES_FILE = "/checkpoints/pumap/wikipedia-en-chunked-120-all-MiniLM-L6-v2/precomputed_negatives.pkl"

# Create an image with just the required dependencies for this precomputation
st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.3",
        "torch==2.1.2",
        "scipy",
        "tqdm",
        "parametric_umap",
        "annoy"
    )
)

app = App("precompute-negatives")

@app.cls(
    cpu=CPU_CONCURRENCY,  # many CPUs for parallel sampling
    timeout=60 * 60 * 10,  # 10 hours timeout
    container_idle_timeout=1200,
    image=st_image,
    volumes={
        "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
        "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
    },
)
class RemoteNegativesPrecomputer:
    @enter()
    def start_engine(self):
        print("Starting precomputation engine.")
        
    @method()
    def precompute(self, random_state: int = 0):
        print(f"Starting precomputation of negative edges with random_state={random_state} and n_processes={CPU_CONCURRENCY}")
        
        # Load or create training data via memmap arrays (keeps memory usage low)
        X_train = MemmapArrayConcatenator(DATASET, D_IN)
        print("Loaded training data memmap with shape:", X_train.shape)
        
        # Compute the symmetric probability matrix (P_sym) used for gathering positive edges
        print("Computing symmetric probability matrix (P_sym)...")
        P_sym = compute_all_p_umap(X_train, k=N_NEIGHBORS)
        print("P_sym computed.")
        
        # Create the EdgeDataset instance and display info on positive edges
        ed = EdgeDataset(P_sym)
        print("EdgeDataset created with", len(ed.pos_edges), "positive edges.")
        
        # Precompute negative edges and save them to file
        print("Precomputing negative edges and saving to", NEGATIVE_EDGES_FILE)
        ed.precompute_and_save_negative_edges(NEGATIVE_EDGES_FILE, random_state=random_state, n_processes=CPU_CONCURRENCY, verbose=True)
        print("Negative edge precomputation complete.")

@app.local_entrypoint()
def run(random_state: int = 0):
    print("Running negative edge precomputation with random_state:", random_state)
    precomputer = RemoteNegativesPrecomputer()
    precomputer.precompute.remote(random_state=random_state)