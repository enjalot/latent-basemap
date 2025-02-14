#!/usr/bin/env python
"""
Modal script to precompute the symmetric probability matrix (P_sym)
and save it to a file for later use in negative edge computation.
This version loads data from a LanceDB scope.
"""

from modal import App, Image, Volume, method, enter, gpu
import logging

# Import the LanceDBLoader (which reads from a LanceDB table)
from basemap.lancedb_loader import LanceDBLoader

from basemap.pumap.parametric_umap.utils.graph import compute_and_save_all_p_umap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Constants and defaults for P_sym precomputation using LanceDB scope
# Adjust the following as needed.
DB_NAME = "enjalot/ls-fineweb-edu-100k"      # This is the (sub)directory or identifier for your DB inside /lancedb.
TABLE_NAME = "scopes-001"           # The table name (or scope) to read from
COLUMNS = ["vector"]           # The column that holds the embedding vectors
N_NEIGHBORS = 45
PSYM_RESULTS_FILE = "/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_psym-45.pkl"
TESTING = False

# Create an image with the required dependencies (note that lancedb is installed)
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

app = App("precompute-psym-scope")

@app.cls(
    # You may define cpu/gpu parameters as needed.
    timeout=60 * 60 * 4,  # 4 hours timeout (adjust as needed)
    container_idle_timeout=1200,
    image=st_image,
    volumes={
        "/lancedb": Volume.from_name("lancedb", create_if_missing=True),
        "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
    },
)
class RemoteScopePrecomputer:
    @enter()
    def start_engine(self):
        print("Starting P_sym precomputation engine for scope data.")

    @method()
    def precompute_psym(self):
        print("Connecting to LanceDB scope with DB_NAME:", DB_NAME, "TABLE:", TABLE_NAME)
        # Create the LanceDBLoader instance using the /lancedb volume.
        # Here, we construct the database path by prefixing with '/lancedb/'.
        scope_loader = LanceDBLoader(db_name=f"/lancedb/{DB_NAME}", table_name=TABLE_NAME, columns=COLUMNS)
        print("Loaded scope data with shape:", scope_loader.shape)
    
        print("Computing symmetric probability matrix (P_sym) with k =", N_NEIGHBORS)
        # Compute and save P_sym based on the data read from the LanceDB scope.
        compute_and_save_all_p_umap(scope_loader, k=N_NEIGHBORS, file_path=PSYM_RESULTS_FILE)
        print("P_sym computed and saved to:", PSYM_RESULTS_FILE)

@app.local_entrypoint()
def run():
    print("Running P_sym precomputation for scope data.")
    precomputer = RemoteScopePrecomputer()
    precomputer.precompute_psym.remote() 