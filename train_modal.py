"""
modal run train_modal.py --batch-size 512 --n-epochs 1 --learning-rate 0.0001
"""

from modal import App, Image, Secret, Volume, build, enter, exit, gpu, method
from basemap.pumap.parametric_umap import ParametricUMAP  # Use the unwrapped class
# from parametric_umap import ParametricUMAP
# TODO: importing these here means having the dependencies installed locally even
# though we only run them on Modal. If I dont depend here i can't figure out relative path imports
from basemap.data_loader import MemmapArrayConcatenator
from basemap.lancedb_loader import LanceDBLoader
# from basemap.monitored import UMAPMonitor, MonitoredParametricUMAP
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# DB_NAME = "enjalot/ls-fineweb-edu-100k"      # This is the (sub)directory or identifier for your DB inside /lancedb.
DB_NAME = "enjalot/ls-dataisplural"      # This is the (sub)directory or identifier for your DB inside /lancedb.
TABLE_NAME = "scopes-001"           # The table name (or scope) to read from
COLUMNS = ["vector"]           # The column that holds the embedding vectors

DATASET = [
    # f"/embeddings/fineweb-edu-sample-10BT-chunked-500-all-MiniLM-L6-v2/train",
    # f"/embeddings/RedPajama-Data-V2-sample-10B-chunked-500-all-MiniLM-L6-v2/train",
    # f"/embeddings/pile-uncopyrighted-chunked-500-all-MiniLM-L6-v2/train",
    # f"/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    # f"/embeddings/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    # f"/embeddings/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
    f"/embeddings/wikipedia-en-chunked-120-all-MiniLM-L6-v2/train",
    # f"/embeddings/wikipedia-en-chunked-500-all-MiniLM-L6-v2/train",
]
D_IN = 384
TESTING = True

# WANDB_PROJECT = "basemap-all-minilm-l6-v2-wikipedia-120-0"
# WANDB_PROJECT = "basemap-all-minilm-l6-v2-wikipedia-500"
# WANDB_PROJECT = "basemap-lancedb-fineweb-edu-100k"
WANDB_PROJECT = "basemap-lancedb-dataisplural"

GPU_CONCURRENCY = 1
CPU_CONCURRENCY = 2
# GPU_CONFIG = gpu.A100(size="80GB")
# GPU_CONFIG = gpu.A100(size="40GB")
GPU_CONFIG = gpu.A10G()
# GPU_CONFIG = gpu.H100()

# ------------------------------------------------------------------
# New constants for precomputed files (generated from psym_modal.py and edges_modal.py)
# PSYM_RESULTS_FILE = "/checkpoints/pumap/wikipedia-en-chunked-120-all-MiniLM-L6-v2/precomputed_psym-0.pkl"
# NEGATIVE_EDGES_FILE = "/checkpoints/pumap/wikipedia-en-chunked-120-all-MiniLM-L6-v2/precomputed_negatives-0.pkl"
# PSYM_RESULTS_FILE = "/checkpoints/pumap/wikipedia-en-chunked-500-all-MiniLM-L6-v2/precomputed_psym.pkl"
# NEGATIVE_EDGES_FILE = "/checkpoints/pumap/wikipedia-en-chunked-500-all-MiniLM-L6-v2/precomputed_negatives.pkl"
# PSYM_RESULTS_FILE = "/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_psym.pkl"
PSYM_RESULTS_FILE = "/checkpoints/pumap/ls-dataisplural/precomputed_psym-15.pkl"
# PSYM_RESULTS_FILE = "/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_psym-45.pkl"
# NEGATIVE_EDGES_FILE = "/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_negatives.pkl"
# NEGATIVE_EDGES_FILE = "/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_negatives-45-45.pkl"
# NEGATIVE_EDGES_FILE = f"/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_negatives-150.pkl"
# NEGATIVE_EDGES_FILE = f"/checkpoints/pumap/ls-fineweb-edu-100k/precomputed_negatives-300.pkl"
NEGATIVE_EDGES_FILE = f"/checkpoints/pumap/ls-dataisplural/precomputed_negatives-300.pkl"
#

# ------------------------------------------------------------------

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "numpy==1.26.3",
        # "transformers==4.39.3",
        # "hf-transfer==0.1.6",
        # "huggingface_hub==0.22.2",
        "annoy",
        "parametric_umap==0.1.1",
        "einops==0.7.0",
        "bitsandbytes",
        "safetensors",
        "accelerate",
        "dataclasses",
        "tqdm",
        "pyarrow",
        "datasets",
        "simple_parsing",
        "wandb",
        "lancedb"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # .run_function(
    #     download_model_to_image,
    #     timeout=60 * 20,
    #     kwargs={
    #         "model_dir": MODEL_DIR,
    #         "model_name": MODEL_ID,
    #         "model_revision": MODEL_REVISION,
    #     },
    #     secrets=[Secret.from_name("huggingface-secret")],
    # )
)

with st_image.imports():
    import torch


app = App(
    "train-fineweb-sae"
)  

@app.cls(
    gpu=GPU_CONFIG,
    cpu=CPU_CONCURRENCY,
    concurrency_limit=GPU_CONCURRENCY,
    timeout=60 * 60 * 10, # 10 hours
    container_idle_timeout=1200,
    allow_concurrent_inputs=1,
    image=st_image,
    volumes={
        "/embeddings": Volume.from_name("embeddings", create_if_missing=True),
        "/checkpoints": Volume.from_name("checkpoints", create_if_missing=True),
        "/lancedb": Volume.from_name("lancedb", create_if_missing=True),
    }, 
    secrets=[Secret.from_name("enjalot-wandb-secret")],
)
class RemoteTrainer:
    @enter()
    def start_engine(self):
        self.device = torch.device("cuda")
        print("starting engine")

    @method()
    def train(self, batch_size, n_epochs, learning_rate):
        # print(f"Training on datasets: {DATASET}, dimensions: {D_IN}")
        # X_train = MemmapArrayConcatenator(DATASET, D_IN, testing=TESTING)
        X_train = LanceDBLoader(db_name=f"/lancedb/{DB_NAME}", table_name=TABLE_NAME, columns=COLUMNS)

        print("initializing model")
        # Initialize the model
        pumap = ParametricUMAP(
            n_components=2,
            n_layers=3,
            hidden_dim=1024,
            a=0.1,
            b=1.0,
            correlation_weight=0.1,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            device='cuda',
            use_batchnorm=True,
            use_dropout=False,
            clip_grad_norm=1.0,
            clip_grad_value=None,
            pos_ratio=0.75,
        )
        print("fitting model")
        # Fit the model with wandb instrumentation enabled
        pumap.fit(
            X_train, 
            low_memory=True,
            verbose=True,
            n_processes=CPU_CONCURRENCY,
            precomputed_p_sym_path=PSYM_RESULTS_FILE,
            precomputed_negatives_path=NEGATIVE_EDGES_FILE,
            use_wandb=True,
            wandb_project=WANDB_PROJECT,
            wandb_run_name=f"train-{batch_size}-{n_epochs}-{learning_rate}"
        )
        print("saving model")
        save_path = f"/checkpoints/{WANDB_PROJECT}/{pumap.wandb_run_id}"
        pumap.save(save_path)
        print(f"Model saved to {save_path}")

@app.local_entrypoint()
def run(batch_size: int = 512, n_epochs: int = 10, learning_rate: float = 1e-4):
    print(f"Running with batch size: {batch_size}, n_epochs: {n_epochs}, learning_rate: {learning_rate}")
    job = RemoteTrainer()
    job.train.remote(
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate
    )
