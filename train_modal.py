"""
modal run train_modal.py --batch-size 512 --n-epochs 1 --learning-rate 0.0001
"""

from modal import App, Image, Secret, Volume, build, enter, exit, gpu, method
# from parametric_umap import ParametricUMAP
# TODO: importing these here means having the dependencies installed locally even
# though we only run them on Modal. If I dont depend here i can't figure out relative path imports
from basemap.data_loader import MemmapArrayConcatenator
from basemap.monitored import UMAPMonitor, MonitoredParametricUMAP
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATASET = [
    # f"/embeddings/fineweb-edu-sample-10BT-chunked-500-all-MiniLM-L6-v2/train",
    # f"/embeddings/RedPajama-Data-V2-sample-10B-chunked-500-all-MiniLM-L6-v2/train",
    # f"/embeddings/pile-uncopyrighted-chunked-500-all-MiniLM-L6-v2/train",
    # f"/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    # f"/embeddings/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    # f"/embeddings/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
    f"/embeddings/wikipedia-en-chunked-120-all-MiniLM-L6-v2/train",
]
TESTING = True
WANDB_PROJECT = "basemap-all-minilm-l6-v2-wikipedia-120-0"
D_IN = 384
GPU_CONCURRENCY = 1
CPU_CONCURRENCY = 2
# GPU_CONFIG = gpu.A100(size="80GB")
# GPU_CONFIG = gpu.A100(size="40GB")
GPU_CONFIG = gpu.A10G()
# GPU_CONFIG = gpu.H100()

# ------------------------------------------------------------------
# New constants for precomputed files (generated from psym_modal.py and edges_modal.py)
PSYM_RESULTS_FILE = "/checkpoints/pumap/wikipedia-en-chunked-120-all-MiniLM-L6-v2/precomputed_psym-0.pkl"
NEGATIVE_EDGES_FILE = "/checkpoints/pumap/wikipedia-en-chunked-120-all-MiniLM-L6-v2/precomputed_negatives-0.pkl"
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
    }, 
    secrets=[Secret.from_name("enjalot-wandb-secret")],
)
class RemoteTrainer:
    @enter()
    def start_engine(self):
        self.device = torch.device("cuda")
        print("starting engine")

    @method()
    def train(self, dataset, batch_size, n_epochs, learning_rate):
        print(f"Training on datasets: {DATASET}, dimensions: {D_IN}")
        
        X_train = MemmapArrayConcatenator(DATASET, D_IN, testing=TESTING)

        print("making monitor")
        monitor = UMAPMonitor(
            use_wandb=True,
            # use_wandb=False,
            wandb_project=WANDB_PROJECT,
            wandb_run_name=f"train-{batch_size}-{n_epochs}-{learning_rate}",
        )
        print("initializing model")
        # Initialize the model
        pumap = MonitoredParametricUMAP(
            n_components=2,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            device='cuda',
        )
        print("fitting model")
        # Fit the model
        pumap.fit(
            X_train, 
            monitor=monitor,
            low_memory=True,
            verbose=True,
            n_processes=CPU_CONCURRENCY,
            precomputed_p_sym_path=PSYM_RESULTS_FILE,
            precomputed_negatives_path=NEGATIVE_EDGES_FILE
        )
        print("saving model")
        pumap.save(f"/checkpoints/{WANDB_PROJECT}-{batch_size}-{n_epochs}-{learning_rate}")
        print("done")

@app.local_entrypoint()
def run(batch_size: int = 512, n_epochs: int = 10, learning_rate: float = 1e-4):
    print(f"Running with batch size: {batch_size}, n_epochs: {n_epochs}, learning_rate: {learning_rate}")
    job = RemoteTrainer()
    job.train.remote(
        dataset=DATASET,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate
    )
