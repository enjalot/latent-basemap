"""
python train_local.py --batch-size 32 --n-epochs 1 --learning-rate 0.0001
"""

import torch
# from basemap.data_loader import MemmapArrayConcatenator
from basemap.lancedb_loader import LanceDBLoader
from basemap.monitored import UMAPMonitor, MonitoredParametricUMAP
import argparse
import numpy as np

# Configuration
# DATASET = ["data/"]
DATASET = "/Users/enjalot/latent-scope-demo/ls-fineweb-edu-100k/lancedb"
TABLE = "scopes-001"

WANDB_PROJECT = "basemap-ls-fineweb-edu-100k"
D_IN = 768

UMAP_RESULTS_FILE = "data/precomputed_umap_results_ls-fineweb-edu-100k.pkl"
NEGATIVE_EDGES_FILE = "data/precomputed_negatives_lance_ls-fineweb-edu-100k.pkl"

def train(dataset, batch_size, n_epochs, learning_rate):
    print(f"Training on datasets: {dataset}, dimensions: {D_IN}")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # X_train = MemmapArrayConcatenator(dataset, D_IN)
    X_train = LanceDBLoader(db_name=DATASET, table_name=TABLE, columns=["vector"])
    # Convert to regular numpy array and take subset
    print("Converting to numpy array...")
    # X_train = np.array(X_train)  # This uses the __array__ method
    # test_size = 1000
    # X_train = X_train[:test_size]
    print("as array")
    X = np.asarray(X_train).astype(np.float32)
    print("shape", X.shape)
    
    print(X_train.shape)
    
    print("Making monitor")
    monitor = UMAPMonitor(
        # use_wandb=True,
        use_wandb=False,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=f"train-local-{batch_size}-{n_epochs}-{learning_rate}",
    )
    
    print("Initializing model")
    pumap = MonitoredParametricUMAP(
        n_components=2,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device,
    )
    
    print("Fitting model")
    pumap.fit(
        X_train, 
        monitor=monitor,
        low_memory=True,
        verbose=True,
        n_processes=8,
        precomputed_p_sym_path=UMAP_RESULTS_FILE,
        precomputed_negatives_path=NEGATIVE_EDGES_FILE
    )
    
    print("Saving model")
    pumap.save(f"data/pumap-{WANDB_PROJECT}-local-{batch_size}-{n_epochs}-{learning_rate}")
    print("Done")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    print(f"Running with batch size: {args.batch_size}, n_epochs: {args.n_epochs}, learning_rate: {args.learning_rate}")
    train(
        dataset=DATASET,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()
