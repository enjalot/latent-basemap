"""
python train_local.py --batch-size 512 --n-epochs 1 --learning-rate 0.0001
"""

import torch
from basemap.data_loader import MemmapArrayConcatenator
from basemap.monitored import UMAPMonitor, MonitoredParametricUMAP
import argparse

# Configuration
DATASET = [
    # "data/wikipedia-en-chunked-120-all-MiniLM-L6-v2/train"
    "data/"
]
WANDB_PROJECT = "basemap-all-minilm-l6-v2"
D_IN = 384

def train(dataset, batch_size, n_epochs, learning_rate):
    print(f"Training on datasets: {dataset}, dimensions: {D_IN}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_train = MemmapArrayConcatenator(dataset, D_IN)

    print("Making monitor")
    monitor = UMAPMonitor(
        use_wandb=True,
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
        n_processes=8
    )
    
    print("Saving model")
    pumap.save(f"checkpoints/{WANDB_PROJECT}-local-{batch_size}-{n_epochs}-{learning_rate}")
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
