"""
Upload latent-scope H5 embeddings, UMAP parquet files, and precomputed graphs to a Modal volume.

Usage:
  modal run upload_data_modal.py --dataset ls-squad
  modal run upload_data_modal.py --dataset ls-squad --include-precomputed
  modal run upload_data_modal.py --dataset ls-fineweb-edu-100k
"""
import os
import glob
from modal import App, Volume

app = App("upload-pumap-data")
vol = Volume.from_name("pumap-data", create_if_missing=True)

DATASETS = {
    "ls-squad": {
        "h5": os.path.expanduser("~/latent-scope-demo/ls-squad/embeddings/embedding-003.h5"),
        "umap": os.path.expanduser("~/latent-scope-demo/ls-squad/umaps/umap-001.parquet"),
    },
    "ls-fineweb-edu-100k": {
        "h5": os.path.expanduser("~/latent-scope-demo/ls-fineweb-edu-100k/embeddings/embedding-001.h5"),
        "umap": os.path.expanduser("~/latent-scope-demo/ls-fineweb-edu-100k/umaps/umap-001.parquet"),
    },
}


@app.local_entrypoint()
def run(dataset: str = "ls-squad", include_precomputed: bool = False):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASETS.keys())}")

    paths = DATASETS[dataset]

    with vol.batch_upload(force=True) as batch:
        # Upload embeddings and UMAP reference
        for name, local_path in paths.items():
            remote_path = f"/{dataset}/{os.path.basename(local_path)}"
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"Uploading {local_path} ({size_mb:.1f} MB) -> {remote_path}")
            batch.put_file(local_path, remote_path)

        # Upload precomputed graphs if requested
        if include_precomputed:
            precomputed_dir = "data/precomputed"
            pattern = f"{precomputed_dir}/{dataset}_nn*"
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"No precomputed files found matching {pattern}")
                print(f"Run: python precompute_local.py --dataset {dataset} first")
            else:
                for local_path in files:
                    remote_path = f"/precomputed/{os.path.basename(local_path)}"
                    size_mb = os.path.getsize(local_path) / (1024 * 1024)
                    print(f"Uploading {local_path} ({size_mb:.1f} MB) -> {remote_path}")
                    batch.put_file(local_path, remote_path)

    print(f"\nAll files for {dataset} uploaded to pumap-data volume.")
    print("Verify with: modal volume ls pumap-data /")
