#!/usr/bin/env python3
"""Run the R0175 formula probe inside the pinned approx-umap toolchain."""
from __future__ import annotations

import argparse
import importlib.metadata
import json

import numpy as np
from approx_umap import ApproxUMAP


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(17_500)
    train = rng.normal(size=(512, 16)).astype(np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    queries = rng.normal(size=(73, 16)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    teacher = rng.normal(size=(512, 2)).astype(np.float32)

    model = ApproxUMAP(n_neighbors=15, metric="cosine", k=1, fn="inv", n_jobs=1)
    model._knn.fit(train)
    model.embedding_ = teacher
    observed = np.asarray(model.transform(queries), dtype=np.float32)
    distances, ids = model._knn.kneighbors(
        queries, n_neighbors=15, return_distance=True
    )
    weights = 1.0 / (distances + 1.0e-8)
    expected = np.sum(
        (weights / weights.sum(axis=1, keepdims=True))[:, :, None] * teacher[ids],
        axis=1,
    ).astype(np.float32)
    maximum = float(np.max(np.abs(observed - expected)))
    payload = {
        "schema": "round0175-official-approx-umap-probe-v1",
        "package": "approx-umap==0.2.0",
        "class": "approx_umap.ApproxUMAP",
        "metric": "cosine",
        "neighbors": 15,
        "k": 1,
        "fn": "inv",
        "epsilon": 1.0e-8,
        "train_rows": len(train),
        "query_rows": len(queries),
        "max_abs_error": maximum,
        "passed": maximum <= 1.0e-6,
        "environment_versions": {
            package: importlib.metadata.version(package)
            for package in ("approx-umap", "numpy", "scikit-learn", "umap-learn")
        },
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not payload["passed"]:
        raise RuntimeError(f"official aUMAP formula mismatch: {maximum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
