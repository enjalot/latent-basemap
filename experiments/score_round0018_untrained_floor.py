#!/usr/bin/env python3
"""Score deterministic same-architecture untrained controls on the R0018 map.

This is a post-hoc correction for the registered MiniLM projection control.  It
reuses R0018's authenticated coordinates, held-out queries, and persisted
high-D query truth; it performs no training and does not mutate R0018 outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import atomic_write_new_json


DEFAULT_COORDINATES = (
    "/data/latent-basemap/runs/round-0018/queue/artifacts/coordinates"
)
DEFAULT_QUERIES = "/data/latent-basemap/track1/minilm_queries.npy"
DEFAULT_TRUTH = (
    "/data/latent-basemap/runs/round-0018/queue/artifacts/panel/"
    "query-truth-cache/f1ff1ebc45655b35a2f34917ffce921e96f8302b783f3b8bc502271a34aed312.npz"
)
DEFAULT_PANEL = (
    "/data/latent-basemap/runs/round-0018/queue/artifacts/panel/panel.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", default=DEFAULT_COORDINATES)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--truth-cache", default=DEFAULT_TRUTH)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if len(args.seeds) < 3 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("at least three distinct untrained-control seeds are required")

    # Configure the shared production node module before constructing its exact
    # R0018 coordinate view and model architecture.
    from experiments import run_round0014_node as node
    node.configure_round0018()
    from experiments.score_complete_panel import projection_ffr
    import torch

    queries = np.load(args.queries, mmap_mode="r", allow_pickle=False)
    if queries.ndim != 2 or queries.shape[1] != 384 or queries.dtype != np.float32:
        raise ValueError(
            f"R0018 queries must be float32 [n,384], got {queries.shape} {queries.dtype}"
        )
    with np.load(args.truth_cache, allow_pickle=False) as archive:
        truth = np.asarray(archive["neighbors"], dtype=np.int64)
        truth_k = int(archive["k"])
        truth_rows = int(archive["query_rows"])
        truth_corpus = int(archive["corpus_cardinality"])
        truth_payload_sha256 = str(archive["payload_sha256"])
    if (
        truth_rows != len(queries)
        or truth_corpus != 30_000_000
        or truth_k < 10
        or truth.shape != (len(queries), truth_k)
        or np.any(truth < 0)
        or np.any(truth >= truth_corpus)
    ):
        raise ValueError("persisted R0018 query truth does not match the query/map universe")

    with open(args.panel, encoding="utf-8") as handle:
        panel = json.load(handle)
    trained_ffr = float(panel["projection"]["proj_ffr"])
    coordinates = node.StreamedCoordinateArray(os.path.realpath(args.coordinates))
    config = node._panel_config()

    cells = []
    started = time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = node._new_exact_model()
        model._init_model(queries.shape[1])
        model.is_fitted = True
        cell_started = time.monotonic()
        query_coordinates = model.transform(queries, batch_size=8192).astype(
            np.float32, copy=False
        )
        ffr, recall = projection_ffr(
            None,
            coordinates,
            None,
            query_coordinates,
            config,
            hi_truth=truth[:, : config.k_hit],
        )
        cells.append(
            {
                "seed": seed,
                "projection_ffr": ffr,
                "projection_recall_at_k": recall,
                "query_coordinate_min": query_coordinates.min(axis=0).tolist(),
                "query_coordinate_max": query_coordinates.max(axis=0).tolist(),
                "query_coordinate_std": query_coordinates.std(axis=0).tolist(),
                "wall_seconds": round(time.monotonic() - cell_started, 6),
            }
        )
        del model, query_coordinates
        torch.cuda.empty_cache()

    floors = [float(cell["projection_ffr"]) for cell in cells]
    conservative_floor = max(floors)
    body = {
        "schema": "round0018-minilm-untrained-projection-floor.v1",
        "purpose": "post-hoc evaluation of the registered same-architecture control",
        "trained_round": "0018",
        "architecture": {
            "name": node.TRAIN_CONFIG["model"]["architecture"],
            "input_dimension": node.TRAIN_CONFIG["model"]["input_dimension"],
            "hidden_dimension": node.TRAIN_CONFIG["model"]["hidden_dimension"],
            "hidden_layers": node.TRAIN_CONFIG["model"]["hidden_layers"],
            "output_dimension": node.TRAIN_CONFIG["model"]["output_dimension"],
            "use_batchnorm": node.TRAIN_CONFIG["model"]["use_batchnorm"],
            "use_dropout": node.TRAIN_CONFIG["model"]["use_dropout"],
        },
        "controls": cells,
        "floor_policy": "maximum of three deterministic untrained seeds",
        "untrained_projection_floor_ffr": conservative_floor,
        "trained_projection_ffr": trained_ffr,
        "trained_lift_over_untrained_floor": (
            trained_ffr / conservative_floor
            if conservative_floor > 0
            else float("inf")
        ),
        "inputs": {
            "coordinate_capability": expected_input_signature(
                os.path.join(args.coordinates, "actual-transform.json")
            ),
            "queries": expected_input_signature(args.queries),
            "query_truth_cache": expected_input_signature(args.truth_cache),
            "query_truth_payload_sha256": truth_payload_sha256,
            "registered_panel": expected_input_signature(args.panel),
        },
        "total_wall_seconds": round(time.monotonic() - started, 6),
        "training_performed": False,
    }
    receipt = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    atomic_write_new_json(args.out, receipt, immutable=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
