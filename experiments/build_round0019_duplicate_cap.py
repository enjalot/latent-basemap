#!/usr/bin/env python3
"""Build the preregistered R0019 cap for R0018's extreme duplicate island."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.duplicate_multiplicity import SCHEMA
from basemap.output_safety import atomic_save_new_npz


COORDINATE_ROOT = (
    "/data/latent-basemap/runs/round-0018/queue/artifacts/coordinates"
)
GRAPH_PATH = "/data/checkpoints/pumap/edges_30m_k15.npz"
ROW_COUNT = 30_000_000
K = 15


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from experiments import run_round0014_node as node
    node.configure_round0018()
    candidates = []
    candidate_coordinates = []
    for chunk_index in range(30):
        path = os.path.join(
            COORDINATE_ROOT, f"chunk-{chunk_index:05d}", "coordinates.npy"
        )
        coordinates = np.load(path, mmap_mode="r", allow_pickle=False)
        if coordinates.shape != (1_000_000, 2) or coordinates.dtype != np.float32:
            raise ValueError(f"R0018 coordinate chunk changed: {path}")
        local = np.flatnonzero(coordinates[:, 0] < -900.0)
        if len(local):
            candidates.append(local.astype(np.int64) + chunk_index * 1_000_000)
            candidate_coordinates.append(np.asarray(coordinates[local], dtype=np.float32))
    candidate_rows = np.concatenate(candidates)
    candidate_z = np.concatenate(candidate_coordinates)
    if len(candidate_rows) != 10_166:
        raise ValueError(f"registered extreme component has {len(candidate_rows)} rows")

    X = node.RoundMaterializedArray()
    candidate_x = np.asarray(X[candidate_rows], dtype=np.float16)
    unique_x, inverse, counts = np.unique(
        candidate_x, axis=0, return_inverse=True, return_counts=True
    )
    groups = []
    exclusions = []
    for group_id in range(len(unique_x)):
        rows = np.sort(candidate_rows[inverse == group_id])
        coords = candidate_z[inverse == group_id]
        if not np.all(coords == coords[0]):
            raise ValueError("an exact embedding family does not map to one exact coordinate")
        representative = int(rows[0])
        exclusions.extend(rows[1:].tolist())
        groups.append(
            {
                "representative_row": representative,
                "count": int(len(rows)),
                "embedding_sha256": hashlib.sha256(
                    np.ascontiguousarray(unique_x[group_id]).tobytes()
                ).hexdigest(),
                "r0018_coordinate": [float(value) for value in coords[0]],
            }
        )
    groups.sort(key=lambda group: group["representative_row"])
    representatives = np.asarray(
        [group["representative_row"] for group in groups], dtype=np.int64
    )
    family_counts = np.asarray([group["count"] for group in groups], dtype=np.int64)
    excluded = np.asarray(sorted(exclusions), dtype=np.int64)
    if family_counts.tolist() != [9852, 307, 6, 1] or len(excluded) != 10_162:
        raise ValueError(
            f"registered family counts changed: {family_counts.tolist()}, {len(excluded)}"
        )

    # Bind the treatment to the graph topology that makes the component
    # scientifically suspicious: every source row has exactly k ordered edges,
    # and every outgoing edge of the selected component remains inside it.
    with np.load(GRAPH_PATH, allow_pickle=False) as graph:
        sources = graph["sources"]
        targets = graph["targets"]
        if (
            int(graph["n_nodes"]) != ROW_COUNT
            or int(graph["k"]) != K
            or len(sources) != ROW_COUNT * K
            or len(targets) != ROW_COUNT * K
        ):
            raise ValueError("R0018 fixed-k graph shape changed")
        block_rows = 1_000_000
        for row_start in range(0, ROW_COUNT, block_rows):
            row_stop = min(row_start + block_rows, ROW_COUNT)
            observed = sources[row_start * K : row_stop * K]
            expected = np.repeat(
                np.arange(row_start, row_stop, dtype=np.int32), K
            )
            if not np.array_equal(observed, expected):
                raise ValueError("graph is not ordered fixed-k by source row")
        component_targets = targets[
            (candidate_rows[:, None] * K + np.arange(K)).reshape(-1)
        ]
        membership = np.zeros(ROW_COUNT, dtype=bool)
        membership[candidate_rows] = True
        if not membership[component_targets].all():
            raise ValueError("registered extreme component has an outgoing external edge")

    arrays = {
        "excluded_rows": excluded,
        "representative_rows": representatives,
        "family_counts": family_counts,
    }
    payload = {
        "schema": SCHEMA,
        "row_count": ROW_COUNT,
        "fixed_edges_per_source": K,
        "multiplicity_cap": 1,
        "positive_source_policy": "uniform-over-retained-rows-and-k-neighbor-slots",
        "positive_destination_policy": "original-authenticated-graph-target-row",
        "negative_node_policy": "uniform-over-retained-rows",
        "selection": {
            "source_map": "round-0018-seed42",
            "criterion": "coordinate_x < -900.0",
            "candidate_rows": len(candidate_rows),
            "exact_embedding_families": len(groups),
            "outgoing_edges_external_to_component": 0,
        },
        "families": groups,
        "excluded_row_count": len(excluded),
        "retained_row_count": ROW_COUNT - len(excluded),
        "effective_positive_edges": (ROW_COUNT - len(excluded)) * K,
        "array_sha256": {
            name: ordered_array_sha256(value) for name, value in arrays.items()
        },
        "inputs": {
            "coordinate_capability": expected_input_signature(
                os.path.join(COORDINATE_ROOT, "actual-transform.json")
            ),
            "graph": expected_input_signature(GRAPH_PATH),
        },
    }
    metadata = {
        **payload,
        "identity_sha256": sha256_bytes(canonical_json(payload)),
    }
    atomic_save_new_npz(
        args.out,
        immutable=True,
        metadata=np.asarray(canonical_json(metadata)),
        **arrays,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
