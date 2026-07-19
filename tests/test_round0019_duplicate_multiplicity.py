"""CPU checks for the R0019 duplicate-multiplicity treatment."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from basemap.artifact_identity import canonical_json, ordered_array_sha256, sha256_bytes
from basemap.duplicate_multiplicity import SCHEMA, load_duplicate_cap
from basemap.duplicate_diagnostics import duplicate_component_diagnostics
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
    DeviceArrayDataset,
    DeviceEdgeSampler,
)
from basemap.round0019_program import TRAIN_CONFIG


def _write_cap(path):
    arrays = {
        "excluded_rows": np.asarray([1, 3], dtype=np.int64),
        "representative_rows": np.asarray([0, 2], dtype=np.int64),
        "family_counts": np.asarray([2, 2], dtype=np.int64),
    }
    payload = {
        "schema": SCHEMA,
        "row_count": 6,
        "fixed_edges_per_source": 3,
        "multiplicity_cap": 1,
        "positive_source_policy": "uniform-over-retained-rows-and-k-neighbor-slots",
        "positive_destination_policy": "original-authenticated-graph-target-row",
        "negative_node_policy": "uniform-over-retained-rows",
        "excluded_row_count": 2,
        "retained_row_count": 4,
        "effective_positive_edges": 12,
        "array_sha256": {
            name: ordered_array_sha256(value) for name, value in arrays.items()
        },
    }
    metadata = {**payload, "identity_sha256": sha256_bytes(canonical_json(payload))}
    np.savez(path, metadata=np.asarray(canonical_json(metadata)), **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_duplicate_cap_artifact_and_program_config_are_exact(tmp_path):
    path = tmp_path / "cap.npz"
    digest = _write_cap(path)
    loaded = load_duplicate_cap(
        str(path), expected_sha256=digest, row_count=6, fixed_edges_per_source=3
    )
    assert loaded["excluded_rows"].tolist() == [1, 3]
    assert loaded["metadata"]["effective_positive_edges"] == 12

    treatment = TRAIN_CONFIG["execution"]["duplicate_multiplicity"]
    assert treatment["excluded_rows"] == 10_162
    assert treatment["retained_rows"] == 29_989_838
    assert treatment["effective_positive_edges"] == 449_847_570
    assert TRAIN_CONFIG["optimizer"]["use_amp"] == "bf16"
    assert TRAIN_CONFIG["optimizer"]["successful_positive_lr_updates"] == 500_000


def test_sampler_caps_positive_sources_and_negative_node_universe():
    n_nodes, k = 6, 3
    values = np.column_stack(
        (np.arange(n_nodes, dtype=np.float32), np.ones(n_nodes, dtype=np.float32))
    )
    dataset = DeviceArrayDataset(values, "cpu")
    sources = np.repeat(np.arange(n_nodes, dtype=np.int32), k)
    targets = np.asarray(
        [(source + slot + 1) % n_nodes for source in range(n_nodes) for slot in range(k)],
        dtype=np.int32,
    )
    allowed = np.asarray([0, 2, 4, 5], dtype=np.int64)
    sampler = DeviceEdgeSampler(
        dataset,
        sources,
        targets,
        np.ones(len(sources), dtype=np.float32),
        n_nodes=n_nodes,
        pos_ratio=0.25,
        batch_size=120,
        random_state=19,
        positive_target_mode="binary",
        weighted_edge_sampling=False,
        uniform_with_replacement=True,
        positive_source_rows=allowed,
        fixed_edges_per_source=k,
        device="cpu",
    )
    assert sampler.n_pos == len(allowed) * k
    assert sampler.source_n_pos == n_nodes * k

    src, dst, labels = next(iter(sampler))
    positive = labels == 1
    negative = labels == 0
    positive_sources = src[positive, 0].to(torch.int64).numpy()
    negative_sources = src[negative, 0].to(torch.int64).numpy()
    negative_targets = dst[negative, 0].to(torch.int64).numpy()
    assert set(positive_sources).issubset(set(allowed.tolist()))
    assert set(negative_sources).issubset(set(allowed.tolist()))
    assert set(negative_targets).issubset(set(allowed.tolist()))
    assert np.all(negative_sources != negative_targets)
    assert {1, 3}.isdisjoint(positive_sources)


def test_component_offset_metric_is_affine_frame_invariant():
    rng = np.random.RandomState(7)
    coordinates = rng.randn(500, 2).astype(np.float32)
    coordinates[[10, 11]] = np.asarray([[12.0, -4.0], [12.0, -4.0]])
    kwargs = {
        "excluded_rows": np.asarray([11], dtype=np.int64),
        "representative_rows": np.asarray([10], dtype=np.int64),
        "sample_seed": 91,
        "sample_size": 200,
    }
    baseline = duplicate_component_diagnostics(coordinates, **kwargs)
    transform = np.asarray([[2.0, 0.5], [-0.25, 1.5]], dtype=np.float32)
    moved = coordinates @ transform + np.asarray([30.0, -9.0], dtype=np.float32)
    changed_frame = duplicate_component_diagnostics(moved, **kwargs)
    assert baseline["sample_ids_sha256"] == changed_frame["sample_ids_sha256"]
    assert np.isclose(
        baseline["maximum_representative_mahalanobis"],
        changed_frame["maximum_representative_mahalanobis"],
        rtol=1e-6,
    )
