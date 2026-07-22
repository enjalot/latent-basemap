"""Round 0027: matched jina MRL input-prefix experiment at two million rows."""
from __future__ import annotations

import copy
import json
import os
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .data_loader import (
    PrefixL2NormalizedArray,
    prefix_l2_preprocessing_contract,
)
from .graph_validation import data_fingerprint


ROUND_ID = "0027"
ROWS = 2_000_000
SOURCE_DIMENSION = 768
DIMENSIONS = (768, 384, 256)
SEEDS = (42, 43)
CELL_LABELS = tuple(
    f"d{dimension}_s{seed}"
    for dimension in DIMENSIONS
    for seed in SEEDS
)

TRAIN_PATH = (
    "/data/latent-basemap/jina-en-2M-nested/train/data-00000.npy"
)
TRAIN_SHA256 = (
    "7941f827eac6ac38ad45301198dc238a9fd7bbe16204c36a4031ce63a4115007"
)
TRAIN_BYTES = 3_072_000_128
SOURCE_4M_PATH = (
    "/data/latent-basemap/jina-en-4M-nested/train/data-00000.npy"
)
SOURCE_4M_SHA256 = (
    "4ca9dd6c6d2c10ccd70c4a61c0137dddeee0e89111a9a70d52abc47e68f9fdd4"
)
SOURCE_4M_BYTES = 6_144_000_128
PREFIX_PAYLOAD_SHA256 = (
    "0417424652affc27f2b7d0115db2d62b41497f34c356bede8787712b74efb2cf"
)

GRAPH_PATH = "/data/latent-basemap/jina-en-2M-nested/edges_k50_fuzzy.npz"
GRAPH_SHA256 = (
    "52567a6f9e6b4d4ad735e35f107c980a7ded2ca499d5a012fc4bb0b8c688902b"
)
GRAPH_BYTES = 1_788_739_880
GRAPH_EDGES = 149_061_552
GRAPH_K = 50
GRAPH_ENDPOINT_PROBE = {
    "edge_cosine": 0.5699,
    "random_cosine": 0.1137,
    "margin": 0.4562,
    "n_probe": 5000,
    "seed": 0,
}

CENTROIDS = {
    256: {
        "path": "/data/latent-basemap/jina-en-4M-nested/centroids_k256.npy",
        "sha256": "744b16279c30d8d38462e8f6b7c7ffac78f6154169fb8edce61ee7817ff674ec",
        "bytes": 786_560,
    },
    1024: {
        "path": "/data/latent-basemap/jina-en-4M-nested/centroids_k1024.npy",
        "sha256": "c5137332218c907e58f41999ff357fbe3ae98b63b5878e9f3553318060ae9968",
        "bytes": 3_145_856,
    },
}

QUERY_SEED = 20_260_727
QUERY_ROWS = 20_000
QUERY_UNIVERSE_START = ROWS
QUERY_UNIVERSE_STOP = 4_000_000
PANEL_ANCHORS = 10_000
PANEL_SEED = 123
PANEL_FRACTION = 0.001
SUCCESSFUL_UPDATES = 500_000
CANARY_UPDATES = 1_000


def parse_cell(label: str) -> tuple[int, int]:
    if label not in CELL_LABELS:
        raise ValueError(f"unknown Round 0027 cell: {label!r}")
    dimension_text, seed_text = label.split("_")
    return int(dimension_text[1:]), int(seed_text[1:])


def preprocessing_for_dimension(dimension: int) -> tuple[dict, dict]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"unsupported Round 0027 input dimension: {dimension}")
    return prefix_l2_preprocessing_contract(
        source_dimension=SOURCE_DIMENSION,
        output_dimension=dimension,
        normalize=dimension < SOURCE_DIMENSION,
    )


def input_array(dimension: int, *, path: str = TRAIN_PATH) \
        -> PrefixL2NormalizedArray:
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if (source.shape != (ROWS, SOURCE_DIMENSION) or
            source.dtype != np.dtype("float16") or
            not source.flags.c_contiguous):
        raise ValueError(
            f"Round 0027 input header changed: {source.shape} {source.dtype}")
    return PrefixL2NormalizedArray(
        source,
        source_dimension=SOURCE_DIMENSION,
        output_dimension=dimension,
        normalize=dimension < SOURCE_DIMENSION,
        source_paths=[path],
    )


def cosine_truth_array(*, path: str = TRAIN_PATH) -> PrefixL2NormalizedArray:
    """Full-768d row-renormalized view whose L2 ranking is exact cosine."""
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if (source.ndim != 2 or source.shape[1] != SOURCE_DIMENSION or
            source.dtype != np.dtype("float16")):
        raise ValueError(f"Round 0027 cosine-truth header changed: {path}")
    return PrefixL2NormalizedArray(
        source,
        source_dimension=SOURCE_DIMENSION,
        output_dimension=SOURCE_DIMENSION,
        normalize=True,
        source_paths=[path],
    )


def graph_manifest_for_dimension(dimension: int) -> dict[str, Any]:
    """Build the exact queue-local manifest for one effective input view.

    The top-level fingerprint is the trainer's row-alignment admission tuple;
    it deliberately changes with the network input preprocessing.  The graph
    itself remains authenticated as full-768d truth in the separate
    ``graph_construction_truth`` block.  Keeping both identities explicit
    avoids implying that a reduced-prefix graph was rebuilt for this round.
    """
    X = input_array(dimension)
    _, fingerprint = data_fingerprint(X)
    _, stamp = preprocessing_for_dimension(dimension)
    graph_truth_X = input_array(SOURCE_DIMENSION)
    _, graph_truth_fingerprint = data_fingerprint(graph_truth_X)
    _, graph_truth_stamp = preprocessing_for_dimension(SOURCE_DIMENSION)
    return {
        "schema": "graph_manifest.v2",
        "n_nodes": ROWS,
        "n_edges": GRAPH_EDGES,
        "source_min": 0,
        "source_max": ROWS - 1,
        "target_min": 0,
        "target_max": ROWS - 1,
        "node_namespace": "contiguous_0..n_nodes",
        "directed": True,
        "k": GRAPH_K,
        "metric": "cosine",
        "metric_input": "full_768d_source",
        "weight_semantics": "fuzzy_simplicial_set(k50)",
        "graph_path": os.path.basename(GRAPH_PATH),
        "graph_sha256": GRAPH_SHA256,
        "graph_bytes": GRAPH_BYTES,
        "data_len": ROWS,
        "data_fingerprint": fingerprint,
        "data_fingerprint_n": 2048,
        "data_shard_records": [{
            "ordinal": 0,
            "canonical_path": TRAIN_PATH,
            "bytes": TRAIN_BYTES,
            "sha256": TRAIN_SHA256,
        }],
        "input_preprocessing": stamp,
        "model_input_alignment": {
            "effective_dimension": int(dimension),
            "data_fingerprint": fingerprint,
            "input_preprocessing": stamp,
            "purpose": "trainer row-order and effective-feature admission",
        },
        "graph_construction_truth": {
            "source_path": TRAIN_PATH,
            "source_sha256": TRAIN_SHA256,
            "source_dimension": SOURCE_DIMENSION,
            "data_fingerprint": graph_truth_fingerprint,
            "input_preprocessing": graph_truth_stamp,
            "metric": "cosine",
            "shared_across_all_six_cells": True,
            "reduced_dimension_graph_rebuilt": False,
        },
        "endpoint_cosine": GRAPH_ENDPOINT_PROBE,
        "post_hoc_identity_verified": True,
        "verified_by": "round0027-queue-local-adapter-v1",
    }


def train_config_for_cell(
    label: str,
    *,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    dimension, seed = parse_cell(label)
    preprocessing, stamp = preprocessing_for_dimension(dimension)
    config: dict[str, Any] = {
        "schema": f"round0027-{label}-production-config-v1",
        "phrase": (
            f"2M jina MRL projector input {dimension}d seed {seed}; "
            "shared full-768d fuzzy graph truth"
        ),
        "row_universe": {
            "rows": ROWS,
            "source_path": TRAIN_PATH,
            "source_sha256": TRAIN_SHA256,
            "source_dimension": SOURCE_DIMENSION,
            "source_dtype": "<f2",
            "input_preprocessing": preprocessing,
        },
        "graph": {
            "path": GRAPH_PATH,
            "sha256": GRAPH_SHA256,
            "manifest_path": graph_manifest_path,
            "manifest_sha256": graph_manifest_sha256,
            "k": GRAPH_K,
            "directed_edges": GRAPH_EDGES,
            "sampling": "fuzzy-weight-proportional-with-replacement",
            "weights_consumed_for_sampling": True,
            "positive_target_mode": "binary",
            "truth_dimension": SOURCE_DIMENSION,
            "identical_across_cells": True,
        },
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": dimension,
            "hidden_dimension": 2048,
            "hidden_layers": 3,
            "output_dimension": 2,
            "use_batchnorm": False,
            "use_dropout": False,
            "low_dim_kernel": "legacy_lp",
            "a": 1.0,
            "b": 1.0,
        },
        "optimizer": {
            "seed": seed,
            "learning_rate": 0.001,
            "batch_size": 8192,
            "positive_ratio": 0.05,
            "positive_target_mode": "binary",
            "weighted_edge_sampling": True,
            "correlation_weight": 0.0,
            "clip_grad_norm": 1.0,
            "use_amp": "bf16",
            "schedule": "cosine-v3-positive-budget",
            "warmup_successful_updates": 200,
            "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
            "reject_neighbors": False,
            "anchored_init": "none",
            "midnear_enabled": False,
        },
        "execution": {
            "device_count": 1,
            "required_pipeline": "device",
            "gpu_resident_data": "auto",
            "gpu_resident_vram_budget_gb": 31.0,
            "residency": "device_fp16",
            "minimum_train_upd_s": 150.0,
            "warning_train_upd_s": 200.0,
            "canary_optimizer_updates": CANARY_UPDATES,
            "full_run_retry_count": 0,
            "expected_pipeline_stamp": {
                "pipeline": "device",
                "sampler_class": "DeviceEdgeSampler",
                "positive_sampling": "weighted_with_replacement",
                "x_residency": "device_fp16",
                "weighted_requested": True,
                "weighted_effective": True,
                "uniform_with_replacement": False,
                "positive_with_replacement": True,
                "multiplicity_policy": "row_multiplicity_uncapped",
                **stamp,
            },
        },
        "transform": {
            "source_rows": ROWS,
            "query_rows": QUERY_ROWS,
            "model_batch_rows": 8192,
            "output_dtype": "<f4",
            "output_dimension": 2,
            "input_preprocessing_sha256": stamp[
                "input_preprocessing_sha256"],
        },
        "scorer": {
            "truth_dimension": SOURCE_DIMENSION,
            "fraction": PANEL_FRACTION,
            "anchors": PANEL_ANCHORS,
            "anchor_seed": PANEL_SEED,
            "query_seed": QUERY_SEED,
            "query_rows": QUERY_ROWS,
            "query_truth_k": 10,
            "centroids": copy.deepcopy(CENTROIDS),
        },
    }
    return config, sha256_bytes(canonical_json(config))


def validate_job_cell(job: dict[str, Any]) -> dict[str, Any]:
    label = str(job.get("cell") or "")
    manifest_path = str(job.get("graph_manifest_path") or "")
    manifest_sha = str(job.get("graph_manifest_sha256") or "")
    expected, digest = train_config_for_cell(
        label,
        graph_manifest_path=manifest_path,
        graph_manifest_sha256=manifest_sha,
    )
    expected_json_payload = json.loads(canonical_json(expected))
    if (job.get("production_config") != expected_json_payload or
            job.get("production_config_sha256") != digest):
        raise ValueError(f"Round 0027 {label} production config changed")
    dimension, seed = parse_cell(label)
    return {
        "label": label,
        "dimension": dimension,
        "seed": seed,
        "train_config": expected,
        "train_config_sha256": digest,
    }


def query_row_ids() -> np.ndarray:
    rng = np.random.RandomState(QUERY_SEED)
    return np.sort(rng.choice(
        np.arange(QUERY_UNIVERSE_START, QUERY_UNIVERSE_STOP, dtype=np.int64),
        QUERY_ROWS,
        replace=False,
    )).astype(np.int64, copy=False)


def x_only_row_ceiling(dimension: int, budget_bytes: int = 31_000_000_000) -> int:
    if dimension not in DIMENSIONS:
        raise ValueError(dimension)
    return int(budget_bytes // (dimension * np.dtype("float16").itemsize))


def build_registered_decision(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the pre-registered two-seed adoption rule without I/O."""
    if set(cells) != set(CELL_LABELS):
        raise ValueError("Round 0027 decision requires exactly six registered cells")
    metrics = (
        "ffr", "purity_k256", "purity_k1024", "density",
        "oos_proj_ffr", "updates_per_s", "peak_reserved_bytes",
    )
    for label, cell in cells.items():
        dimension, seed = parse_cell(label)
        if (cell.get("dimension") != dimension or cell.get("seed") != seed or
                any(not isinstance(cell.get(metric), (int, float)) or
                    not np.isfinite(cell[metric]) for metric in metrics)):
            raise ValueError(f"Round 0027 decision cell is malformed: {label}")
    means: dict[int, dict[str, float]] = {}
    for dimension in DIMENSIONS:
        members = [cells[f"d{dimension}_s{seed}"] for seed in SEEDS]
        means[dimension] = {
            metric: float(np.mean([item[metric] for item in members]))
            for metric in metrics
        }
    control_values = [cells[f"d768_s{seed}"]["ffr"] for seed in SEEDS]
    control_spread = float(max(control_values) - min(control_values))
    qualifications = {}
    for dimension in (256, 384):
        oos_ratio = (
            means[dimension]["oos_proj_ffr"] / means[768]["oos_proj_ffr"]
            if means[768]["oos_proj_ffr"] > 0 else 0.0)
        transductive_floor = means[768]["ffr"] - control_spread
        checks = {
            "oos_at_least_90pct_control": oos_ratio >= 0.90,
            "transductive_within_control_seed_spread": (
                means[dimension]["ffr"] >= transductive_floor),
        }
        qualifications[dimension] = {
            "qualified": all(checks.values()),
            "checks": checks,
            "oos_ratio_to_768": oos_ratio,
            "transductive_ffr_floor": transductive_floor,
        }
    adopted = next(
        (dimension for dimension in (256, 384)
         if qualifications[dimension]["qualified"]),
        None,
    )
    return {
        "seed_means": {str(key): value for key, value in means.items()},
        "control_768_ffr_seed_spread_max_minus_min": control_spread,
        "qualification": {
            str(key): value for key, value in qualifications.items()},
        "decision": (
            f"adopt-{adopted}d"
            if adopted else "reject-all-reduced-dimensions"),
        "adopted_input_dimension": adopted,
    }
