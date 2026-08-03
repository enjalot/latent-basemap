"""Execute R0168's CPU-only prompted U12 staging and duplicate census."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.jina_historical_selection import (
    IndexedInventoryFp16Array,
    materialize_indexed_fp16_npy,
)
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0087_inventory import duplicate_census
from basemap.round0103_substrate import validate_inventory
from basemap.round0168_prompted_diverse_staging import (
    CAPABILITY,
    DIMENSION,
    DTYPE,
    MANIFEST_SCHEMA,
    ROUND_ID,
    U12_ROWS,
    Round0168Error,
    prompted_selection,
)


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def _read_sealed(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = str(expected.get("canonical_path") or "")
    if not path or expected_input_signature(path) != dict(expected):
        raise Round0168Error(f"{label} bytes changed")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if not isinstance(value, dict) or value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0168Error(f"{label} identity seal changed")
    return value


def _verify_mapping(expected: Mapping[str, Any]) -> np.ndarray:
    path = str(expected.get("canonical_path") or "")
    if not path or expected_input_signature(path) != dict(expected):
        raise Round0168Error("accepted R0132 U12 mapping bytes changed")
    mapping = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        mapping.dtype != np.dtype("<i8")
        or mapping.shape != (U12_ROWS,)
        or mapping[0] < 0
        or mapping[-1] >= 25_000_000
        or np.any(mapping[1:] <= mapping[:-1])
    ):
        raise Round0168Error("accepted R0132 U12 mapping geometry changed")
    return mapping


def run_staging(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0168Error("R0168 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0168Error("R0168 is CPU-only")
    started = time.monotonic()
    inventory = validate_inventory()
    if inventory["signature"] != dict(job["r0087_inventory"]):
        raise Round0168Error("R0168 accepted R0087 inventory binding changed")
    manifests = {
        str(round_id): _read_sealed(signature, label=f"accepted R{round_id} prompted tranche")
        for round_id, signature in job["prompted_manifests"].items()
    }
    selection = prompted_selection(inventory["manifest"], manifests)
    if selection["ordered_selection_sha256"] != job.get("ordered_selection_sha256"):
        raise Round0168Error("R0168 prompted layout changed after issuance")
    mapping = _verify_mapping(job["u12_mapping"])
    if ordered_array_sha256(mapping) != str(job.get("u12_ordered_array_sha256") or ""):
        raise Round0168Error("R0168 U12 ordered row identity changed after issuance")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0168 prompted U12 staging")
    stage_path = os.path.join(output, "prompted-u12.f16.npy")
    source = IndexedInventoryFp16Array(mapping, selection, dimension=DIMENSION)
    staged = materialize_indexed_fp16_npy(stage_path, source, block_rows=65_536)
    values = np.load(stage_path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.dtype(DTYPE) or values.shape != (U12_ROWS, DIMENSION) or not values.flags.c_contiguous:
        raise Round0168Error("materialized prompted U12 geometry changed")

    census_selection = {
        "selected_rows": U12_ROWS,
        "ranges": [{
            "dataset": "r0132-u12-prompted",
            "dataset_row_start": 0,
            "dataset_row_stop": U12_ROWS,
            "global_row_start": 0,
            "global_row_stop": U12_ROWS,
            "shard": {**staged, "rows": U12_ROWS},
            "shard_row_start": 0,
            "shard_row_stop": U12_ROWS,
        }],
    }
    census_started = time.monotonic()
    census = duplicate_census(census_selection)
    census_wall = time.monotonic() - census_started
    census_path = os.path.join(output, "prompted-u12-exact-family-census.npz")
    atomic_save_new_npz(census_path, immutable=True, **census["arrays"])
    census_signature = expected_input_signature(census_path)

    sample_positions = np.unique(np.linspace(0, U12_ROWS - 1, 4096, dtype=np.int64))
    sample = np.asarray(values[sample_positions], dtype=np.float32)
    maximum_norm_error = float(np.max(np.abs(np.linalg.norm(sample, axis=1) - 1.0)))
    if not np.isfinite(sample).all() or maximum_norm_error > 0.002:
        raise Round0168Error("materialized prompted U12 normalization changed")

    receipt = _seal({
        "schema": MANIFEST_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "rows": U12_ROWS,
        "dimension": DIMENSION,
        "dtype": DTYPE,
        "embedding_convention": "Document: ",
        "population": {
            "law": "exact accepted R0132 U12 global rows, ascending",
            "mapping": dict(job["u12_mapping"]),
            "ordered_array_sha256": ordered_array_sha256(mapping),
            "exact_r0132_population_match": True,
            "polish_held_out": True,
        },
        "prompted_layout": {
            "r0087_inventory": dict(job["r0087_inventory"]),
            "ordered_selection_sha256": selection["ordered_selection_sha256"],
            "source_order": selection["source_order"],
            "source_chunk_count": len(selection["ranges"]),
            "heldout_polish": selection["heldout_polish"],
            "exact_dataset_row_identity": True,
        },
        "host_fp16": staged,
        "materialization": {
            "contiguous": True,
            "immutable": True,
            "logical_order": "R0132 compact row order / ascending R0087 global row",
            "source_files_opened": len(source.segments),
            "sample_rows": len(sample_positions),
            "sample_all_finite": True,
            "sample_maximum_norm_absolute_error": maximum_norm_error,
        },
        "duplicate_control": {
            "policy": "diagnostic census only; exact U12 population is not altered",
            "summary": census["summary"],
            "arrays": census_signature,
            "census_wall_seconds": census_wall,
        },
        "training_performed": False,
        "graph_built": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "prompted-u12-manifest.json"), receipt, immutable=True)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "stage_prompted_diverse_u12":
        raise Round0168Error("unknown R0168 action")
    run_staging(active, job)
