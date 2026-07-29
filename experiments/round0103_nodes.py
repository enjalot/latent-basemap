"""Stage the exact full-768 int8 substrate for the diverse 25M Jina atlas."""
from __future__ import annotations

import os
import resource
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
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0087_inventory import (
    DIMENSION,
    DTYPE as SOURCE_DTYPE,
    TARGET_ROWS,
    drop_file_cache,
)
from basemap.round0103_substrate import (
    ELIGIBILITY_PATH,
    ELIGIBILITY_SHA256,
    EXCLUDED_ROWS,
    INVENTORY_IDENTITY,
    INVENTORY_PATH,
    INVENTORY_SHA256,
    OUTPUT_DTYPE,
    RECONSTRUCTION_COSINE_P01_FLOOR,
    RETAINED_ROWS,
    ROUND_ID,
    Round0103Error,
    SAMPLE_ROWS,
    SAMPLE_SEED,
    SCALE_DTYPE,
    SUBSTRATE_SCHEMA,
    build_label_arrays,
    retained_sample_rows,
    row_scales,
    validate_inventory,
)


BLOCK_ROWS = 32_768


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {
        **value,
        "identity_sha256": sha256_bytes(canonical_json(value)),
    }


def _load_eligibility() -> dict[str, Any]:
    signature = expected_input_signature(ELIGIBILITY_PATH)
    if signature["sha256"] != ELIGIBILITY_SHA256:
        raise Round0103Error("R0087 eligibility bytes changed")
    with np.load(ELIGIBILITY_PATH, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name], dtype=np.int64)
            for name in archive.files
        }
    expected = {
        "zero_rows",
        "nonfinite_rows",
        "excluded_rows",
        "duplicate_excluded_rows",
        "duplicate_representative_rows",
        "representative_rows",
        "family_counts",
        "family_offsets",
        "member_rows",
    }
    excluded = arrays.get("excluded_rows", np.empty(0, dtype=np.int64))
    if (
        set(arrays) != expected
        or len(arrays["zero_rows"]) != 0
        or len(arrays["nonfinite_rows"]) != 0
        or len(excluded) != EXCLUDED_ROWS
        or len(excluded) and (
            excluded[0] < 0
            or excluded[-1] >= TARGET_ROWS
            or np.any(excluded[1:] <= excluded[:-1])
        )
        or TARGET_ROWS - len(excluded) != RETAINED_ROWS
        or not np.array_equal(
            arrays["duplicate_excluded_rows"],
            excluded,
        )
    ):
        raise Round0103Error("R0087 eligibility selector changed")
    return {
        "signature": signature,
        "arrays": arrays,
    }


def _unique_shards(
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    shards: dict[str, dict[str, Any]] = {}
    for item in selection["ranges"]:
        shard = dict(item["shard"])
        path = os.path.realpath(str(shard["canonical_path"]))
        normalized = {
            "canonical_path": path,
            "kind": "file",
            "bytes": int(shard["bytes"]),
            "sha256": str(shard["sha256"]),
            "rows": int(shard["rows"]),
        }
        previous = shards.setdefault(path, normalized)
        if previous != normalized:
            raise Round0103Error("one source shard has conflicting inventory rows")
    return list(shards.values())


def _verify_source_shards(
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for shard in _unique_shards(selection):
        path = shard["canonical_path"]
        signature = expected_input_signature(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            signature["sha256"] != shard["sha256"]
            or signature["bytes"] != shard["bytes"]
            or array.ndim != 2
            or array.shape != (shard["rows"], DIMENSION)
            or array.dtype != SOURCE_DTYPE
            or not array.flags.c_contiguous
        ):
            raise Round0103Error(f"source shard changed: {path}")
        verified.append({
            **signature,
            "rows": shard["rows"],
            "dimension": DIMENSION,
            "dtype": SOURCE_DTYPE.str,
        })
        del array
    return verified


def _range_blocks(item: Mapping[str, Any]):
    global_start = int(item["global_row_start"])
    global_stop = int(item["global_row_stop"])
    shard_start = int(item["shard_row_start"])
    for output_start in range(global_start, global_stop, BLOCK_ROWS):
        output_stop = min(output_start + BLOCK_ROWS, global_stop)
        local_start = shard_start + (output_start - global_start)
        local_stop = local_start + (output_stop - output_start)
        yield output_start, output_stop, local_start, local_stop


def _write_scales(
    path: str,
    selection: Mapping[str, Any],
) -> None:
    output = np.memmap(
        path,
        mode="w+",
        dtype=SCALE_DTYPE,
        shape=(TARGET_ROWS,),
    )
    for item in selection["ranges"]:
        source_path = str(item["shard"]["canonical_path"])
        source = np.load(source_path, mmap_mode="r", allow_pickle=False)
        for start, stop, local_start, local_stop in _range_blocks(item):
            values = np.asarray(
                source[local_start:local_stop],
                dtype=SOURCE_DTYPE,
            )
            scales = row_scales(values)
            output[start:stop] = scales
        del source
    output.flush()
    del output


def _write_int8(
    path: str,
    selection: Mapping[str, Any],
    scales_path: str,
) -> None:
    output = np.memmap(
        path,
        mode="w+",
        dtype=OUTPUT_DTYPE,
        shape=(TARGET_ROWS, DIMENSION),
    )
    stored_scales = np.memmap(
        scales_path,
        mode="r",
        dtype=SCALE_DTYPE,
        shape=(TARGET_ROWS,),
    )
    for item in selection["ranges"]:
        source_path = str(item["shard"]["canonical_path"])
        source = np.load(source_path, mmap_mode="r", allow_pickle=False)
        for start, stop, local_start, local_stop in _range_blocks(item):
            values = np.asarray(
                source[local_start:local_stop],
                dtype=np.float32,
            )
            scales = np.asarray(
                stored_scales[start:stop],
                dtype=np.float32,
            )
            encoded = np.rint(values / scales[:, None])
            np.clip(encoded, -127.0, 127.0, out=encoded)
            output[start:stop] = encoded.astype(OUTPUT_DTYPE)
        del source
    output.flush()
    del output
    del stored_scales


def _source_sample(
    selection: Mapping[str, Any],
    rows: np.ndarray,
) -> np.ndarray:
    values = np.empty((len(rows), DIMENSION), dtype=SOURCE_DTYPE)
    filled = np.zeros(len(rows), dtype=bool)
    for item in selection["ranges"]:
        start = int(item["global_row_start"])
        stop = int(item["global_row_stop"])
        left = int(np.searchsorted(rows, start, side="left"))
        right = int(np.searchsorted(rows, stop, side="left"))
        if right <= left:
            continue
        source_path = str(item["shard"]["canonical_path"])
        source = np.load(source_path, mmap_mode="r", allow_pickle=False)
        local = (
            int(item["shard_row_start"])
            + rows[left:right]
            - start
        )
        values[left:right] = source[local]
        filled[left:right] = True
        del source
    if not filled.all():
        raise Round0103Error("reconstruction sample did not map to source rows")
    return values


def _quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p01", "p50", "p99", "max"),
            np.quantile(array, (0.0, 0.01, 0.5, 0.99, 1.0)),
            strict=True,
        )
    }


def _reconstruction_sample(
    *,
    selection: Mapping[str, Any],
    excluded_rows: np.ndarray,
    int8_path: str,
    scales_path: str,
    output_path: str,
) -> dict[str, Any]:
    rows = retained_sample_rows(excluded_rows)
    source = _source_sample(selection, rows).astype(np.float32)
    encoded = np.memmap(
        int8_path,
        mode="r",
        dtype=OUTPUT_DTYPE,
        shape=(TARGET_ROWS, DIMENSION),
    )
    scales = np.memmap(
        scales_path,
        mode="r",
        dtype=SCALE_DTYPE,
        shape=(TARGET_ROWS,),
    )
    reconstructed = (
        np.asarray(encoded[rows], dtype=np.float32)
        * np.asarray(scales[rows], dtype=np.float32)[:, None]
    )
    source_norm = np.linalg.norm(source, axis=1)
    reconstructed_norm = np.linalg.norm(reconstructed, axis=1)
    cosine = np.einsum(
        "ij,ij->i",
        source,
        reconstructed,
        dtype=np.float64,
    ) / (
        source_norm.astype(np.float64)
        * reconstructed_norm.astype(np.float64)
    )
    maximum_component_error = np.max(
        np.abs(source - reconstructed),
        axis=1,
    )
    if (
        len(rows) != SAMPLE_ROWS
        or not np.isfinite(cosine).all()
        or not np.isfinite(maximum_component_error).all()
        or float(np.quantile(cosine, 0.01))
        < RECONSTRUCTION_COSINE_P01_FLOOR
    ):
        raise Round0103Error("int8 reconstruction guard failed")
    atomic_save_new_npz(
        output_path,
        immutable=True,
        compressed=False,
        sample_rows=rows,
        source_norm=source_norm.astype("<f4"),
        dequantized_norm=reconstructed_norm.astype("<f4"),
        cosine=cosine.astype("<f8"),
        maximum_component_error=maximum_component_error.astype("<f4"),
    )
    del encoded
    del scales
    return {
        "seed": SAMPLE_SEED,
        "sample_rows": SAMPLE_ROWS,
        "ordered_sample_rows_sha256": ordered_array_sha256(rows),
        "source_norm": _quantiles(source_norm),
        "dequantized_norm": _quantiles(reconstructed_norm),
        "cosine": _quantiles(cosine),
        "maximum_component_error": _quantiles(maximum_component_error),
        "cosine_p01_floor": RECONSTRUCTION_COSINE_P01_FLOOR,
        "passed": True,
        "arrays": expected_input_signature(output_path),
    }


def run_stage(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0103 full-768 diverse-Jina int8 substrate",
    )
    started = time.monotonic()
    inventory = validate_inventory(
        str(job["inventory"]),
        expected_sha256=str(job["inventory_sha256"]),
    )
    if inventory["manifest"].get("identity_sha256") != INVENTORY_IDENTITY:
        raise Round0103Error("R0103 inventory identity changed")
    selection = inventory["selection"]
    eligibility = _load_eligibility()
    verified_shards = _verify_source_shards(selection)

    scales_path = os.path.join(output, "scales.f16")
    atomic_build_new_file(
        scales_path,
        lambda path: _write_scales(path, selection),
        immutable=True,
    )
    int8_path = os.path.join(output, "embeddings.i8")
    atomic_build_new_file(
        int8_path,
        lambda path: _write_int8(path, selection, scales_path),
        immutable=True,
    )
    labels = build_label_arrays(selection)
    labels_path = os.path.join(output, "labels.npz")
    atomic_save_new_npz(
        labels_path,
        immutable=True,
        compressed=False,
        **labels["arrays"],
    )
    sample_path = os.path.join(output, "reconstruction-sample.npz")
    sample = _reconstruction_sample(
        selection=selection,
        excluded_rows=eligibility["arrays"]["excluded_rows"],
        int8_path=int8_path,
        scales_path=scales_path,
        output_path=sample_path,
    )
    for shard in verified_shards:
        drop_file_cache(str(shard["canonical_path"]))
    int8 = expected_input_signature(int8_path)
    scales = expected_input_signature(scales_path)
    label_signature = expected_input_signature(labels_path)
    if (
        int8["bytes"] != TARGET_ROWS * DIMENSION
        or scales["bytes"] != TARGET_ROWS * SCALE_DTYPE.itemsize
    ):
        raise Round0103Error("staged payload byte counts do not close")
    selection_identity = sha256_bytes(canonical_json({
        "source_order": selection["source_order"],
        "budgets": selection["budgets"],
        "ranges": selection["ranges"],
        "row_order": selection["row_order"],
    }))
    body = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "row_count": TARGET_ROWS,
        "dimension": DIMENSION,
        "embedding_prompt": "raw",
        "source_dtype": SOURCE_DTYPE.str,
        "output_dtype": OUTPUT_DTYPE.str,
        "scale_dtype": SCALE_DTYPE.str,
        "inventory": inventory["signature"],
        "inventory_identity_sha256": INVENTORY_IDENTITY,
        "selection_identity_sha256": selection_identity,
        "source_order": selection["source_order"],
        "source_ranges": len(selection["ranges"]),
        "source_shards_reverified": verified_shards,
        "source_fp16_reference": (
            "exact R0087 selected ranges in the reverified immutable shard "
            "hash table; source values are not copied or renormalized"
        ),
        "eligibility": eligibility["signature"],
        "duplicate_control": {
            "retained_row_count": RETAINED_ROWS,
            "excluded_row_count": EXCLUDED_ROWS,
            "selector_namespace": "R0087 contiguous diverse-25M global order",
        },
        "labels": {
            "arrays": label_signature,
            "dtype": "|u1",
            "shape": [TARGET_ROWS],
            "vocabulary": labels["vocabulary"],
            "counts": labels["counts"],
        },
        "quantization": {
            "method": (
                "row-local symmetric signed int8; stored fp16 scale is "
                "max(abs(source))/127 and is used for rounding"
            ),
            "normalization_applied": False,
            "dimension_truncated": False,
            "row_reordered": False,
            "prompt_applied": False,
        },
        "outputs": {
            "int8": int8,
            "scales": scales,
            "labels": label_signature,
            "reconstruction_sample": sample["arrays"],
        },
        "reconstruction": sample,
        "training_performed": False,
        "optimizer_updates": 0,
        "gpu_used": False,
        "performance": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    manifest = _seal(body)
    manifest_path = os.path.join(
        output,
        "jina-diverse-25m-full768-int8-substrate-v1.json",
    )
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    return {
        **manifest,
        "receipt": expected_input_signature(manifest_path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0103Error("R0103 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if (
        selected.get("action") != "stage_full768_int8"
        or len(selected.get("outputs") or []) != 1
    ):
        raise Round0103Error("R0103 accepts one substrate stage job")
    return run_stage(active, selected)
