"""Exact full-768 substrate contract for the diverse 25M Jina atlas."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0087_inventory import (
    DIMENSION,
    DTYPE as SOURCE_DTYPE,
    FINEWEB,
    PILE,
    REDPAJAMA,
    TARGET_ROWS,
)


ROUND_ID = "0103"
INVENTORY_PATH = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-inventory-v1.json"
)
INVENTORY_SHA256 = (
    "364aaa2f7a5e886f9cacdb96d3ffef1bbe697148e1babe10eee7817af0fc7163"
)
INVENTORY_IDENTITY = (
    "6c73f781208d16a84fc9e619e66c89f8fe56375dad77f7eda75e795f85cfec9b"
)
ELIGIBILITY_PATH = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-eligibility-v1.npz"
)
ELIGIBILITY_SHA256 = (
    "11a9c197f0e20cb1e5d6968bc9ec3a9e2c89fa66c711d252f864959016eac274"
)
RETAINED_ROWS = 24_948_663
EXCLUDED_ROWS = 51_337
OUTPUT_DTYPE = np.dtype("int8")
SCALE_DTYPE = np.dtype("<f2")
SAMPLE_SEED = 103
SAMPLE_ROWS = 10_000
RECONSTRUCTION_COSINE_P01_FLOOR = 0.999
SUBSTRATE_SCHEMA = "jina-diverse-25m-full768-int8-substrate-v1"
ENGLISH_DATASETS = (FINEWEB, REDPAJAMA, PILE)


class Round0103Error(RuntimeError):
    """The registered diverse-Jina substrate contract changed."""


def validate_inventory(
    path: str = INVENTORY_PATH,
    *,
    expected_sha256: str = INVENTORY_SHA256,
) -> dict[str, Any]:
    """Authenticate the accepted R0087 inventory and selection geometry."""
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0103Error("R0087 inventory bytes changed")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise Round0103Error("R0087 inventory is not an object")
    body = {
        key: value
        for key, value in manifest.items()
        if key != "identity_sha256"
    }
    selection = manifest.get("selection") or {}
    duplicate = manifest.get("duplicate_control") or {}
    summary = duplicate.get("summary") or {}
    eligibility = duplicate.get("eligibility") or {}
    if (
        manifest.get("schema") != "jina-diverse-25m-inventory-v1"
        or manifest.get("round_id") != "0087"
        or manifest.get("identity_sha256") != INVENTORY_IDENTITY
        or manifest.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or manifest.get("embedding_prompt") != "raw"
        or manifest.get("capability_ready") is not True
        or selection.get("complete") is not True
        or int(selection.get("selected_rows", -1)) != TARGET_ROWS
        or int(summary.get("row_count", -1)) != TARGET_ROWS
        or int(summary.get("retained_row_count", -1)) != RETAINED_ROWS
        or int(summary.get("excluded_row_count", -1)) != EXCLUDED_ROWS
        or int(summary.get("zero_row_count", -1)) != 0
        or int(summary.get("nonfinite_row_count", -1)) != 0
        or int(summary.get("fingerprint_collision_splits", -1)) != 0
        or eligibility.get("sha256") != ELIGIBILITY_SHA256
        or eligibility.get("canonical_path") != ELIGIBILITY_PATH
    ):
        raise Round0103Error("R0087 inventory contract changed")
    ranges = list(selection.get("ranges") or [])
    source_order = list(selection.get("source_order") or [])
    if (
        len(source_order) != 22
        or source_order[:3] != list(ENGLISH_DATASETS)
        or len(set(source_order)) != len(source_order)
        or not ranges
    ):
        raise Round0103Error("R0087 source order changed")
    cursor = 0
    observed_order: list[str] = []
    for item in ranges:
        dataset = str(item.get("dataset", ""))
        start = int(item.get("global_row_start", -1))
        stop = int(item.get("global_row_stop", -1))
        shard_start = int(item.get("shard_row_start", -1))
        shard_stop = int(item.get("shard_row_stop", -1))
        shard = item.get("shard") or {}
        if (
            dataset not in source_order
            or start != cursor
            or stop <= start
            or shard_start < 0
            or shard_stop - shard_start != stop - start
            or not shard.get("canonical_path")
            or not shard.get("sha256")
            or int(shard.get("bytes", -1)) <= 0
            or int(shard.get("rows", -1)) < shard_stop
        ):
            raise Round0103Error("R0087 selected range is malformed")
        if not observed_order or observed_order[-1] != dataset:
            observed_order.append(dataset)
        cursor = stop
    if cursor != TARGET_ROWS or observed_order != source_order:
        raise Round0103Error("R0087 selected ranges do not close in order")
    return {
        "manifest": manifest,
        "signature": signature,
        "selection": selection,
        "eligibility": eligibility,
    }


def row_scales(values: np.ndarray) -> np.ndarray:
    """Compute the exact stored row-local fp16 symmetric scale."""
    source = np.asarray(values)
    if (
        source.ndim != 2
        or source.dtype != SOURCE_DTYPE
        or source.shape[1] <= 0
    ):
        raise Round0103Error("source block is not a 2D little-endian fp16 array")
    maximum = np.max(np.abs(source.astype(np.float32)), axis=1)
    scales = (maximum / 127.0).astype(SCALE_DTYPE)
    if (
        not np.isfinite(maximum).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise Round0103Error("source block contains a nonfinite or zero row")
    return scales


def quantize_block(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise symmetric int8 quantization using the stored fp16 scale."""
    source = np.asarray(values)
    scales = row_scales(source)
    work = source.astype(np.float32)
    stored = scales.astype(np.float32)
    encoded = np.rint(work / stored[:, None])
    np.clip(encoded, -127.0, 127.0, out=encoded)
    return encoded.astype(OUTPUT_DTYPE), scales


def retained_sample_rows(
    excluded_rows: np.ndarray,
    *,
    row_count: int = TARGET_ROWS,
    sample_count: int = SAMPLE_ROWS,
    seed: int = SAMPLE_SEED,
) -> np.ndarray:
    """Draw ordered retained rows without materializing a 25M mask."""
    excluded = np.asarray(excluded_rows, dtype=np.int64)
    if (
        excluded.ndim != 1
        or len(excluded) >= row_count
        or (len(excluded) and (
            excluded[0] < 0
            or excluded[-1] >= row_count
            or np.any(excluded[1:] <= excluded[:-1])
        ))
        or sample_count <= 0
        or sample_count > row_count - len(excluded)
    ):
        raise Round0103Error("eligibility selector is malformed")
    retained_count = row_count - len(excluded)
    compact = np.sort(
        np.random.default_rng(seed).choice(
            retained_count,
            size=sample_count,
            replace=False,
        ).astype(np.int64)
    )
    global_rows = compact.copy()
    while True:
        shifted = compact + np.searchsorted(
            excluded,
            global_rows,
            side="right",
        )
        if np.array_equal(shifted, global_rows):
            break
        global_rows = shifted
    if (
        np.any(global_rows >= row_count)
        or np.intersect1d(global_rows, excluded).size
        or np.any(global_rows[1:] <= global_rows[:-1])
    ):
        raise Round0103Error("retained sample mapping failed")
    return global_rows


def build_label_arrays(
    selection: Mapping[str, Any],
    *,
    row_count: int = TARGET_ROWS,
) -> dict[str, Any]:
    """Materialize compact dataset, English-corpus, and language identities."""
    source_order = [str(value) for value in selection.get("source_order") or []]
    ranges = list(selection.get("ranges") or [])
    if (
        not source_order
        or len(source_order) >= 255
        or source_order[:3] != list(ENGLISH_DATASETS)
    ):
        raise Round0103Error("label source order is malformed")
    dataset_code = {name: index for index, name in enumerate(source_order)}
    languages_by_dataset: dict[str, str] = {}
    for item in ranges:
        dataset = str(item.get("dataset", ""))
        language = item.get("language")
        label = "eng_Latn" if dataset in ENGLISH_DATASETS else str(language)
        if not label or label == "None":
            raise Round0103Error("language label is missing")
        previous = languages_by_dataset.setdefault(dataset, label)
        if previous != label:
            raise Round0103Error("dataset language label changed across ranges")
    language_labels = ["eng_Latn"] + [
        languages_by_dataset[name]
        for name in source_order
        if name not in ENGLISH_DATASETS
    ]
    if len(set(language_labels)) != len(language_labels):
        raise Round0103Error("language labels are not unique")
    language_code = {
        label: index
        for index, label in enumerate(language_labels)
    }
    dataset_ids = np.full(row_count, 255, dtype=np.uint8)
    english_corpus_ids = np.full(row_count, 255, dtype=np.uint8)
    language_ids = np.full(row_count, 255, dtype=np.uint8)
    dataset_counts = {label: 0 for label in source_order}
    english_counts = {
        label: 0
        for label in ["not-english-source", *ENGLISH_DATASETS]
    }
    language_counts = {label: 0 for label in language_labels}
    cursor = 0
    for item in ranges:
        start = int(item.get("global_row_start", -1))
        stop = int(item.get("global_row_stop", -1))
        dataset = str(item.get("dataset", ""))
        if start != cursor or stop <= start or dataset not in dataset_code:
            raise Round0103Error("label ranges do not close contiguously")
        dataset_ids[start:stop] = dataset_code[dataset]
        english_corpus_ids[start:stop] = (
            ENGLISH_DATASETS.index(dataset) + 1
            if dataset in ENGLISH_DATASETS
            else 0
        )
        language_ids[start:stop] = language_code[
            languages_by_dataset[dataset]
        ]
        count = stop - start
        dataset_counts[dataset] += count
        english_label = (
            dataset
            if dataset in ENGLISH_DATASETS
            else "not-english-source"
        )
        english_counts[english_label] += count
        language_counts[languages_by_dataset[dataset]] += count
        cursor = stop
    if (
        cursor != row_count
        or np.any(dataset_ids == 255)
        or np.any(english_corpus_ids == 255)
        or np.any(language_ids == 255)
    ):
        raise Round0103Error("label arrays are incomplete")
    return {
        "arrays": {
            "dataset_id": dataset_ids,
            "english_corpus_id": english_corpus_ids,
            "language_id": language_ids,
        },
        "vocabulary": {
            "dataset": source_order,
            "english_corpus": [
                "not-english-source",
                *ENGLISH_DATASETS,
            ],
            "language": language_labels,
        },
        "counts": {
            "dataset": dataset_counts,
            "english_corpus": english_counts,
            "language": language_counts,
        },
    }
