"""Exact row mapping for the historical shuffled jina English ladder.

The R0037 2M training rows are a prefix of an 8M array produced by sampling
three source corpora independently, concatenating those samples, and applying
one stored permutation.  The R0087 inventory addresses the same source rows in
a corpus-contiguous 25M global universe.  This module reconstructs that mapping
without comparing floating-point values or relying on embedding similarity.

It also derives a size-preserving representative-policy selector: scan the
historical shuffled stream, reject rows excluded by the R0087 exact-duplicate
eligibility artifact, and retain the first requested number of eligible rows.
That distinction matters.  Filtering only the historical 2M prefix would
silently shrink the training set instead of replacing excluded copies.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_identity import ordered_array_sha256
from .round0087_inventory import FINEWEB, PILE, REDPAJAMA


HISTORICAL_CORPORA = (
    ("fineweb_idx", FINEWEB),
    ("rpj_idx", REDPAJAMA),
    ("pile_idx", PILE),
)
HISTORICAL_SEED = 42


class HistoricalJinaSelectionError(RuntimeError):
    """Historical and current jina row identities cannot be reconciled."""


def _int64_vector(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype != np.dtype("<i8"):
        raise HistoricalJinaSelectionError(
            f"{label} must be one little-endian int64 vector"
        )
    return array


def validate_historical_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_seed: int = HISTORICAL_SEED,
) -> dict[str, np.ndarray]:
    """Validate the exact arrays emitted by ``build_8m_sample.py``.

    The permutation check is deliberately exact.  A range check alone would
    allow repeated or missing pre-shuffle rows and make the reconstructed row
    identity ambiguous.
    """
    expected = {"seed", "perm", *(key for key, _ in HISTORICAL_CORPORA)}
    if set(provenance) != expected:
        raise HistoricalJinaSelectionError(
            "historical provenance members changed"
        )
    seed_value = np.asarray(provenance["seed"])
    if seed_value.ndim != 0 or int(seed_value) != expected_seed:
        raise HistoricalJinaSelectionError("historical shuffle seed changed")

    arrays = {
        key: _int64_vector(provenance[key], label=key)
        for key in ("perm", *(name for name, _ in HISTORICAL_CORPORA))
    }
    total = sum(len(arrays[key]) for key, _ in HISTORICAL_CORPORA)
    permutation = arrays["perm"]
    if len(permutation) != total or total <= 0:
        raise HistoricalJinaSelectionError(
            "historical sample lengths do not close"
        )
    for key, _ in HISTORICAL_CORPORA:
        rows = arrays[key]
        if (
            len(rows) == 0
            or rows[0] < 0
            or np.any(rows[1:] <= rows[:-1])
        ):
            raise HistoricalJinaSelectionError(
                f"{key} must be strictly increasing nonnegative source rows"
            )

    ordered = np.sort(permutation)
    if (
        ordered[0] != 0
        or ordered[-1] != total - 1
        or np.any(ordered[1:] != ordered[:-1] + 1)
    ):
        raise HistoricalJinaSelectionError(
            "historical perm is not an exact permutation"
        )
    return {"seed": seed_value.copy(), **arrays}


def load_historical_provenance(
    path: str | Path,
    *,
    expected_seed: int = HISTORICAL_SEED,
) -> dict[str, np.ndarray]:
    """Load and validate a historical provenance archive without pickle."""
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    return validate_historical_provenance(
        values,
        expected_seed=expected_seed,
    )


def _selection(value: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = value.get("selection", value)
    if not isinstance(selection, Mapping):
        raise HistoricalJinaSelectionError("inventory selection is not an object")
    return selection


def _validated_ranges(
    inventory_or_selection: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], int]:
    selection = _selection(inventory_or_selection)
    ranges = selection.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise HistoricalJinaSelectionError("inventory selection has no ranges")
    selected_rows = int(selection.get("selected_rows", -1))
    global_cursor = 0
    dataset_cursors: dict[str, int] = {}
    normalized: list[Mapping[str, Any]] = []
    for item in ranges:
        if not isinstance(item, Mapping):
            raise HistoricalJinaSelectionError("inventory range is not an object")
        dataset = str(item.get("dataset", ""))
        dataset_start = int(item.get("dataset_row_start", -1))
        dataset_stop = int(item.get("dataset_row_stop", -1))
        global_start = int(item.get("global_row_start", -1))
        global_stop = int(item.get("global_row_stop", -1))
        width = dataset_stop - dataset_start
        if (
            not dataset
            or dataset_start != dataset_cursors.get(dataset, 0)
            or global_start != global_cursor
            or width <= 0
            or global_stop - global_start != width
        ):
            raise HistoricalJinaSelectionError(
                "inventory ranges are not contiguous in dataset/global order"
            )
        dataset_cursors[dataset] = dataset_stop
        global_cursor = global_stop
        normalized.append(item)
    if selected_rows != global_cursor:
        raise HistoricalJinaSelectionError(
            "inventory selected-row accounting does not close"
        )
    return normalized, selected_rows


def map_dataset_rows_to_global(
    rows: Any,
    *,
    dataset: str,
    inventory_or_selection: Mapping[str, Any],
) -> np.ndarray:
    """Map original dataset row IDs to R0087 global row IDs."""
    source = np.asarray(rows, dtype=np.int64)
    if source.ndim != 1 or np.any(source < 0):
        raise HistoricalJinaSelectionError(
            "dataset rows must be one nonnegative vector"
        )
    ranges, _ = _validated_ranges(inventory_or_selection)
    selected = [item for item in ranges if item["dataset"] == dataset]
    if not selected:
        raise HistoricalJinaSelectionError(
            f"dataset is absent from inventory selection: {dataset}"
        )
    starts = np.asarray(
        [int(item["dataset_row_start"]) for item in selected],
        dtype=np.int64,
    )
    stops = np.asarray(
        [int(item["dataset_row_stop"]) for item in selected],
        dtype=np.int64,
    )
    global_starts = np.asarray(
        [int(item["global_row_start"]) for item in selected],
        dtype=np.int64,
    )
    if not len(source):
        return np.empty(0, dtype=np.int64)
    range_ids = np.searchsorted(stops, source, side="right")
    if (
        np.any(range_ids >= len(selected))
        or np.any(source < starts[range_ids])
    ):
        raise HistoricalJinaSelectionError(
            f"historical {dataset} row falls outside the R0087 selection"
        )
    return global_starts[range_ids] + source - starts[range_ids]


def map_historical_positions(
    provenance: Mapping[str, Any],
    inventory_or_selection: Mapping[str, Any],
    positions: Any,
    *,
    expected_seed: int = HISTORICAL_SEED,
) -> dict[str, np.ndarray]:
    """Map shuffled historical positions to exact source and global row IDs."""
    arrays = validate_historical_provenance(
        provenance,
        expected_seed=expected_seed,
    )
    historical_positions = np.asarray(positions, dtype=np.int64)
    if (
        historical_positions.ndim != 1
        or np.any(historical_positions < 0)
        or np.any(historical_positions >= len(arrays["perm"]))
    ):
        raise HistoricalJinaSelectionError(
            "historical positions are outside the shuffled sample"
        )
    pre_shuffle = arrays["perm"][historical_positions]
    lengths = np.asarray(
        [len(arrays[key]) for key, _ in HISTORICAL_CORPORA],
        dtype=np.int64,
    )
    boundaries = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    corpus_ids = np.searchsorted(boundaries[1:], pre_shuffle, side="right")
    global_rows = np.empty(len(pre_shuffle), dtype=np.int64)
    dataset_rows = np.empty(len(pre_shuffle), dtype=np.int64)
    for corpus_id, (key, dataset) in enumerate(HISTORICAL_CORPORA):
        mask = corpus_ids == corpus_id
        local = pre_shuffle[mask] - boundaries[corpus_id]
        source = arrays[key][local]
        dataset_rows[mask] = source
        global_rows[mask] = map_dataset_rows_to_global(
            source,
            dataset=dataset,
            inventory_or_selection=inventory_or_selection,
        )
    _, selected_rows = _validated_ranges(inventory_or_selection)
    if (
        np.any(global_rows < 0)
        or np.any(global_rows >= selected_rows)
        or len(np.unique(global_rows)) != len(global_rows)
    ):
        raise HistoricalJinaSelectionError(
            "historical-to-global mapping is not unique and in range"
        )
    return {
        "historical_positions": historical_positions,
        "pre_shuffle_positions": pre_shuffle,
        "corpus_ids": corpus_ids.astype(np.int8, copy=False),
        "dataset_rows": dataset_rows,
        "global_rows": global_rows,
    }


def _excluded_mask(rows: np.ndarray, excluded_rows: np.ndarray) -> np.ndarray:
    if not len(excluded_rows):
        return np.zeros(len(rows), dtype=np.bool_)
    indices = np.searchsorted(excluded_rows, rows)
    return (indices < len(excluded_rows)) & (
        excluded_rows[np.minimum(indices, len(excluded_rows) - 1)] == rows
    )


def derive_first_eligible_historical_rows(
    provenance: Mapping[str, Any],
    inventory_or_selection: Mapping[str, Any],
    excluded_rows: Any,
    *,
    target_rows: int,
    expected_seed: int = HISTORICAL_SEED,
) -> dict[str, Any]:
    """Return the first ``target_rows`` eligible rows in historical order.

    Rows excluded by the current 25M exact-family policy are replaced by later
    eligible rows from the same shuffled 8M stream.  The returned
    ``historical_positions`` therefore need not be a contiguous prefix, while
    their order remains exactly the historical shuffle order.
    """
    if "perm" not in provenance:
        raise HistoricalJinaSelectionError("historical provenance has no perm")
    historical_rows = len(_int64_vector(provenance["perm"], label="perm"))
    if target_rows <= 0 or target_rows > historical_rows:
        raise HistoricalJinaSelectionError("eligible target size is invalid")
    excluded = np.asarray(excluded_rows, dtype=np.int64)
    _, selected_rows = _validated_ranges(inventory_or_selection)
    if (
        excluded.ndim != 1
        or np.any(excluded < 0)
        or np.any(excluded >= selected_rows)
        or (len(excluded) and np.any(excluded[1:] <= excluded[:-1]))
    ):
        raise HistoricalJinaSelectionError(
            "excluded rows must be sorted, unique, and in range"
        )
    all_positions = np.arange(historical_rows, dtype=np.int64)
    mapped = map_historical_positions(
        provenance,
        inventory_or_selection,
        all_positions,
        expected_seed=expected_seed,
    )
    excluded_mask = _excluded_mask(mapped["global_rows"], excluded)
    eligible_positions = np.flatnonzero(~excluded_mask)
    if len(eligible_positions) < target_rows:
        raise HistoricalJinaSelectionError(
            "historical stream has too few eligible rows"
        )
    chosen = eligible_positions[:target_rows]
    output_arrays = {
        key: value[chosen]
        for key, value in mapped.items()
    }
    scan_rows = int(chosen[-1]) + 1
    raw_prefix_exclusions = int(excluded_mask[:target_rows].sum())
    corpus_counts = np.bincount(
        output_arrays["corpus_ids"],
        minlength=len(HISTORICAL_CORPORA),
    )
    raw_corpus_counts = np.bincount(
        mapped["corpus_ids"][:target_rows],
        minlength=len(HISTORICAL_CORPORA),
    )
    if (
        len(output_arrays["global_rows"]) != target_rows
        or np.any(_excluded_mask(output_arrays["global_rows"], excluded))
        or np.any(output_arrays["historical_positions"][1:]
                  <= output_arrays["historical_positions"][:-1])
    ):
        raise HistoricalJinaSelectionError(
            "eligible historical selector failed closure checks"
        )
    summary = {
        "target_rows": target_rows,
        "historical_stream_rows": historical_rows,
        "scan_rows": scan_rows,
        "skipped_excluded_rows": scan_rows - target_rows,
        "raw_prefix_excluded_rows": raw_prefix_exclusions,
        "replacement_rows_beyond_raw_prefix": int(
            np.count_nonzero(
                output_arrays["historical_positions"] >= target_rows
            )
        ),
        "corpora": [dataset for _, dataset in HISTORICAL_CORPORA],
        "raw_prefix_corpus_counts": raw_corpus_counts.tolist(),
        "eligible_selector_corpus_counts": corpus_counts.tolist(),
        "array_sha256": {
            key: ordered_array_sha256(value)
            for key, value in output_arrays.items()
        },
    }
    return {"arrays": output_arrays, "summary": summary}


def evenly_spaced_validation_positions(
    row_count: int,
    *,
    count: int,
) -> np.ndarray:
    """Select deterministic positions spanning a row stream for byte checks."""
    if row_count <= 0 or count <= 0:
        raise ValueError("row_count and count must be positive")
    return np.unique(
        np.linspace(0, row_count - 1, min(count, row_count), dtype=np.int64)
    )


def verify_embedding_rows(
    historical_embedding_path: str | Path,
    mapped: Mapping[str, np.ndarray],
    inventory_or_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify mapped rows against the source shards with exact array equality."""
    positions = np.asarray(mapped["historical_positions"], dtype=np.int64)
    global_rows = np.asarray(mapped["global_rows"], dtype=np.int64)
    if positions.ndim != 1 or global_rows.shape != positions.shape:
        raise HistoricalJinaSelectionError("mapped validation rows changed shape")
    historical = np.load(
        historical_embedding_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if historical.ndim != 2 or np.any(positions >= historical.shape[0]):
        raise HistoricalJinaSelectionError(
            "historical validation positions exceed embedding array"
        )
    ranges, selected_rows = _validated_ranges(inventory_or_selection)
    if np.any(global_rows < 0) or np.any(global_rows >= selected_rows):
        raise HistoricalJinaSelectionError("validation global row is out of range")
    stops = np.asarray(
        [int(item["global_row_stop"]) for item in ranges],
        dtype=np.int64,
    )
    range_ids = np.searchsorted(stops, global_rows, side="right")
    arrays: dict[str, np.ndarray] = {}
    mismatches: list[int] = []
    for index, (position, global_row, range_id) in enumerate(
        zip(positions.tolist(), global_rows.tolist(), range_ids.tolist())
    ):
        item = ranges[range_id]
        shard = item.get("shard")
        if not isinstance(shard, Mapping) or not isinstance(
            shard.get("canonical_path"), str
        ):
            raise HistoricalJinaSelectionError(
                "inventory range lacks a canonical shard path"
            )
        path = str(shard["canonical_path"])
        source = arrays.get(path)
        if source is None:
            source = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays[path] = source
        local = int(item["shard_row_start"]) + (
            global_row - int(item["global_row_start"])
        )
        if not np.array_equal(historical[position], source[local]):
            mismatches.append(index)
    if mismatches:
        raise HistoricalJinaSelectionError(
            f"{len(mismatches)} mapped embedding rows differ byte-for-byte"
        )
    return {
        "validated_rows": len(positions),
        "historical_positions_sha256": ordered_array_sha256(positions),
        "global_rows_sha256": ordered_array_sha256(global_rows),
        "exact_array_equal": True,
        "source_shards_opened": len(arrays),
    }
