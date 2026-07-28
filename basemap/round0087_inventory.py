"""Deterministic inventory and selection contract for the diverse jina atlas."""
from __future__ import annotations

import bisect
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numba import njit, prange

from .artifact_identity import expected_input_signature


ROUND_ID = "0087"
DIMENSION = 768
DTYPE = np.dtype("<f2")
TARGET_ROWS = 25_000_000
EMBEDDING_ROOT = Path("/data/embeddings")
CATALOG_PATH = Path("/data/catalog.json")
FINEWEB = "fineweb-edu-sample-10BT-chunked-500-jina-v5-nano"
REDPAJAMA = "RedPajama-Data-V2-sample-10B-chunked-500-jina-v5-nano"
PILE = "pile-uncopyrighted-chunked-500-jina-v5-nano"
HELDOUT = "fineweb-edu-sample-10BT-chunked-500-jina-v5-nano-heldout"
POLISH = "fineweb2-pol_Latn-chunked-500-jina-v5-nano"
ENGLISH_BUDGETS = {
    FINEWEB: 2_890_362,
    REDPAJAMA: 2_836_978,
    PILE: 3_399_036,
}
LANGUAGE_PREFIX = "fineweb2-"
LANGUAGE_SUFFIX = "-chunked-500-jina-v5-nano"
MULTILINGUAL_ROWS = TARGET_ROWS - sum(ENGLISH_BUDGETS.values())
MULTILINGUAL_LANGUAGE_COUNT = 19
MULTILINGUAL_BASE, MULTILINGUAL_REMAINDER = divmod(
    MULTILINGUAL_ROWS, MULTILINGUAL_LANGUAGE_COUNT
)
FP_DTYPE = np.dtype([("h0", "<u8"), ("h1", "<u8"), ("row", "<u4")])


class Round0087Error(RuntimeError):
    """The diverse inventory differs from the registered contract."""


def drop_file_cache(path: str) -> None:
    """Best-effort eviction of this scan's sequential pages.

    R0087 is allowed to share the machine with a GPU graph build.  Its input
    bytes are not reused after each inventory/fingerprint pass, while the graph
    worker depends on a hot 150M rerank substrate.  Dropping only R0087's pages
    prevents this sequential inventory from needlessly displacing that working
    set.  Correctness never depends on advisory-cache support.
    """
    if not hasattr(os, "posix_fadvise") or not hasattr(
        os, "POSIX_FADV_DONTNEED"
    ):
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def language_code(dataset: str) -> str | None:
    if dataset.startswith(LANGUAGE_PREFIX) and dataset.endswith(
        LANGUAGE_SUFFIX
    ):
        return dataset[
            len(LANGUAGE_PREFIX):-len(LANGUAGE_SUFFIX)
        ]
    return None


def discover_dataset_paths(
    root: Path = EMBEDDING_ROOT,
) -> dict[str, Path]:
    paths = {
        path.name: path
        for path in root.glob("*-jina-v5-nano*")
        if path.is_dir()
    }
    dadabase = root / "dadabase" / "jina-v5-nano.npy"
    if dadabase.is_file():
        paths["dadabase-jina-v5-nano-probe"] = dadabase
    return dict(sorted(paths.items()))


def discover_inventory_files(
    root: Path = EMBEDDING_ROOT,
) -> list[str]:
    files: list[str] = []
    for path in discover_dataset_paths(root).values():
        if path.is_file():
            files.append(str(path.resolve()))
            continue
        files.extend(
            str(item.resolve())
            for item in path.rglob("*")
            if item.is_file()
            and (item.suffix == ".npy" or item.name == "manifest.json")
        )
    return sorted(set(files))


def inspect_shard(path: str) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        array.ndim != 2
        or array.shape[1] != DIMENSION
        or array.dtype != DTYPE
        or not array.flags.c_contiguous
    ):
        raise Round0087Error(
            f"{path} is not a contiguous fp16 {DIMENSION}d shard"
        )
    signature = expected_input_signature(path)
    expected_bytes = int(array.shape[0]) * DIMENSION * DTYPE.itemsize
    payload_offset = int(getattr(array, "offset", -1))
    if (
        payload_offset <= 0
        or signature["bytes"] != payload_offset + expected_bytes
    ):
        raise Round0087Error(
            f"{path} has trailing or incomplete bytes outside its NumPy payload"
        )
    result = {
        **signature,
        "rows": int(array.shape[0]),
        "dimension": DIMENSION,
        "dtype": DTYPE.str,
    }
    del array
    drop_file_cache(path)
    return result


def _invalid_shard(path: Path, error: BaseException) -> dict[str, Any]:
    record: dict[str, Any] = {
        "canonical_path": str(path.resolve()),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    try:
        record["file"] = expected_input_signature(str(path))
    except (OSError, ValueError) as signature_error:
        record["file"] = None
        record["signature_error"] = str(signature_error)
    return record


def inventory_datasets(
    root: Path = EMBEDDING_ROOT,
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for name, path in discover_dataset_paths(root).items():
        if path.is_file():
            shard_paths = [path]
            auxiliary: list[dict[str, Any]] = []
        else:
            shard_paths = sorted(path.rglob("*.npy"))
            auxiliary = [
                expected_input_signature(str(item))
                for item in sorted(path.rglob("manifest.json"))
            ]
        shards: list[dict[str, Any]] = []
        invalid_shards: list[dict[str, Any]] = []
        for item in shard_paths:
            try:
                shards.append(inspect_shard(str(item)))
            except (OSError, ValueError, EOFError, Round0087Error) as error:
                invalid_shards.append(_invalid_shard(item, error))
        inventory[name] = {
            "dataset": name,
            "root": str(path.resolve()),
            "shards": shards,
            "invalid_shards": invalid_shards,
            "enumerated_shard_count": len(shard_paths),
            "auxiliary_manifests": auxiliary,
            "rows": sum(int(item["rows"]) for item in shards),
            "bytes": sum(int(item["bytes"]) for item in shards),
            "dimension": DIMENSION,
            "dtype": DTYPE.str,
            "language": language_code(name),
            "role": (
                "heldout-probe"
                if name == HELDOUT
                else "ood-probe"
                if name == "dadabase-jina-v5-nano-probe"
                else "heldout-language-probe"
                if name == POLISH
                else "training-candidate"
            ),
        }
    return inventory


def registered_budgets(inventory: Mapping[str, Any]) -> dict[str, int]:
    languages = sorted(
        name
        for name in inventory
        if language_code(name) is not None and name != POLISH
    )
    if len(languages) != MULTILINGUAL_LANGUAGE_COUNT:
        raise Round0087Error(
            f"expected 19 in-mix languages, found {len(languages)}"
        )
    budgets = dict(ENGLISH_BUDGETS)
    for index, name in enumerate(languages):
        budgets[name] = (
            MULTILINGUAL_BASE
            + (1 if index < MULTILINGUAL_REMAINDER else 0)
        )
    if sum(budgets.values()) != TARGET_ROWS:
        raise Round0087Error("registered diverse budget does not total 25M")
    return budgets


def build_selection(
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    budgets = registered_budgets(inventory)
    ranges: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    global_row = 0
    for dataset, budget in budgets.items():
        available = int(inventory.get(dataset, {}).get("rows", 0))
        take = min(budget, available)
        remaining = take
        dataset_row = 0
        for shard in inventory.get(dataset, {}).get("shards", []):
            if remaining <= 0:
                break
            shard_rows = int(shard["rows"])
            count = min(remaining, shard_rows)
            ranges.append({
                "dataset": dataset,
                "language": language_code(dataset),
                "shard": {
                    key: shard[key]
                    for key in (
                        "canonical_path",
                        "sha256",
                        "bytes",
                        "rows",
                    )
                },
                "shard_row_start": 0,
                "shard_row_stop": count,
                "dataset_row_start": dataset_row,
                "dataset_row_stop": dataset_row + count,
                "global_row_start": global_row,
                "global_row_stop": global_row + count,
            })
            remaining -= count
            dataset_row += count
            global_row += count
        if take < budget:
            missing = budget - take
            gaps.append({
                "dataset": dataset,
                "budget_rows": budget,
                "available_rows": available,
                "missing_rows": missing,
                "estimated_embed_wall_hours_at_319_rows_per_second": (
                    missing / 319.0 / 3600.0
                ),
            })
    return {
        "target_rows": TARGET_ROWS,
        "selected_rows": global_row,
        "complete": global_row == TARGET_ROWS and not gaps,
        "source_order": list(budgets),
        "budgets": budgets,
        "ranges": ranges,
        "gaps": gaps,
        "probe_exclusions": {
            "heldout_language": POLISH,
            "heldout_english": HELDOUT,
            "ood": ["dadabase-jina-v5-nano-probe"],
        },
        "row_order": (
            "dataset order above, then lexicographic shard path, then "
            "ascending row within shard"
        ),
    }


def _catalog_objects(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            yield value
        for item in value.values():
            yield from _catalog_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _catalog_objects(item)


def reconcile_catalog(
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    catalog_path: Path = CATALOG_PATH,
) -> dict[str, Any]:
    with open(catalog_path, encoding="utf-8") as handle:
        catalog = json.load(handle)
    by_path = {
        os.path.realpath(str(item["path"])): item
        for item in _catalog_objects(catalog)
    }
    rows: dict[str, Any] = {}
    for name, dataset in inventory.items():
        catalog_row = by_path.get(os.path.realpath(str(dataset["root"])))
        observed = {
            "rows": dataset["rows"],
            "dimension": dataset["dimension"],
            "dtype": "float16",
            "shards_present": len(dataset["shards"]),
        }
        expected = (
            {
                key: catalog_row.get(key)
                for key in observed
            }
            if catalog_row is not None else None
        )
        rows[name] = {
            "catalog_entry_found": catalog_row is not None,
            "catalog_generated_at": catalog.get("generated_at"),
            "observed": observed,
            "catalog": expected,
            "matches": expected == observed,
        }
    return {
        "catalog": expected_input_signature(str(catalog_path)),
        "datasets": rows,
        "all_match": all(row["matches"] for row in rows.values()),
    }


@njit(parallel=True, cache=True)
def _fingerprint_fp16(
    bits: np.ndarray,
    out_h0: np.ndarray,
    out_h1: np.ndarray,
    zero_mask: np.ndarray,
    nonfinite_mask: np.ndarray,
) -> None:
    rows, dimension = bits.shape
    for row in prange(rows):
        h0 = np.uint64(1469598103934665603)
        h1 = np.uint64(7809847782465536322)
        zero = True
        nonfinite = False
        for column in range(dimension):
            value = np.uint64(bits[row, column])
            zero = zero and (value & np.uint64(0x7FFF)) == 0
            nonfinite = nonfinite or (
                value & np.uint64(0x7C00)
            ) == np.uint64(0x7C00)
            h0 = (h0 ^ (value & np.uint64(255))) * np.uint64(
                1099511628211
            )
            h0 = (h0 ^ (value >> np.uint64(8))) * np.uint64(
                1099511628211
            )
            h1 = (
                h1 ^ (value + np.uint64(column + 1))
            ) * np.uint64(14029467366897019727)
        out_h0[row] = h0
        out_h1[row] = h1
        zero_mask[row] = zero
        nonfinite_mask[row] = nonfinite


class _SelectedRows:
    def __init__(self, ranges: list[Mapping[str, Any]]) -> None:
        self.ranges = ranges
        self.stops = [int(item["global_row_stop"]) for item in ranges]
        self.arrays: dict[str, np.ndarray] = {}

    def bytes(self, row: int) -> bytes:
        index = bisect.bisect_right(self.stops, row)
        item = self.ranges[index]
        path = str(item["shard"]["canonical_path"])
        array = self.arrays.get(path)
        if array is None:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            self.arrays[path] = array
        local = int(item["shard_row_start"]) + (
            row - int(item["global_row_start"])
        )
        return np.asarray(array[local]).tobytes(order="C")

    def lexicographic_key(self, row: int) -> tuple[str, str, int]:
        index = bisect.bisect_right(self.stops, row)
        item = self.ranges[index]
        local = int(item["shard_row_start"]) + (
            row - int(item["global_row_start"])
        )
        return (
            str(item.get("dataset", "")),
            str(item["shard"]["canonical_path"]),
            local,
        )

    def close_and_drop_cache(self) -> None:
        paths = list(self.arrays)
        self.arrays.clear()
        for path in paths:
            drop_file_cache(path)


def duplicate_census(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    ranges = list(selection["ranges"])
    row_count = int(selection["selected_rows"])
    if row_count <= 0 or row_count > np.iinfo(np.uint32).max:
        raise Round0087Error("selection row count cannot be fingerprinted")
    records = np.empty(row_count, dtype=FP_DTYPE)
    zero_mask = np.empty(row_count, dtype=np.bool_)
    nonfinite_mask = np.empty(row_count, dtype=np.bool_)
    fingerprint_started = time.monotonic()
    for item in ranges:
        start = int(item["global_row_start"])
        stop = int(item["global_row_stop"])
        array = np.load(
            str(item["shard"]["canonical_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        source = np.asarray(
            array[
                int(item["shard_row_start"]):
                int(item["shard_row_stop"])
            ]
        ).view("<u2")
        _fingerprint_fp16(
            source,
            records["h0"][start:stop],
            records["h1"][start:stop],
            zero_mask[start:stop],
            nonfinite_mask[start:stop],
        )
        records["row"][start:stop] = np.arange(
            start, stop, dtype=np.uint32
        )
        del source, array
        drop_file_cache(str(item["shard"]["canonical_path"]))
    fingerprint_seconds = time.monotonic() - fingerprint_started
    sort_started = time.monotonic()
    records.sort(order=("h0", "h1"), kind="stable")
    sort_seconds = time.monotonic() - sort_started

    accessor = _SelectedRows(ranges)
    families: list[np.ndarray] = []
    collision_splits = 0
    repeated_groups = 0
    index = 0
    while index < row_count - 1:
        stop = index + 1
        h0 = records["h0"][index]
        h1 = records["h1"][index]
        while (
            stop < row_count
            and records["h0"][stop] == h0
            and records["h1"][stop] == h1
        ):
            stop += 1
        if stop - index >= 2:
            candidates = np.sort(
                records["row"][index:stop].astype(np.int64)
            )
            candidates = candidates[
                ~zero_mask[candidates] & ~nonfinite_mask[candidates]
            ]
            if len(candidates) >= 2:
                repeated_groups += 1
                exact: dict[bytes, list[int]] = {}
                for row in candidates.tolist():
                    exact.setdefault(accessor.bytes(row), []).append(row)
                collision_splits += max(len(exact) - 1, 0)
                for rows in exact.values():
                    if len(rows) >= 2:
                        rows.sort(key=accessor.lexicographic_key)
                        families.append(np.asarray(rows, dtype=np.int64))
        index = stop
    accessor.close_and_drop_cache()
    families.sort(
        key=lambda value: accessor.lexicographic_key(int(value[0]))
    )
    counts = np.asarray([len(rows) for rows in families], dtype=np.int64)
    representatives = np.asarray(
        [rows[0] for rows in families], dtype=np.int64
    )
    offsets = np.zeros(len(families) + 1, dtype=np.int64)
    if len(counts):
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        members = np.concatenate(families)
        duplicate_rows = np.concatenate([rows[1:] for rows in families])
        duplicate_reps = np.concatenate([
            np.full(len(rows) - 1, rows[0], dtype=np.int64)
            for rows in families
        ])
        order = np.argsort(duplicate_rows, kind="stable")
        duplicate_rows = duplicate_rows[order]
        duplicate_reps = duplicate_reps[order]
    else:
        members = np.empty(0, dtype=np.int64)
        duplicate_rows = np.empty(0, dtype=np.int64)
        duplicate_reps = np.empty(0, dtype=np.int64)
    zero_rows = np.flatnonzero(zero_mask).astype(np.int64)
    nonfinite_rows = np.flatnonzero(nonfinite_mask).astype(np.int64)
    excluded = np.unique(np.concatenate([
        zero_rows, nonfinite_rows, duplicate_rows
    ])).astype(np.int64)
    retained = row_count - len(excluded)
    arrays = {
        "zero_rows": zero_rows,
        "nonfinite_rows": nonfinite_rows,
        "excluded_rows": excluded,
        "duplicate_excluded_rows": duplicate_rows,
        "duplicate_representative_rows": duplicate_reps,
        "representative_rows": representatives,
        "family_counts": counts,
        "family_offsets": offsets,
        "member_rows": members,
    }
    summary = {
        "row_count": row_count,
        "zero_row_count": len(zero_rows),
        "nonfinite_row_count": len(nonfinite_rows),
        "exact_nonzero_finite_family_count": len(families),
        "rows_in_exact_nonzero_finite_families": int(counts.sum()),
        "duplicate_copy_rows_excluded": len(duplicate_rows),
        "excluded_row_count": len(excluded),
        "retained_row_count": retained,
        "family_size_histogram": {
            str(size): count
            for size, count in sorted(Counter(counts.tolist()).items())
        },
        "repeated_fingerprint_groups": repeated_groups,
        "fingerprint_collision_splits": collision_splits,
        "fingerprint_wall_seconds": fingerprint_seconds,
        "sort_wall_seconds": sort_seconds,
    }
    if (
        len(excluded) != len(np.unique(excluded))
        or int(counts.sum()) - len(families) != len(duplicate_rows)
        or retained + len(excluded) != row_count
    ):
        raise Round0087Error("duplicate eligibility accounting does not close")
    return {"arrays": arrays, "summary": summary}
