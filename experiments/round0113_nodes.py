"""Execute the paired raw/document Jina map contrast for Round 0113."""
from __future__ import annotations

import gc
import hashlib
import math
import os
import random
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0104_training import L2NormalizedArray
from basemap.round0108_evaluation import exact_reference_copy_mask
from basemap.round0112_prompt_substrate import (
    MODEL_ROOT,
    PROMPT_PREFIX,
)
from basemap.round0113_prompt_contrast import (
    ARMS,
    ASSEMBLY_SCHEMA,
    BASELINE_EXCLUDED_ROWS,
    BASELINE_RETAINED_ROWS,
    BATCH_SIZE,
    DECISION_METRICS,
    DIMENSION,
    GRAPH_K,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NLIST,
    GRAPH_NPROBE_GRID,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_QUALITY_ROWS,
    GRAPH_QUALITY_SEED,
    GRAPH_SCHEMA,
    GRAPH_TRAIN_ROWS,
    GRAPH_TRAIN_SEED,
    PANEL_ANCHORS,
    PANEL_SEED,
    PERFORMANCE_WARMUP_UPDATES,
    PERFORMANCE_WINDOWS,
    POLISH_HISTORICAL_EMBEDDING_SHA256,
    POLISH_QUERY_ROWS,
    POLISH_SOURCE_ROWS,
    PROMPT_UNION_EXTRA_EXCLUDED_ROWS,
    PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256,
    QUERY_CANDIDATES,
    QUERY_ROWS,
    QUERY_SCHEMA,
    QUERY_SELECTION_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID,
    SEED,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
    HostFp16EndpointArray,
    PromptTrainingInput,
    Round0113Error,
    baseline_compact_mapping,
    compact_mapping,
    load_graph,
    load_substrate_manifest,
    paired_decision,
    panel_config,
    polish_query_rows,
    query_candidate_rows,
    query_source_layout,
    read_sealed,
    seal,
    synchronize_runtime_counters,
    train_config,
    verify_signature,
)

def _schema(stem: str) -> str:
    return f"round0113-{stem}-v1"


def _faiss_gpu_options(faiss: Any) -> Any:
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    return options


def _without_self(
    rows: np.ndarray, ids: np.ndarray, width: int
) -> np.ndarray:
    out = np.empty((len(rows), width), dtype=np.int64)
    for index, row in enumerate(np.asarray(rows, dtype=np.int64)):
        kept = row[row != int(ids[index])]
        if len(kept) < width or len(np.unique(kept[:width])) != width:
            raise Round0113Error("search did not return enough unique nonself rows")
        out[index] = kept[:width]
    return out


def _recall_rows(observed: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.isin(observed[index], truth[index]).sum() / truth.shape[1]
            for index in range(len(truth))
        ],
        dtype=np.float64,
    )


def _recall(high: np.ndarray, low: np.ndarray, k: int) -> float:
    return float(
        np.mean(
            [
                len(np.intersect1d(high[index, :k], low[index])) / k
                for index in range(len(high))
            ]
        )
    )


def _fetch_parquet_rows(
    path: str,
    rows: np.ndarray,
    *,
    expected_rows: int,
) -> list[str]:
    """Stream one text column and retain only requested ordered positions."""
    import pyarrow.parquet as pq

    requested = np.asarray(rows, dtype=np.int64)
    if (
        requested.ndim != 1
        or not len(requested)
        or np.any(requested[1:] <= requested[:-1])
        or requested[0] < 0
        or requested[-1] >= expected_rows
    ):
        raise Round0113Error("R0113 parquet row request is malformed")
    parquet = pq.ParquetFile(path)
    if int(parquet.metadata.num_rows) != expected_rows:
        raise Round0113Error("R0113 parquet source row count changed")
    output: list[str | None] = [None] * len(requested)
    cursor = 0
    for batch in parquet.iter_batches(batch_size=65_536, columns=["chunk_text"]):
        stop = cursor + len(batch)
        left = int(np.searchsorted(requested, cursor, side="left"))
        right = int(np.searchsorted(requested, stop, side="left"))
        if right > left:
            local = requested[left:right] - cursor
            values = batch.column(0).take(local.tolist()).to_pylist()
            for index, value in enumerate(values, start=left):
                output[index] = value
        cursor = stop
    if (
        cursor != expected_rows
        or any(not isinstance(value, str) for value in output)
    ):
        raise Round0113Error("R0113 parquet text fetch did not close")
    return [str(value) for value in output]


def _require_unique_stored_rows(values: np.ndarray, *, label: str) -> None:
    stored = np.ascontiguousarray(values.astype("<f2", copy=False))
    keys = stored.view(np.dtype((np.void, stored.shape[1] * stored.dtype.itemsize)))
    if len(np.unique(keys)) != len(stored):
        raise Round0113Error(f"R0113 {label} contains exact repeated rows")


def _text_row_hashes(texts: list[str]) -> np.ndarray:
    """Return one complete UTF-8 SHA-256 identity per ordered source text."""
    hashes = np.empty(len(texts), dtype="V32")
    for index, text in enumerate(texts):
        hashes[index] = hashlib.sha256(text.encode("utf-8")).digest()
    return hashes


def _sorted_hash_membership(
    sorted_reference: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(sorted_reference)
    values = np.asarray(queries)
    if (
        reference.ndim != 1
        or values.ndim != 1
        or reference.dtype != np.dtype("V32")
        or values.dtype != np.dtype("V32")
        or not np.array_equal(
            reference, np.sort(reference, kind="stable")
        )
        or (
            len(reference) > 1
            and np.any(reference[1:] == reference[:-1])
        )
    ):
        raise Round0113Error("R0113 source-text hash index is malformed")
    positions = np.searchsorted(reference, values)
    present = positions < len(reference)
    if np.any(present):
        present[present] = (
            reference[positions[present]] == values[present]
        )
    return present


def _exact_duplicate_audit(
    source: np.ndarray,
    *,
    mapping: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find complete-byte duplicate families via a lossless candidate index."""
    values = np.asarray(source)
    if (
        values.ndim != 2
        or not len(values)
        or values.shape[1] <= 0
        or values.dtype != np.float16
    ):
        raise Round0113Error("R0113 duplicate-audit source is malformed")
    if mapping is not None:
        global_rows = np.asarray(mapping, dtype=np.int64)
        if global_rows.shape != (len(values),):
            raise Round0113Error("R0113 duplicate-audit mapping is malformed")
    else:
        global_rows = np.arange(len(values), dtype=np.int64)
    positions = np.unique(
        np.linspace(0, values.shape[1] - 1, 32, dtype=np.int64)
    )
    projection = np.ascontiguousarray(values[:, positions]).view(np.uint16)
    key_dtype = np.dtype(
        (np.void, projection.dtype.itemsize * projection.shape[1])
    )
    keys = projection.view(key_dtype).reshape(-1)
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    equal = ordered[1:] == ordered[:-1]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(~equal).astype(np.int64) + 1,
        )
    )
    stops = np.concatenate(
        (starts[1:], np.asarray([len(values)], dtype=np.int64))
    )
    repeated = np.flatnonzero(stops - starts > 1)
    exact_families = 0
    exact_rows = 0
    maximum_family = 1
    collision_splits = 0
    examples: list[list[int]] = []
    for group in repeated.tolist():
        members = order[int(starts[group]) : int(stops[group])]
        exact: dict[bytes, list[int]] = {}
        for member in members.tolist():
            exact.setdefault(
                np.asarray(values[member]).tobytes(order="C"), []
            ).append(member)
        collision_splits += max(len(exact) - 1, 0)
        for family in exact.values():
            if len(family) < 2:
                continue
            exact_families += 1
            exact_rows += len(family)
            maximum_family = max(maximum_family, len(family))
            if len(examples) < 16:
                examples.append(global_rows[family].astype(int).tolist())
    return {
        "identity": "complete stored fp16 row bytes",
        "candidate_projection_positions": positions.tolist(),
        "candidate_repeated_groups": int(len(repeated)),
        "candidate_collision_splits": int(collision_splits),
        "exact_nontrivial_family_count": int(exact_families),
        "rows_in_exact_nontrivial_families": int(exact_rows),
        "maximum_exact_family_size": int(maximum_family),
        "example_global_families": examples,
        "passed_no_retained_exact_duplicates": exact_families == 0,
    }


def _exact_families_from_chunks(
    chunks: list[dict[str, Any]],
    mapping: np.ndarray,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Enumerate complete-byte families on a chunked fp16 population."""
    compact = np.asarray(mapping, dtype=np.int64)
    if (
        len(chunks) != 80
        or compact.shape != (BASELINE_RETAINED_ROWS,)
        or np.any(compact[1:] <= compact[:-1])
    ):
        raise Round0113Error("R0113 source-family census input is malformed")
    positions = np.unique(
        np.linspace(0, DIMENSION - 1, 32, dtype=np.int64)
    )
    projection = np.empty(
        (len(compact), len(positions)), dtype=np.uint16
    )
    arrays: list[np.ndarray] = []
    for chunk_index, signature in enumerate(chunks):
        values = np.load(
            str(signature["canonical_path"]), mmap_mode="r", allow_pickle=False
        )
        if values.shape != (25_000, DIMENSION) or values.dtype != np.float16:
            raise Round0113Error("R0113 source-family chunk geometry changed")
        start = chunk_index * 25_000
        stop = start + 25_000
        left = int(np.searchsorted(compact, start, side="left"))
        right = int(np.searchsorted(compact, stop, side="left"))
        local = compact[left:right] - start
        selected = np.ascontiguousarray(
            np.take(values, positions, axis=1)[local]
        ).view(np.uint16)
        projection[left:right] = selected
        arrays.append(values)
    key_dtype = np.dtype(
        (np.void, projection.dtype.itemsize * projection.shape[1])
    )
    keys = projection.view(key_dtype).reshape(-1)
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    equal = ordered[1:] == ordered[:-1]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(~equal).astype(np.int64) + 1,
        )
    )
    stops = np.concatenate(
        (starts[1:], np.asarray([len(compact)], dtype=np.int64))
    )
    repeated = np.flatnonzero(stops - starts > 1)
    families: list[list[int]] = []
    collision_splits = 0
    for group in repeated.tolist():
        members = order[int(starts[group]) : int(stops[group])]
        exact: dict[bytes, list[int]] = {}
        for member in members.tolist():
            global_row = int(compact[member])
            chunk_index, local_row = divmod(global_row, 25_000)
            exact.setdefault(
                np.asarray(arrays[chunk_index][local_row]).tobytes(order="C"),
                [],
            ).append(global_row)
        collision_splits += max(len(exact) - 1, 0)
        for family in exact.values():
            if len(family) >= 2:
                families.append(sorted(family))
    families.sort(key=lambda family: (family[0], len(family), family))
    return families, {
        "identity": "complete stored fp16 row bytes",
        "candidate_projection_positions": positions.tolist(),
        "candidate_repeated_groups": int(len(repeated)),
        "candidate_collision_splits": int(collision_splits),
        "exact_nontrivial_family_count": int(len(families)),
        "rows_in_exact_nontrivial_families": int(
            sum(len(family) for family in families)
        ),
        "maximum_exact_family_size": int(
            max((len(family) for family in families), default=1)
        ),
        "families_global_rows": families,
    }


def _iter_selected_texts(
    layout: list[dict[str, Any]],
    selected_global_rows: np.ndarray,
):
    """Yield exact source texts for sorted selected rows without full materialization."""
    import pyarrow.parquet as pq

    selected = np.asarray(selected_global_rows, dtype=np.int64)
    if (
        selected.ndim != 1
        or not len(selected)
        or np.any(selected[1:] <= selected[:-1])
    ):
        raise Round0113Error("R0113 selected text rows are malformed")
    yielded = 0
    for item in layout:
        global_start = int(item["global_row_start"])
        global_stop = int(item["global_row_stop"])
        shard_start = int(item["shard_row_start"])
        shard_stop = int(item["shard_row_stop"])
        left = int(np.searchsorted(selected, global_start, side="left"))
        right = int(np.searchsorted(selected, global_stop, side="left"))
        if right <= left:
            continue
        targets = selected[left:right] - global_start + shard_start
        parquet = pq.ParquetFile(str(item["text"]["canonical_path"]))
        if int(parquet.metadata.num_rows) != int(item["shard_rows"]):
            raise Round0113Error("R0113 training text shard rows changed")
        cursor = 0
        target_cursor = 0
        for batch in parquet.iter_batches(
            batch_size=65_536, columns=["chunk_text"]
        ):
            stop = cursor + len(batch)
            if stop <= shard_start:
                cursor = stop
                continue
            if cursor >= shard_stop:
                break
            batch_left = int(
                np.searchsorted(targets, cursor, side="left")
            )
            batch_right = int(
                np.searchsorted(targets, stop, side="left")
            )
            if batch_right > batch_left:
                local = targets[batch_left:batch_right] - cursor
                texts = batch.column(0).take(local.tolist()).to_pylist()
                for offset, text in enumerate(texts, start=batch_left):
                    if not isinstance(text, str):
                        raise Round0113Error(
                            "R0113 training source text is not a string"
                        )
                    yield int(selected[left + offset]), text
                    yielded += 1
                target_cursor = batch_right
            cursor = stop
        if target_cursor != len(targets):
            raise Round0113Error("R0113 training text shard fetch is incomplete")
    if yielded != len(selected):
        raise Round0113Error("R0113 selected training texts did not close")


def _exact_text_families(
    layout: list[dict[str, Any]],
    mapping: np.ndarray,
) -> tuple[list[list[int]], dict[str, Any], np.ndarray]:
    """Enumerate complete UTF-8 source-text families over retained rows."""
    compact = np.asarray(mapping, dtype=np.int64)
    if compact.shape != (BASELINE_RETAINED_ROWS,):
        raise Round0113Error("R0113 text-family mapping is malformed")
    hashes = np.empty(len(compact), dtype="V32")
    for index, (global_row, text) in enumerate(
        _iter_selected_texts(layout, compact)
    ):
        if global_row != int(compact[index]):
            raise Round0113Error("R0113 text-family row order changed")
        hashes[index] = hashlib.sha256(text.encode("utf-8")).digest()
    order = np.argsort(hashes, kind="stable")
    ordered = hashes[order]
    equal = ordered[1:] == ordered[:-1]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(~equal).astype(np.int64) + 1,
        )
    )
    stops = np.concatenate(
        (starts[1:], np.asarray([len(compact)], dtype=np.int64))
    )
    repeated = np.flatnonzero(stops - starts > 1)
    candidate_compact = np.sort(
        np.concatenate(
            [
                order[int(starts[group]) : int(stops[group])]
                for group in repeated.tolist()
            ]
        )
        if len(repeated)
        else np.empty(0, dtype=np.int64)
    )
    candidate_globals = compact[candidate_compact]
    texts_by_row = {
        global_row: text.encode("utf-8")
        for global_row, text in _iter_selected_texts(
            layout, candidate_globals
        )
    } if len(candidate_globals) else {}
    families: list[list[int]] = []
    collision_splits = 0
    for group in repeated.tolist():
        members = compact[
            order[int(starts[group]) : int(stops[group])]
        ]
        exact: dict[bytes, list[int]] = {}
        for global_row in members.tolist():
            exact.setdefault(texts_by_row[int(global_row)], []).append(
                int(global_row)
            )
        collision_splits += max(len(exact) - 1, 0)
        for family in exact.values():
            if len(family) >= 2:
                families.append(sorted(family))
    families.sort(key=lambda family: (family[0], len(family), family))
    return families, {
        "identity": "complete source-text UTF-8 bytes",
        "candidate_hash": "SHA-256 over each complete UTF-8 text",
        "candidate_repeated_groups": int(len(repeated)),
        "candidate_collision_splits": int(collision_splits),
        "exact_nontrivial_family_count": int(len(families)),
        "rows_in_exact_nontrivial_families": int(
            sum(len(family) for family in families)
        ),
        "maximum_exact_family_size": int(
            max((len(family) for family in families), default=1)
        ),
        "families_global_rows": families,
    }, hashes


def _union_prompt_exclusions(
    families_by_arm: Mapping[str, list[list[int]]],
    baseline_mapping: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose one shared lowest-global-row representative per union family."""
    family_sources = {"text", *ARMS}
    if set(families_by_arm) != family_sources:
        raise Round0113Error("R0113 prompt-family sources are incomplete")
    retained = np.asarray(baseline_mapping, dtype=np.int64)
    parent: dict[int, int] = {}

    def find(row: int) -> int:
        root = parent.setdefault(row, row)
        while parent[root] != root:
            root = parent[root]
        while parent[row] != row:
            next_row = parent[row]
            parent[row] = root
            row = next_row
        return root

    def union(left: int, right: int) -> None:
        a = find(left)
        b = find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for families in families_by_arm.values():
        for family in families:
            members = sorted(set(int(row) for row in family))
            positions = np.searchsorted(retained, members)
            if (
                len(members) < 2
                or np.any(positions >= len(retained))
                or not np.array_equal(retained[positions], members)
            ):
                raise Round0113Error(
                    "R0113 prompt family is outside baseline population"
                )
            for row in members[1:]:
                union(members[0], row)
    components: dict[int, list[int]] = {}
    for row in sorted(parent):
        components.setdefault(find(row), []).append(row)
    union_families = [
        sorted(family)
        for family in components.values()
        if len(family) >= 2
    ]
    union_families.sort(key=lambda family: (family[0], len(family), family))
    extra = np.asarray(
        sorted(row for family in union_families for row in family[1:]),
        dtype=np.int64,
    )
    return extra, {
        "selection_rule": (
            "union exact source-text, raw-fp16, and document-fp16 family "
            "relations over R0112 baseline representatives; keep the lowest "
            "global row per transitive component in both arms"
        ),
        "source_family_counts": {
            source: len(families_by_arm[source])
            for source in sorted(family_sources)
        },
        "union_family_count": len(union_families),
        "union_families_global_rows": union_families,
        "extra_excluded_global_rows": extra.tolist(),
        "extra_excluded_rows": int(len(extra)),
    }


def _data_identity(
    assembly: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    return {
        "kind": "round0113-compact-prompt-array",
        "shape": [RETAINED_ROWS, DIMENSION],
        "dtype": np.dtype("<f2").str,
        "arm": arm,
        "source": assembly["outputs"][arm],
        "mapping": assembly["mapping"],
        "substrate": assembly["substrate"],
    }


def _materialize_normalized(source: np.ndarray) -> np.ndarray:
    values = np.empty((len(source), DIMENSION), dtype=np.float32)
    for start in range(0, len(source), 65_536):
        stop = min(start + 65_536, len(source))
        block = np.asarray(source[start:stop], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if (
            not np.isfinite(block).all()
            or not np.isfinite(norms).all()
            or np.any(norms <= 0)
        ):
            raise Round0113Error("R0113 source has zero/nonfinite rows")
        values[start:stop] = block / norms
    return values


def _load_assembly(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = os.path.join(str(job["assembly_output"]), "assembly-manifest.json")
    manifest = read_sealed(path, label="R0113 compact assembly")
    if (
        manifest.get("schema") != ASSEMBLY_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or int(manifest.get("retained_rows", -1)) != RETAINED_ROWS
        or set(manifest.get("outputs") or {}) != set(ARMS)
    ):
        raise Round0113Error("R0113 compact assembly contract changed")
    verify_signature(manifest["mapping"], label="R0113 compact mapping")
    text_hash_index_path = verify_signature(
        manifest["source_text_hash_index"],
        label="R0113 compact source-text hash index",
    )
    text_hash_index = np.load(
        text_hash_index_path, mmap_mode="r", allow_pickle=False
    )
    if (
        text_hash_index.shape != (RETAINED_ROWS,)
        or text_hash_index.dtype != np.dtype("V32")
        or not np.array_equal(
            text_hash_index, np.sort(text_hash_index, kind="stable")
        )
        or (
            len(text_hash_index) > 1
            and np.any(text_hash_index[1:] == text_hash_index[:-1])
        )
    ):
        raise Round0113Error("R0113 compact source-text hash index changed")
    discovery_path = verify_signature(
        manifest["source_prompt_family_discovery"],
        label="R0113 source prompt-family discovery",
    )
    discovery = read_sealed(
        discovery_path, label="R0113 source prompt-family discovery"
    )
    if discovery.get("matched_preregistered_union") is not True:
        raise Round0113Error("R0113 prompt-family discovery changed")
    audit_path = verify_signature(
        manifest["retained_duplicate_audit"],
        label="R0113 retained duplicate audit",
    )
    audit = read_sealed(audit_path, label="R0113 retained duplicate audit")
    if (
        audit.get("passed") is not True
        or set(audit.get("arms") or {}) != set(ARMS)
        or any(
            report.get("passed_no_retained_exact_duplicates") is not True
            for report in audit["arms"].values()
        )
    ):
        raise Round0113Error("R0113 retained duplicate audit changed")
    for arm in ARMS:
        verify_signature(manifest["outputs"][arm], label=f"R0113 {arm} compact input")
    return manifest, expected_input_signature(path)


def _open_compact(
    assembly: Mapping[str, Any], arm: str
) -> np.memmap:
    if arm not in ARMS:
        raise Round0113Error(f"unknown R0113 arm {arm!r}")
    path = verify_signature(
        assembly["outputs"][arm], label=f"R0113 {arm} compact array"
    )
    source = np.memmap(
        path,
        mode="r",
        dtype="<f2",
        shape=(RETAINED_ROWS, DIMENSION),
    )
    return source


def _write_compact(
    path: str,
    *,
    chunks: list[dict[str, Any]],
    mapping: np.ndarray,
) -> None:
    output = np.memmap(
        path,
        mode="w+",
        dtype="<f2",
        shape=(RETAINED_ROWS, DIMENSION),
    )
    written = 0
    for chunk_index, signature in enumerate(chunks):
        start = chunk_index * 25_000
        stop = start + 25_000
        left = int(np.searchsorted(mapping, start, side="left"))
        right = int(np.searchsorted(mapping, stop, side="left"))
        rows = mapping[left:right] - start
        values = np.load(
            str(signature["canonical_path"]), mmap_mode="r", allow_pickle=False
        )
        if values.shape != (25_000, DIMENSION) or values.dtype != np.float16:
            raise Round0113Error("R0112 embedding chunk geometry changed")
        output[written : written + len(rows)] = values[rows]
        written += len(rows)
    if written != RETAINED_ROWS:
        raise Round0113Error("R0113 compact assembly row count did not close")
    output.flush()
    del output


def run_assemble(active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0], label="R0113 compact prompt arrays"
    )
    substrate = load_substrate_manifest(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
        verify_chunks=False,
    )
    baseline_mapping = baseline_compact_mapping(substrate["excluded"])
    discovery_started = time.monotonic()
    text_layout = list(job.get("source_text_layout") or [])
    text_families, text_report, baseline_text_hashes = _exact_text_families(
        text_layout, baseline_mapping
    )
    families_by_arm: dict[str, list[list[int]]] = {
        "text": text_families
    }
    family_reports: dict[str, Any] = {"text": text_report}
    for arm in ARMS:
        families, report = _exact_families_from_chunks(
            substrate["chunks"][arm], baseline_mapping
        )
        families_by_arm[arm] = families
        family_reports[arm] = report
    derived_extra, union_report = _union_prompt_exclusions(
        families_by_arm, baseline_mapping
    )
    derived_extra_sha256 = ordered_array_sha256(derived_extra)
    matched_preregistered_union = bool(
        len(derived_extra) == PROMPT_UNION_EXTRA_EXCLUDED_ROWS
        and derived_extra_sha256 == PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256
    )
    discovery = seal(
        {
            "schema": _schema("source-prompt-family-discovery"),
            "round_id": ROUND_ID,
            "release_sha": active["manifest"]["release_sha"],
            "substrate": substrate["signature"],
            "baseline_selector": substrate["selector"],
            "baseline_retained_rows": BASELINE_RETAINED_ROWS,
            "source_text_layout": text_layout,
            "arms": family_reports,
            "union": union_report,
            "derived_extra_exclusions_sha256": derived_extra_sha256,
            "expected_extra_excluded_rows": PROMPT_UNION_EXTRA_EXCLUDED_ROWS,
            "expected_extra_exclusions_sha256": (
                PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256
            ),
            "matched_preregistered_union": matched_preregistered_union,
            "wall_s": time.monotonic() - discovery_started,
        }
    )
    discovery_path = os.path.join(output, "source-prompt-family-discovery.json")
    atomic_write_new_json(discovery_path, discovery, immutable=True)
    if discovery["matched_preregistered_union"] is not True:
        raise Round0113Error(
            "R0113 complete prompt-family union differs from preregistration"
        )
    mapping = compact_mapping(substrate["excluded"], derived_extra)
    mapping_path = os.path.join(output, "compact-to-global.i64.npy")
    atomic_save_new_npy(mapping_path, mapping, immutable=True)
    mapping_positions = np.searchsorted(baseline_mapping, mapping)
    if (
        np.any(mapping_positions >= len(baseline_mapping))
        or not np.array_equal(baseline_mapping[mapping_positions], mapping)
    ):
        raise Round0113Error("R0113 final source-text mapping changed")
    retained_text_hashes = np.sort(
        baseline_text_hashes[mapping_positions], kind="stable"
    )
    if (
        retained_text_hashes.shape != (RETAINED_ROWS,)
        or (
            len(retained_text_hashes) > 1
            and np.any(retained_text_hashes[1:] == retained_text_hashes[:-1])
        )
    ):
        raise Round0113Error(
            "R0113 shared selector leaves exact source-text duplicates"
        )
    text_hash_index_path = os.path.join(
        output, "source-text-sha256-sorted.v32.npy"
    )
    atomic_save_new_npy(
        text_hash_index_path, retained_text_hashes, immutable=True
    )
    outputs: dict[str, Any] = {}
    started = time.monotonic()
    arm_wall: dict[str, float] = {}
    for arm in ARMS:
        arm_started = time.monotonic()
        path = os.path.join(output, f"{arm}-compact.f16")
        atomic_build_new_file(
            path,
            lambda temporary, arm=arm: _write_compact(
                temporary,
                chunks=substrate["chunks"][arm],
                mapping=mapping,
            ),
            immutable=True,
        )
        signature = expected_input_signature(path)
        if signature["bytes"] != RETAINED_ROWS * DIMENSION * 2:
            raise Round0113Error("R0113 compact fp16 byte count changed")
        outputs[arm] = signature
        arm_wall[arm] = time.monotonic() - arm_started
    audit_started = time.monotonic()
    duplicate_audits: dict[str, Any] = {}
    for arm in ARMS:
        source = np.memmap(
            str(outputs[arm]["canonical_path"]),
            mode="r",
            dtype="<f2",
            shape=(RETAINED_ROWS, DIMENSION),
        )
        duplicate_audits[arm] = _exact_duplicate_audit(
            source, mapping=mapping
        )
        del source
        gc.collect()
    audit = seal(
        {
            "schema": _schema("retained-duplicate-audit"),
            "round_id": ROUND_ID,
            "release_sha": active["manifest"]["release_sha"],
            "selector": substrate["selector"],
            "mapping": expected_input_signature(mapping_path),
            "arms": duplicate_audits,
            "shared_population_required": True,
            "policy": (
                "the preregistered shared source/raw/document union selector "
                "must leave zero complete-fp16-row families in either fresh "
                "convention"
            ),
            "passed": all(
                report["passed_no_retained_exact_duplicates"]
                for report in duplicate_audits.values()
            ),
            "wall_s": time.monotonic() - audit_started,
        }
    )
    audit_path = os.path.join(output, "retained-duplicate-audit.json")
    atomic_write_new_json(audit_path, audit, immutable=True)
    if audit["passed"] is not True:
        raise Round0113Error(
            "R0113 shared selector leaves retained exact duplicates"
        )
    body = {
        "schema": ASSEMBLY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "selector": substrate["selector"],
        "source_prompt_family_discovery": expected_input_signature(
            discovery_path
        ),
        "source_rows": 2_000_000,
        "baseline_excluded_rows": BASELINE_EXCLUDED_ROWS,
        "prompt_union_extra_excluded_rows": PROMPT_UNION_EXTRA_EXCLUDED_ROWS,
        "prompt_union_extra_exclusions_sha256": (
            PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256
        ),
        "excluded_rows": (
            BASELINE_EXCLUDED_ROWS + PROMPT_UNION_EXTRA_EXCLUDED_ROWS
        ),
        "retained_rows": RETAINED_ROWS,
        "dimension": DIMENSION,
        "dtype": np.dtype("<f2").str,
        "mapping": expected_input_signature(mapping_path),
        "source_text_hash_index": expected_input_signature(
            text_hash_index_path
        ),
        "source_text_hash_identity": (
            "sorted unique SHA-256 over each retained complete source-text "
            "UTF-8 byte string"
        ),
        "outputs": outputs,
        "retained_duplicate_audit": expected_input_signature(audit_path),
        "paired_row_population_identical": True,
        "training_performed": False,
        "performance": {
            "arm_wall_s": arm_wall,
            "total_wall_s": time.monotonic() - started,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
            ),
        },
    }
    manifest = seal(body)
    path = os.path.join(output, "assembly-manifest.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return {**manifest, "receipt": expected_input_signature(path)}


def run_embed_queries(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    from experiments.round0112_nodes import (
        _cosine_rows,
        _encode,
        _load_model,
        _normalized_guard,
        _prompt_equivalence,
    )
    from basemap.round0112_prompt_substrate import ordered_text_sha256

    output = create_fresh_directory(
        job["outputs"][0], label="R0113 dual-prompt heldout query reserve"
    )
    rows, selector = query_candidate_rows()
    layout = query_source_layout(rows)
    if layout != list(job.get("authenticated_query_layout") or []):
        raise Round0113Error(
            "R0113 authenticated query source layout changed"
        )
    texts: list[str | None] = [None] * QUERY_CANDIDATES
    for item in layout:
        positions = np.flatnonzero(
            (rows >= int(item["global_row_start"]))
            & (rows < int(item["global_row_stop"]))
        )
        local_rows = (
            rows[positions]
            - int(item["global_row_start"])
            + int(item["shard_row_start"])
        )
        values = _fetch_parquet_rows(
            str(item["text"]["canonical_path"]),
            local_rows,
            expected_rows=int(item["shard_rows"]),
        )
        for position, value in zip(positions.tolist(), values, strict=True):
            texts[position] = value
    if (
        any(not isinstance(text, str) for text in texts)
        or len(texts) != QUERY_CANDIDATES
    ):
        raise Round0113Error("R0113 query text fetch is incomplete")
    exact_texts = [str(text) for text in texts]
    polish_source = dict(job.get("polish_source") or {})
    if set(polish_source) != {"historical_embedding", "manifest", "text"}:
        raise Round0113Error("R0113 Polish source binding is incomplete")
    historical_signature = dict(polish_source["historical_embedding"])
    historical_path = str(historical_signature.get("canonical_path") or "")
    if (
        historical_signature.get("sha256")
        != POLISH_HISTORICAL_EMBEDDING_SHA256
        or not os.path.isfile(historical_path)
        or os.path.getsize(historical_path)
        != int(historical_signature.get("bytes", -1))
    ):
        raise Round0113Error("R0113 Polish historical source changed")
    historical = np.load(historical_path, mmap_mode="r", allow_pickle=False)
    if (
        historical.shape != (POLISH_SOURCE_ROWS, DIMENSION)
        or historical.dtype != np.float16
    ):
        raise Round0113Error("R0113 Polish historical geometry changed")
    polish_rows = polish_query_rows()
    polish_texts = _fetch_parquet_rows(
        str(polish_source["text"]["canonical_path"]),
        polish_rows,
        expected_rows=POLISH_SOURCE_ROWS,
    )
    model, runtime, members = _load_model()
    equivalence = _prompt_equivalence(model, exact_texts)
    started = time.monotonic()
    raw, raw_telemetry = _encode(model, exact_texts)
    document, document_telemetry = _encode(
        model, [PROMPT_PREFIX + text for text in exact_texts]
    )
    polish_raw, polish_raw_telemetry = _encode(model, polish_texts)
    polish_document, polish_document_telemetry = _encode(
        model, [PROMPT_PREFIX + text for text in polish_texts]
    )
    _normalized_guard(raw, label="R0113 raw queries")
    _normalized_guard(document, label="R0113 document queries")
    _normalized_guard(polish_raw, label="R0113 raw Polish queries")
    _normalized_guard(polish_document, label="R0113 document Polish queries")
    polish_historical_cosines = _cosine_rows(
        polish_raw,
        np.asarray(historical[polish_rows], dtype=np.float32),
    )
    if (
        float(np.mean(polish_historical_cosines)) < 0.98
        or float(np.min(polish_historical_cosines)) < 0.95
    ):
        raise Round0113Error(
            "R0113 Polish fresh-raw row alignment guard failed"
        )
    _require_unique_stored_rows(polish_raw, label="raw Polish query panel")
    _require_unique_stored_rows(
        polish_document, label="document Polish query panel"
    )
    fineweb_text_hashes = _text_row_hashes(exact_texts)
    polish_text_hashes = _text_row_hashes(polish_texts)
    if len(np.unique(polish_text_hashes)) != POLISH_QUERY_ROWS:
        raise Round0113Error(
            "R0113 Polish diagnostic contains exact repeated source text"
        )
    outputs: dict[str, Any] = {}
    for arm, values in (("raw", raw), ("document", document)):
        path = os.path.join(output, f"{arm}-query-reserve.f16.npy")
        atomic_save_new_npy(path, values.astype("<f2"), immutable=True)
        outputs[arm] = expected_input_signature(path)
    polish_outputs: dict[str, Any] = {}
    for arm, values in (
        ("raw", polish_raw),
        ("document", polish_document),
    ):
        path = os.path.join(output, f"{arm}-polish-queries.f16.npy")
        atomic_save_new_npy(path, values.astype("<f2"), immutable=True)
        polish_outputs[arm] = expected_input_signature(path)
    rows_path = os.path.join(output, "query-global-rows.i64.npy")
    atomic_save_new_npy(rows_path, rows, immutable=True)
    polish_rows_path = os.path.join(output, "polish-query-rows.i64.npy")
    atomic_save_new_npy(polish_rows_path, polish_rows, immutable=True)
    fineweb_text_hash_path = os.path.join(
        output, "query-source-text-sha256.v32.npy"
    )
    polish_text_hash_path = os.path.join(
        output, "polish-source-text-sha256.v32.npy"
    )
    atomic_save_new_npy(
        fineweb_text_hash_path, fineweb_text_hashes, immutable=True
    )
    atomic_save_new_npy(
        polish_text_hash_path, polish_text_hashes, immutable=True
    )
    body = {
        "schema": QUERY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "query_rows": expected_input_signature(rows_path),
        "ordered_query_rows_sha256": ordered_array_sha256(rows),
        "selector": selector,
        "source_layout": layout,
        "source_text_ordered_sha256": ordered_text_sha256(exact_texts),
        "source_text_row_hashes": expected_input_signature(
            fineweb_text_hash_path
        ),
        "source_text_row_hash_identity": (
            "SHA-256 over each complete source-text UTF-8 byte string"
        ),
        "document_text_ordered_sha256": ordered_text_sha256(
            [PROMPT_PREFIX + text for text in exact_texts]
        ),
        "model_root": MODEL_ROOT,
        "model_runtime": runtime,
        "model_members": members,
        "prompt_equivalence": equivalence,
        "outputs": outputs,
        "ood": {
            "pol_Latn": {
                "role": "diagnostic-only",
                "historical_source": historical_signature,
                "source_manifest": polish_source["manifest"],
                "source_text": polish_source["text"],
                "query_rows": expected_input_signature(polish_rows_path),
                "ordered_query_rows_sha256": ordered_array_sha256(
                    polish_rows
                ),
                "source_text_ordered_sha256": ordered_text_sha256(
                    polish_texts
                ),
                "source_text_row_hashes": expected_input_signature(
                    polish_text_hash_path
                ),
                "document_text_ordered_sha256": ordered_text_sha256(
                    [PROMPT_PREFIX + text for text in polish_texts]
                ),
                "outputs": polish_outputs,
                "raw_embedding": polish_raw_telemetry,
                "document_embedding": polish_document_telemetry,
                "historical_raw_alignment": {
                    "mean_cosine": float(np.mean(polish_historical_cosines)),
                    "minimum_cosine": float(np.min(polish_historical_cosines)),
                    "mean_floor": 0.98,
                    "minimum_floor": 0.95,
                    "passed": True,
                },
                "complete_stored_rows_unique_in_each_arm": True,
            }
        },
        "output_dtype": np.dtype("<f2").str,
        "dimension": DIMENSION,
        "training_performed": False,
        "performance": {
            "encode_wall_s": time.monotonic() - started,
            "raw": raw_telemetry,
            "document": document_telemetry,
            "polish_raw": polish_raw_telemetry,
            "polish_document": polish_document_telemetry,
        },
    }
    receipt = seal(body)
    path = os.path.join(output, "query-reserve-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _load_query_reserve(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = os.path.join(
        str(job["query_output"]), "query-reserve-receipt.json"
    )
    receipt = read_sealed(path, label="R0113 query reserve")
    polish = (receipt.get("ood") or {}).get("pol_Latn") or {}
    if (
        receipt.get("schema") != QUERY_SCHEMA
        or int(receipt.get("dimension", -1)) != DIMENSION
        or set(receipt.get("outputs") or {}) != set(ARMS)
        or polish.get("role") != "diagnostic-only"
        or set(polish.get("outputs") or {}) != set(ARMS)
    ):
        raise Round0113Error("R0113 query reserve contract changed")
    verify_signature(receipt["query_rows"], label="R0113 query global rows")
    fineweb_text_hashes = np.load(
        verify_signature(
            receipt["source_text_row_hashes"],
            label="R0113 FineWeb query source-text hashes",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    polish_text_hashes = np.load(
        verify_signature(
            polish["source_text_row_hashes"],
            label="R0113 Polish query source-text hashes",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        fineweb_text_hashes.shape != (QUERY_CANDIDATES,)
        or fineweb_text_hashes.dtype != np.dtype("V32")
        or polish_text_hashes.shape != (POLISH_QUERY_ROWS,)
        or polish_text_hashes.dtype != np.dtype("V32")
        or len(np.unique(polish_text_hashes)) != POLISH_QUERY_ROWS
    ):
        raise Round0113Error("R0113 query source-text hash identity changed")
    for arm in ARMS:
        verify_signature(receipt["outputs"][arm], label=f"R0113 {arm} queries")
        verify_signature(
            polish["outputs"][arm], label=f"R0113 {arm} Polish queries"
        )
    polish_rows = np.load(
        verify_signature(polish["query_rows"], label="R0113 Polish query rows"),
        allow_pickle=False,
    )
    if not np.array_equal(polish_rows, polish_query_rows()):
        raise Round0113Error("R0113 Polish query rows changed")
    return receipt, expected_input_signature(path)


def run_build_graph(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    import faiss
    import umap.umap_ as umap_api
    from basemap.panel_v2 import (
        build_hiD_reference,
        sample_anchors,
        save_hiD_reference,
    )

    arm = str(job["arm"])
    if arm not in ARMS:
        raise Round0113Error("R0113 graph arm changed")
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0113 {arm} graph/reference"
    )
    assembly, assembly_signature = _load_assembly(job)
    query, query_signature = _load_query_reserve(job)
    source = _open_compact(assembly, arm)
    started = time.monotonic()
    X = _materialize_normalized(source)
    materialize_seconds = time.monotonic() - started

    train_rows = np.sort(
        np.random.RandomState(GRAPH_TRAIN_SEED)
        .choice(RETAINED_ROWS, GRAPH_TRAIN_ROWS, replace=False)
        .astype(np.int64)
    )
    quantizer = faiss.IndexFlatIP(DIMENSION)
    cpu_index = faiss.IndexIVFFlat(
        quantizer, DIMENSION, GRAPH_NLIST, faiss.METRIC_INNER_PRODUCT
    )
    cpu_index.cp.seed = GRAPH_TRAIN_SEED
    cpu_index.cp.niter = 25
    cpu_index.cp.spherical = True
    gpu_resource = faiss.StandardGpuResources()
    gpu_resource.setTempMemory(1 << 30)
    index = faiss.index_cpu_to_gpu(
        gpu_resource, 0, cpu_index, _faiss_gpu_options(faiss)
    )
    train_started = time.monotonic()
    index.train(np.ascontiguousarray(X[train_rows]))
    train_seconds = time.monotonic() - train_started
    add_started = time.monotonic()
    for start in range(0, RETAINED_ROWS, 100_000):
        index.add(
            np.ascontiguousarray(X[start : min(start + 100_000, RETAINED_ROWS)])
        )
    add_seconds = time.monotonic() - add_started
    if int(index.ntotal) != RETAINED_ROWS:
        raise Round0113Error("R0113 IVF row count changed")

    quality_ids = np.sort(
        np.random.RandomState(GRAPH_QUALITY_SEED)
        .choice(RETAINED_ROWS, GRAPH_QUALITY_ROWS, replace=False)
        .astype(np.int64)
    )
    exact = faiss.index_cpu_to_gpu(
        gpu_resource,
        0,
        faiss.IndexFlatIP(DIMENSION),
        _faiss_gpu_options(faiss),
    )
    for start in range(0, RETAINED_ROWS, 100_000):
        exact.add(
            np.ascontiguousarray(X[start : min(start + 100_000, RETAINED_ROWS)])
        )
    _truth_dist, truth_raw = exact.search(
        np.ascontiguousarray(X[quality_ids]), GRAPH_K
    )
    truth = _without_self(truth_raw, quality_ids, GRAPH_K - 1)
    cells: dict[str, Any] = {}
    selected_nprobe: int | None = None
    selected_observed: np.ndarray | None = None
    for nprobe in GRAPH_NPROBE_GRID:
        index.nprobe = nprobe
        cell_started = time.monotonic()
        _dist, raw = index.search(
            np.ascontiguousarray(X[quality_ids]), GRAPH_K
        )
        cell_wall = time.monotonic() - cell_started
        observed = _without_self(raw, quality_ids, GRAPH_K - 1)
        recalls = _recall_rows(observed, truth)
        passed = bool(
            recalls.mean() >= GRAPH_MEAN_RECALL_FLOOR
            and np.percentile(recalls, 10) >= GRAPH_P10_RECALL_FLOOR
        )
        cells[str(nprobe)] = {
            "mean_recall_at_49": float(recalls.mean()),
            "p10_recall_at_49": float(np.percentile(recalls, 10)),
            "wall_s": cell_wall,
            "queries_per_s": GRAPH_QUALITY_ROWS / cell_wall,
            "passed": passed,
        }
        if passed and selected_nprobe is None:
            selected_nprobe = nprobe
            selected_observed = observed.copy()
    del exact
    if selected_nprobe is None or selected_observed is None:
        raise Round0113Error(f"R0113 {arm} graph search did not qualify")

    index.nprobe = selected_nprobe
    neighbors = np.empty((RETAINED_ROWS, GRAPH_K), dtype=np.int32)
    distances = np.empty((RETAINED_ROWS, GRAPH_K), dtype=np.float32)
    search_started = time.monotonic()
    for start in range(0, RETAINED_ROWS, 16_384):
        stop = min(start + 16_384, RETAINED_ROWS)
        sims, ids = index.search(np.ascontiguousarray(X[start:stop]), GRAPH_K)
        if np.any(ids < 0) or np.any(ids >= RETAINED_ROWS):
            raise Round0113Error("R0113 full graph search returned invalid IDs")
        neighbors[start:stop] = ids.astype(np.int32, copy=False)
        distances[start:stop] = np.maximum(0.0, 1.0 - sims).astype(
            np.float32, copy=False
        )
    search_seconds = time.monotonic() - search_started

    fuzzy_started = time.monotonic()
    graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
        X,
        n_neighbors=GRAPH_K,
        random_state=np.random.RandomState(SEED),
        metric="cosine",
        knn_indices=neighbors,
        knn_dists=distances,
    )
    coo = graph.tocoo()
    sources = np.asarray(coo.row, dtype=np.int32)
    targets = np.asarray(coo.col, dtype=np.int32)
    weights = np.asarray(coo.data, dtype=np.float32)
    fuzzy_seconds = time.monotonic() - fuzzy_started
    if (
        len(sources) <= RETAINED_ROWS * (GRAPH_K - 1)
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or np.any(weights > 1)
    ):
        raise Round0113Error("R0113 fuzzy graph arrays are invalid")
    graph_path = os.path.join(output, "edges-k50-fuzzy.npz")
    atomic_save_new_npz(
        graph_path,
        immutable=True,
        compressed=False,
        sources=sources,
        targets=targets,
        weights=weights,
        n_nodes=np.asarray(RETAINED_ROWS, dtype=np.int64),
        k=np.asarray(GRAPH_K, dtype=np.int64),
    )
    graph_signature = expected_input_signature(graph_path)

    topology_path = os.path.join(output, "topology-probe.npz")
    atomic_save_new_npz(
        topology_path,
        immutable=True,
        compressed=False,
        anchor_compact_ids=quality_ids,
        exact_neighbors=truth,
        qualified_ann_neighbors=selected_observed,
    )
    cfg = panel_config()
    anchors = sample_anchors(RETAINED_ROWS, cfg)
    if not np.array_equal(
        anchors,
        np.sort(
            np.random.RandomState(PANEL_SEED)
            .choice(RETAINED_ROWS, PANEL_ANCHORS, replace=False)
            .astype(np.int64)
        ),
    ):
        raise Round0113Error("R0113 panel anchor selector changed")
    reference = build_hiD_reference(
        X,
        anchors,
        cfg,
        centroids_by_k=None,
        data_identity=_data_identity(assembly, arm=arm),
        convention={
            "row_order": "R0112 cohort-local representative compact order",
            "distance": "cosine via fp32-L2-normalized squared L2",
            "self_exclusion": True,
            "anchor_namespace": "R0113 compact IDs",
            "embedding_prompt": arm,
        },
    )
    reference_path = os.path.join(output, "high-d-reference.npz")
    save_hiD_reference(reference, reference_path)

    query_values = np.load(
        verify_signature(query["outputs"][arm], label=f"R0113 {arm} query reserve"),
        mmap_mode="r",
        allow_pickle=False,
    )
    polish = query["ood"]["pol_Latn"]
    polish_values = np.load(
        verify_signature(
            polish["outputs"][arm], label=f"R0113 {arm} Polish queries"
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    combined_queries = np.concatenate((query_values, polish_values), axis=0)
    combined_copied, combined_copy_receipt = exact_reference_copy_mask(
        source, combined_queries
    )
    training_text_hashes = np.load(
        verify_signature(
            assembly["source_text_hash_index"],
            label="R0113 retained training source-text hashes",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    query_text_hashes = np.load(
        verify_signature(
            query["source_text_row_hashes"],
            label="R0113 FineWeb query source-text hashes",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    polish_text_hashes = np.load(
        verify_signature(
            polish["source_text_row_hashes"],
            label="R0113 Polish query source-text hashes",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    text_copied = _sorted_hash_membership(
        training_text_hashes, query_text_hashes
    )
    polish_text_copied = _sorted_hash_membership(
        training_text_hashes, polish_text_hashes
    )
    embedding_copied = combined_copied[:QUERY_CANDIDATES]
    polish_embedding_copied = combined_copied[QUERY_CANDIDATES:]
    copied = embedding_copied | text_copied
    polish_copied = polish_embedding_copied | polish_text_copied
    mask_path = os.path.join(output, "query-training-copy-mask.npy")
    atomic_save_new_npy(mask_path, copied, immutable=True)
    if np.any(polish_copied):
        raise Round0113Error(
            f"R0113 {arm} Polish diagnostic leaks a training row"
        )
    polish_mask_path = os.path.join(
        output, "polish-query-training-copy-mask.npy"
    )
    atomic_save_new_npy(polish_mask_path, polish_copied, immutable=True)
    body = {
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "arm": arm,
        "retained_rows": RETAINED_ROWS,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "directed_edge_count": int(len(sources)),
        "graph": graph_signature,
        "assembly": assembly_signature,
        "compact_mapping": assembly["mapping"],
        "source": assembly["outputs"][arm],
        "substrate": assembly["substrate"],
        "query_reserve": query_signature,
        "query_training_copy_mask": expected_input_signature(mask_path),
        "query_training_copy_audit": {
            "identity": (
                "union complete stored-embedding-row bytes and complete "
                "source-text UTF-8 bytes"
            ),
            "query_rows": QUERY_CANDIDATES,
            "query_rows_with_exact_embedding_copy": int(
                embedding_copied.sum()
            ),
            "query_rows_with_exact_source_text_copy": int(
                text_copied.sum()
            ),
            "query_rows_rejected_by_union": int(copied.sum()),
            "exact_training_identity_disjoint": not bool(np.any(copied)),
            "embedding_audit": combined_copy_receipt,
        },
        "polish_query_training_copy_mask": expected_input_signature(
            polish_mask_path
        ),
        "polish_query_training_copy_audit": {
            "identity": (
                "union complete stored-embedding-row bytes and complete "
                "source-text UTF-8 bytes"
            ),
            "query_rows": POLISH_QUERY_ROWS,
            "query_rows_with_exact_embedding_copy": int(
                polish_embedding_copied.sum()
            ),
            "query_rows_with_exact_source_text_copy": int(
                polish_text_copied.sum()
            ),
            "query_rows_rejected_by_union": int(polish_copied.sum()),
            "exact_training_identity_disjoint": True,
            "embedding_audit": combined_copy_receipt,
        },
        "search_qualification": {
            "index": "GPU IndexIVFFlat/IP",
            "selected_nprobe": selected_nprobe,
            "cells": cells,
            "training_rows_sha256": ordered_array_sha256(train_rows),
            "quality_rows_sha256": ordered_array_sha256(quality_ids),
            "mean_recall_floor": GRAPH_MEAN_RECALL_FLOOR,
            "p10_recall_floor": GRAPH_P10_RECALL_FLOOR,
        },
        "topology_probe": expected_input_signature(topology_path),
        "high_d_reference": expected_input_signature(reference_path),
        "high_d_reference_key": reference["key"],
        "high_d_reference_content_sha256": reference["content_sha256"],
        "paired_graph_policy": {
            "shared_compact_ids": True,
            "shared_builder_parameters_and_random_seeds": True,
            "separate_arm_graph_bytes": True,
        },
        "performance": {
            "materialize_s": materialize_seconds,
            "ivf_train_s": train_seconds,
            "ivf_add_s": add_seconds,
            "full_search_s": search_seconds,
            "fuzzy_s": fuzzy_seconds,
            "total_wall_s": time.monotonic() - started,
        },
        "training_performed": False,
    }
    manifest = seal(body)
    manifest_path = os.path.join(output, "graph-manifest.json")
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    del X, neighbors, distances, sources, targets, weights, graph, coo, source
    gc.collect()
    return {**manifest, "receipt": expected_input_signature(manifest_path)}


def run_select_queries(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0], label="R0113 matched clean query selection"
    )
    query, query_signature = _load_query_reserve(job)
    rows = np.load(
        verify_signature(query["query_rows"], label="R0113 query global rows"),
        mmap_mode="r",
        allow_pickle=False,
    )
    clean = np.ones(QUERY_CANDIDATES, dtype=bool)
    graph_signatures: dict[str, Any] = {}
    copy_audits: dict[str, Any] = {}
    values: dict[str, np.ndarray] = {}
    for arm in ARMS:
        graph_path = os.path.join(
            str(job["graph_outputs"][arm]), "graph-manifest.json"
        )
        graph = read_sealed(graph_path, label=f"R0113 {arm} graph")
        mask = np.load(
            verify_signature(
                graph["query_training_copy_mask"],
                label=f"R0113 {arm} query copy mask",
            ),
            allow_pickle=False,
        )
        if mask.shape != (QUERY_CANDIDATES,) or mask.dtype != np.bool_:
            raise Round0113Error("R0113 query copy mask geometry changed")
        polish_mask = np.load(
            verify_signature(
                graph["polish_query_training_copy_mask"],
                label=f"R0113 {arm} Polish query copy mask",
            ),
            allow_pickle=False,
        )
        if (
            polish_mask.shape != (POLISH_QUERY_ROWS,)
            or polish_mask.dtype != np.bool_
            or np.any(polish_mask)
        ):
            raise Round0113Error(
                "R0113 Polish query/training disjointness changed"
            )
        clean &= ~mask
        graph_signatures[arm] = expected_input_signature(graph_path)
        copy_audits[arm] = graph["query_training_copy_audit"]
        values[arm] = np.load(
            verify_signature(query["outputs"][arm], label=f"R0113 {arm} queries"),
            mmap_mode="r",
            allow_pickle=False,
        )
    selected: list[int] = []
    seen = {arm: set() for arm in ARMS}
    text_hashes = np.load(
        verify_signature(
            query["source_text_row_hashes"],
            label="R0113 FineWeb query source-text hashes",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        text_hashes.shape != (QUERY_CANDIDATES,)
        or text_hashes.dtype != np.dtype("V32")
    ):
        raise Round0113Error("R0113 FineWeb query text hashes changed")
    seen_texts: set[bytes] = set()
    duplicate_rejections = {arm: 0 for arm in ARMS}
    text_duplicate_rejections = 0
    for position in np.flatnonzero(clean).tolist():
        keys = {
            arm: np.asarray(values[arm][position]).tobytes(order="C")
            for arm in ARMS
        }
        text_key = np.asarray(text_hashes[position]).tobytes(order="C")
        duplicated = [arm for arm in ARMS if keys[arm] in seen[arm]]
        duplicated_text = text_key in seen_texts
        if duplicated or duplicated_text:
            for arm in duplicated:
                duplicate_rejections[arm] += 1
            if duplicated_text:
                text_duplicate_rejections += 1
            continue
        for arm in ARMS:
            seen[arm].add(keys[arm])
        seen_texts.add(text_key)
        selected.append(position)
        if len(selected) == QUERY_ROWS:
            break
    positions = np.asarray(selected, dtype=np.int64)
    if positions.shape != (QUERY_ROWS,):
        raise Round0113Error("R0113 matched clean query reserve is exhausted")
    selected_rows = np.asarray(rows[positions], dtype=np.int64)
    position_path = os.path.join(output, "query-positions.i64.npy")
    rows_path = os.path.join(output, "query-global-rows.i64.npy")
    atomic_save_new_npy(position_path, positions, immutable=True)
    atomic_save_new_npy(rows_path, selected_rows, immutable=True)
    body = {
        "schema": QUERY_SELECTION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "query_reserve": query_signature,
        "graphs": graph_signatures,
        "copy_audits": copy_audits,
        "candidate_rows": QUERY_CANDIDATES,
        "selected_rows": QUERY_ROWS,
        "positions": expected_input_signature(position_path),
        "global_rows": expected_input_signature(rows_path),
        "ordered_global_rows_sha256": ordered_array_sha256(selected_rows),
        "training_copy_rejections": int(np.sum(~clean)),
        "within_reserve_duplicate_rejections": duplicate_rejections,
        "within_reserve_exact_text_duplicate_rejections": (
            text_duplicate_rejections
        ),
        "selection_rule": (
            "first ascending reserve positions clean against both compact "
            "training populations, unique by complete source-text bytes, and "
            "unique by complete stored embedding-row bytes in both arms"
        ),
        "selected_before_training": True,
        "training_performed": False,
    }
    receipt = seal(body)
    path = os.path.join(output, "query-selection.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _new_model(config: Mapping[str, Any]):
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = config["model"]
    optimizer = config["optimizer"]
    graph = config["graph"]
    execution = config["execution"]
    return ParametricUMAP(
        n_components=model["output_dimension"],
        hidden_dim=model["hidden_dimension"],
        n_layers=model["hidden_layers"],
        n_neighbors=graph["k"],
        a=model["a"],
        b=model["b"],
        low_dim_kernel=model["low_dim_kernel"],
        correlation_weight=optimizer["correlation_weight"],
        learning_rate=optimizer["learning_rate"],
        n_epochs=2,
        batch_size=optimizer["batch_size"],
        device="cuda",
        use_batchnorm=model["use_batchnorm"],
        use_dropout=model["use_dropout"],
        clip_grad_norm=optimizer["clip_grad_norm"],
        clip_grad_value=None,
        pos_ratio=optimizer["positive_ratio"],
        architecture=model["architecture"],
        correlation_distance_transform="raw",
        lr_schedule="cosine",
        warmup_steps=optimizer["warmup_successful_updates"],
        total_steps_estimate=optimizer["successful_positive_lr_updates"],
        require_full_budget=True,
        require_graph_manifest=True,
        required_input_pipeline=execution["required_pipeline"],
        use_amp=optimizer["use_amp"],
        positive_target_mode=optimizer["positive_target_mode"],
        reject_neighbors=optimizer["reject_neighbors"],
        anchored_init="none",
        anchor_hold_weight=0.0,
        midnear_enabled=False,
        mn_pairs_per_batch=0,
        weighted_edge_sampling=optimizer["weighted_edge_sampling"],
        gpu_resident_data=execution["gpu_resident_data"],
        gpu_resident_vram_budget_gb=execution[
            "gpu_resident_vram_budget_gb"
        ],
        graph_manifest_path=graph["manifest_path"],
        graph_manifest_sha256=graph["manifest_sha256"],
    )


def _arm(job: Mapping[str, Any]) -> str:
    arm = str(job.get("arm") or "")
    if arm not in ARMS:
        raise Round0113Error(f"unknown R0113 arm {arm!r}")
    return arm


def run_train(active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    import torch

    arm = _arm(job)
    assembly, assembly_signature = _load_assembly(job)
    graph_manifest_path = str(job["graph_manifest"])
    graph_manifest_signature = expected_input_signature(graph_manifest_path)
    graph = load_graph(
        graph_manifest_path,
        expected_sha256=graph_manifest_signature["sha256"],
        arm=arm,
    )
    config, config_sha = train_config(
        arm,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=len(graph["sources"]),
        retained_rows=graph["n_nodes"],
    )
    source = _open_compact(assembly, arm)
    dataset = HostFp16EndpointArray(
        source,
        arm=arm,
        source_signature=assembly["outputs"][arm],
        mapping_signature=assembly["mapping"],
        buffer_rows=BATCH_SIZE,
    )
    wrapper = PromptTrainingInput(dataset, graph, arm=arm)
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0113 {arm} train output"
    )
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": _schema("production-config"),
            "round_id": ROUND_ID,
            "arm": arm,
            "config": config,
            "config_sha256": config_sha,
        },
        immutable=True,
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _new_model(config)
    model._max_train_steps = SUCCESSFUL_UPDATES
    model._bench_warmup = PERFORMANCE_WARMUP_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = PERFORMANCE_WINDOWS
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    model.fit(
        wrapper,
        low_memory=True,
        verbose=False,
        n_processes=6,
        random_state=SEED,
        resample_negatives=False,
        precomputed_edges_path=graph["signature"]["canonical_path"],
        use_wandb=False,
    )
    wall = time.monotonic() - started
    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    exact = {
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": len(graph["sources"]),
    }
    mismatches.update(
        {
            key: {"expected": value, "observed": accounting.get(key)}
            for key, value in exact.items()
            if accounting.get(key) != value
        }
    )
    expected_rows = SUCCESSFUL_UPDATES * BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        int(runtime["source_rows_gathered"]) != expected_rows
        or int(runtime["destination_rows_gathered"]) != expected_rows
        or int(runtime["host_prefetch_consumer_batches"]) != SUCCESSFUL_UPDATES
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": runtime,
        }
    expected_positive_draws = SUCCESSFUL_UPDATES * POSITIVE_ROWS_PER_UPDATE
    if (
        int(runtime["weight_emitted_draws"]) != expected_positive_draws
        or int(runtime["weight_acceptances"])
        != (
            int(runtime["weight_emitted_draws"])
            + int(runtime["weight_buffered_draws"])
        )
        or int(runtime["weight_proposals"]) < int(runtime["weight_acceptances"])
        or not 0 < float(runtime["weight_acceptance_rate"]) <= 1
    ):
        mismatches["weighted_rejection_accounting"] = {
            "expected_positive_draws": expected_positive_draws,
            "runtime": runtime,
        }
    if mismatches:
        raise Round0113Error(f"R0113 {arm} train accounting failed: {mismatches}")
    synchronize_runtime_counters(accounting, runtime)
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES)
        / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if profiler.get("aborted") is not False or rate < TRAIN_MINIMUM_UPDATES_PER_S:
        raise Round0113Error(f"R0113 {arm} performance admission failed")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    body = {
        "schema": _schema("train-receipt"),
        "round_id": ROUND_ID,
        "arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "model": expected_input_signature(model_path),
        "assembly": assembly_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_s": wall,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
        },
        "memory": {
            "device_total_bytes": int(total_bytes),
            "post_train_free_bytes": int(free_bytes),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
        "training_performed": True,
        "optimizer_updates": SUCCESSFUL_UPDATES,
        "map_decision_made": False,
    }
    receipt = seal(body)
    receipt_path = os.path.join(output, "train-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del model, wrapper, dataset, source, graph
    torch.cuda.empty_cache()
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _authenticate_model(
    job: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    arm = _arm(job)
    assembly, _assembly_signature = _load_assembly(job)
    graph_manifest_path = str(job["graph_manifest"])
    graph_manifest_signature = expected_input_signature(graph_manifest_path)
    graph = load_graph(
        graph_manifest_path,
        expected_sha256=graph_manifest_signature["sha256"],
        arm=arm,
    )
    config, config_sha = train_config(
        arm,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=len(graph["sources"]),
        retained_rows=graph["n_nodes"],
    )
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train = read_sealed(train_path, label=f"R0113 {arm} train receipt")
    if (
        train.get("arm") != arm
        or train.get("production_config_sha256") != config_sha
        or train.get("graph_manifest") != graph["manifest_signature"]
    ):
        raise Round0113Error(f"R0113 {arm} train/config binding changed")
    model_path = verify_signature(train["model"], label=f"R0113 {arm} model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
    expected = config["model"]
    observed = {
        "architecture": model.architecture,
        "input_dimension": model.input_dim,
        "hidden_dimension": model.hidden_dim,
        "hidden_layers": model.n_layers,
        "output_dimension": model.n_components,
        "use_batchnorm": model.use_batchnorm,
        "use_dropout": model.use_dropout,
        "low_dim_kernel": model.low_dim_kernel,
        "a": model.a,
        "b": model.b,
    }
    if observed != expected:
        raise Round0113Error(f"R0113 {arm} model architecture changed")
    return model, train, assembly, graph


def _load_query_selection(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    path = os.path.join(
        str(job["query_selection_output"]), "query-selection.json"
    )
    receipt = read_sealed(path, label="R0113 matched query selection")
    if (
        receipt.get("schema") != QUERY_SELECTION_SCHEMA
        or int(receipt.get("selected_rows", -1)) != QUERY_ROWS
        or receipt.get("selected_before_training") is not True
    ):
        raise Round0113Error("R0113 matched query selection changed")
    positions = np.load(
        verify_signature(receipt["positions"], label="R0113 query positions"),
        allow_pickle=False,
    )
    if (
        positions.shape != (QUERY_ROWS,)
        or positions.dtype != np.int64
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise Round0113Error("R0113 query positions are malformed")
    return receipt, positions


def run_evaluate(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        build_query_truth,
        cross_knn,
        ffr_from_neighbors,
        load_hiD_reference,
        recall_at_k_from_neighbors,
        save_query_truth,
        score_panel,
    )

    arm = _arm(job)
    other = "document" if arm == "raw" else "raw"
    model, train, assembly, graph = _authenticate_model(job)
    query, query_signature = _load_query_reserve(job)
    selection, positions = _load_query_selection(job)
    source = _open_compact(assembly, arm)
    query_values = {
        name: np.asarray(
            np.load(
                verify_signature(
                    query["outputs"][name], label=f"R0113 {name} query reserve"
                ),
                mmap_mode="r",
                allow_pickle=False,
            )[positions],
            dtype=np.float16,
        )
        for name in ARMS
    }
    polish = query["ood"]["pol_Latn"]
    polish_values = {
        name: np.asarray(
            np.load(
                verify_signature(
                    polish["outputs"][name],
                    label=f"R0113 {name} Polish queries",
                ),
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.float16,
        )
        for name in ARMS
    }
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0113 {arm} evaluation"
    )
    started = time.monotonic()
    coordinates = np.asarray(
        model.transform(source, batch_size=8192), dtype=np.float32
    )
    matched_coordinates = np.asarray(
        model.transform(query_values[arm], batch_size=8192), dtype=np.float32
    )
    cross_coordinates = np.asarray(
        model.transform(query_values[other], batch_size=8192), dtype=np.float32
    )
    polish_matched_coordinates = np.asarray(
        model.transform(polish_values[arm], batch_size=8192), dtype=np.float32
    )
    polish_cross_coordinates = np.asarray(
        model.transform(polish_values[other], batch_size=8192), dtype=np.float32
    )
    if (
        coordinates.shape != (RETAINED_ROWS, 2)
        or matched_coordinates.shape != (QUERY_ROWS, 2)
        or cross_coordinates.shape != (QUERY_ROWS, 2)
        or polish_matched_coordinates.shape != (POLISH_QUERY_ROWS, 2)
        or polish_cross_coordinates.shape != (POLISH_QUERY_ROWS, 2)
        or not np.isfinite(coordinates).all()
        or not np.isfinite(matched_coordinates).all()
        or not np.isfinite(cross_coordinates).all()
        or not np.isfinite(polish_matched_coordinates).all()
        or not np.isfinite(polish_cross_coordinates).all()
    ):
        raise Round0113Error(f"R0113 {arm} transform output is invalid")
    coordinate_paths = {
        "training": os.path.join(output, "coordinates.npy"),
        "matched_queries": os.path.join(output, "matched-query-coordinates.npy"),
        "cross_queries": os.path.join(output, "cross-query-coordinates.npy"),
        "polish_matched_queries": os.path.join(
            output, "polish-matched-query-coordinates.npy"
        ),
        "polish_cross_queries": os.path.join(
            output, "polish-cross-query-coordinates.npy"
        ),
    }
    for key, values in (
        ("training", coordinates),
        ("matched_queries", matched_coordinates),
        ("cross_queries", cross_coordinates),
        ("polish_matched_queries", polish_matched_coordinates),
        ("polish_cross_queries", polish_cross_coordinates),
    ):
        atomic_save_new_npy(coordinate_paths[key], values, immutable=True)

    cfg = panel_config()
    X = L2NormalizedArray(source)
    reference = load_hiD_reference(
        graph["manifest"]["high_d_reference"]["canonical_path"],
        expected_key=graph["manifest"]["high_d_reference_key"],
    )
    reference_identity = {
        "data_identity": _data_identity(assembly, arm=arm),
        "convention": {
            "row_order": "R0112 cohort-local representative compact order",
            "distance": "cosine via fp32-L2-normalized squared L2",
            "self_exclusion": True,
            "anchor_namespace": "R0113 compact IDs",
            "embedding_prompt": arm,
        },
    }
    panel = score_panel(
        X,
        coordinates,
        config=cfg,
        centroids_by_k=None,
        hiD_reference=reference,
        reference_identity=reference_identity,
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "arm": arm,
            "release_sha": active["manifest"]["release_sha"],
            "train_receipt": expected_input_signature(
                os.path.join(
                    str(job["train_output"]), "train-receipt.json"
                )
            ),
            "query_selection": selection["identity_sha256"],
        },
    )
    query_global_rows = np.load(
        verify_signature(selection["global_rows"], label="R0113 selected query rows"),
        allow_pickle=False,
    )
    polish_global_rows = np.load(
        verify_signature(
            polish["query_rows"], label="R0113 Polish selected query rows"
        ),
        allow_pickle=False,
    )
    cells = (
        (
            "fineweb",
            "matched",
            arm,
            query_values[arm],
            matched_coordinates,
            ordered_array_sha256(query_global_rows),
        ),
        (
            "fineweb",
            "cross",
            other,
            query_values[other],
            cross_coordinates,
            ordered_array_sha256(query_global_rows),
        ),
        (
            "pol_Latn",
            "matched",
            arm,
            polish_values[arm],
            polish_matched_coordinates,
            ordered_array_sha256(polish_global_rows),
        ),
        (
            "pol_Latn",
            "cross",
            other,
            polish_values[other],
            polish_cross_coordinates,
            ordered_array_sha256(polish_global_rows),
        ),
    )
    cell_identities: list[dict[str, Any]] = []
    cell_ranges: dict[tuple[str, str], tuple[int, int]] = {}
    cursor = 0
    for source_name, role, convention, values, _low, rows_sha in cells:
        stop = cursor + len(values)
        cell_ranges[(source_name, role)] = (cursor, stop)
        cell_identities.append(
            {
                "source": source_name,
                "role": role,
                "query_embedding_convention": convention,
                "map_embedding_convention": arm,
                "row_range": [cursor, stop],
                "source_rows_sha256": rows_sha,
                "ordered_fp16_sha256": ordered_array_sha256(values),
                "disjoint_from_training_complete_stored_rows": True,
            }
        )
        cursor = stop
    combined_values = np.concatenate(
        [np.asarray(cell[3], dtype=np.float16) for cell in cells], axis=0
    )
    combined_low = np.concatenate(
        [np.asarray(cell[4], dtype=np.float32) for cell in cells], axis=0
    )
    expected_combined_rows = 2 * QUERY_ROWS + 2 * POLISH_QUERY_ROWS
    if (
        cursor != expected_combined_rows
        or combined_values.shape != (expected_combined_rows, DIMENSION)
        or combined_low.shape != (expected_combined_rows, 2)
    ):
        raise Round0113Error("R0113 combined query panel did not close")
    query_identity = {
        "schema": "round0113-combined-query-identity-v1",
        "ordered_cells": cell_identities,
        "ordered_combined_fp16_sha256": ordered_array_sha256(combined_values),
    }
    truth = build_query_truth(
        L2NormalizedArray(combined_values),
        X,
        cfg=cfg,
        corpus_identity=_data_identity(assembly, arm=arm),
        query_identity=query_identity,
        k=10,
    )
    truth_path = os.path.join(output, "combined-query-truth-k10.npz")
    save_query_truth(truth, truth_path)
    truth_signature = expected_input_signature(truth_path)
    high10_all = np.asarray(truth["neighbors"], dtype=np.int64)
    k_fraction = max(cfg.k_hit, int(math.ceil(cfg.frac * RETAINED_ROWS)))
    low_fraction_all = cross_knn(
        combined_low, coordinates, k_fraction, cfg, hi_dim=False
    )
    low50_all = cross_knn(
        combined_low, coordinates, 50, cfg, hi_dim=False
    )
    projections: dict[str, Any] = {}
    polish_projections: dict[str, Any] = {}
    for source_name, role, convention, values, _low, _rows_sha in cells:
        start, stop = cell_ranges[(source_name, role)]
        high10 = high10_all[start:stop, : cfg.k_hit]
        low_fraction = low_fraction_all[start:stop]
        low10 = low_fraction[:, : cfg.k_hit]
        report = {
            "ffr": float(
                ffr_from_neighbors(high10, low_fraction, cfg.k_hit)
            ),
            "recall_at_10": float(
                recall_at_k_from_neighbors(high10, low10, cfg.k_hit)
            ),
            "recall_at_50_of_high10": _recall(
                high10, low50_all[start:stop], cfg.k_hit
            ),
            "queries": len(values),
            "k_fraction": k_fraction,
            "truth": truth_signature,
            "truth_row_range": [start, stop],
            "query_embedding_convention": convention,
        }
        target = projections if source_name == "fineweb" else polish_projections
        target[role] = report
    low51 = cross_knn(
        np.asarray(coordinates[reference["anchor_ids"]], dtype=np.float32),
        coordinates,
        51,
        cfg,
        hi_dim=False,
    )
    low50 = _without_self(low51, reference["anchor_ids"], 50)
    transductive_recall50 = _recall(
        reference["hi_hit"], low50, cfg.k_hit
    )
    metrics = {
        "ffr": float(panel["ffr"]),
        "density": float(panel["density"]),
        "recall_at_10": float(panel["recall@k"]),
        "oos_recall_at_10": projections["matched"]["recall_at_10"],
        "oos_recall_at_50": projections["matched"][
            "recall_at_50_of_high10"
        ],
    }
    guards = panel.get("guards") or {}
    execution_gates = {
        "finite_noncollapsed_coordinates": bool(
            guards.get("coords_finite") is True
            and guards.get("coords_collapsed") is False
            and guards.get("emb_finite") is True
            and guards.get("emb_zero_rows") == 0
        ),
        "transductive_recall50_gt_recall10": (
            transductive_recall50 > metrics["recall_at_10"]
        ),
        "matched_projection_recall50_gt_recall10": (
            projections["matched"]["recall_at_50_of_high10"]
            > projections["matched"]["recall_at_10"]
        ),
        "exact_update_closure": bool(
            (train.get("train_checks") or {}).get("exact_update_closure")
        ),
        "zero_numerical_skips": bool(
            (train.get("train_checks") or {}).get("zero_numerical_skips")
        ),
        "no_pipeline_stamp_drift": bool(
            (train.get("train_checks") or {}).get("no_pipeline_stamp_drift")
        ),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise Round0113Error(f"R0113 {arm} metrics are nonfinite")
    if not all(
        np.isfinite(float(value))
        for report in polish_projections.values()
        for key, value in report.items()
        if key in {"ffr", "recall_at_10", "recall_at_50_of_high10"}
    ):
        raise Round0113Error(f"R0113 {arm} Polish metrics are nonfinite")
    body = {
        "schema": _schema("prompt-arm-score"),
        "round_id": ROUND_ID,
        "arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "train_receipt": expected_input_signature(
            os.path.join(str(job["train_output"]), "train-receipt.json")
        ),
        "assembly": train["assembly"],
        "graph_manifest": graph["manifest_signature"],
        "query_reserve": query_signature,
        "query_selection": expected_input_signature(
            os.path.join(
                str(job["query_selection_output"]), "query-selection.json"
            )
        ),
        "coordinates": {
            key: expected_input_signature(path)
            for key, path in coordinate_paths.items()
        },
        "high_d_reference": graph["manifest"]["high_d_reference"],
        "combined_query_truth": truth_signature,
        "combined_query_identity": query_identity,
        "panel": panel,
        "transductive_recall_at_50_of_high10": transductive_recall50,
        "projections": projections,
        "ood": {
            "pol_Latn": {
                "role": (
                    "diagnostic-only; measures attachment to the 2M FineWeb "
                    "atlas and cannot alter the prompt-transfer gate"
                ),
                "query_source": polish["source_text"],
                "query_rows": polish["query_rows"],
                "projections": polish_projections,
                "matched_recall50_to_fineweb_oos_ratio": (
                    polish_projections["matched"][
                        "recall_at_50_of_high10"
                    ]
                    / projections["matched"]["recall_at_50_of_high10"]
                    if projections["matched"]["recall_at_50_of_high10"] > 0
                    else None
                ),
                "matched_recall50_gt_recall10": (
                    polish_projections["matched"][
                        "recall_at_50_of_high10"
                    ]
                    > polish_projections["matched"]["recall_at_10"]
                ),
            }
        },
        "projection_ffr_role": "diagnostic-only",
        "metrics": metrics,
        "execution_gates": execution_gates,
        "wall_s": time.monotonic() - started,
    }
    score = seal(body)
    path = os.path.join(output, "score.json")
    atomic_write_new_json(path, score, immutable=True)
    del model, source, coordinates, X
    gc.collect()
    return {**score, "receipt": expected_input_signature(path)}


def run_decide(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0], label="R0113 paired prompt decision"
    )
    scores: dict[str, Any] = {}
    score_signatures: dict[str, Any] = {}
    topology: dict[str, Any] = {}
    graph_signatures: dict[str, Any] = {}
    for arm in ARMS:
        score_path = os.path.join(str(job["score_outputs"][arm]), "score.json")
        score = read_sealed(score_path, label=f"R0113 {arm} score")
        if (
            score.get("arm") != arm
            or set(score.get("metrics") or {}) != set(DECISION_METRICS)
            or "pol_Latn" not in (score.get("ood") or {})
        ):
            raise Round0113Error(f"R0113 {arm} score contract changed")
        scores[arm] = score
        score_signatures[arm] = expected_input_signature(score_path)
        graph_path = os.path.join(
            str(job["graph_outputs"][arm]), "graph-manifest.json"
        )
        graph = read_sealed(graph_path, label=f"R0113 {arm} graph manifest")
        graph_signatures[arm] = expected_input_signature(graph_path)
        probe_path = verify_signature(
            graph["topology_probe"], label=f"R0113 {arm} topology probe"
        )
        with np.load(probe_path, allow_pickle=False) as archive:
            topology[arm] = {
                name: np.asarray(archive[name])
                for name in archive.files
            }
    if not np.array_equal(
        topology["raw"]["anchor_compact_ids"],
        topology["document"]["anchor_compact_ids"],
    ):
        raise Round0113Error("R0113 topology anchors differ between arms")
    exact_overlap = _recall_rows(
        topology["document"]["exact_neighbors"],
        topology["raw"]["exact_neighbors"],
    )
    ann_overlap = _recall_rows(
        topology["document"]["qualified_ann_neighbors"],
        topology["raw"]["qualified_ann_neighbors"],
    )
    decision = paired_decision(scores["raw"], scores["document"])
    polish_contrast: dict[str, Any] = {}
    for role in ("matched", "cross"):
        polish_contrast[role] = {}
        for metric in ("ffr", "recall_at_10", "recall_at_50_of_high10"):
            raw_value = float(
                scores["raw"]["ood"]["pol_Latn"]["projections"][role][metric]
            )
            document_value = float(
                scores["document"]["ood"]["pol_Latn"]["projections"][role][
                    metric
                ]
            )
            polish_contrast[role][metric] = {
                "raw": raw_value,
                "document": document_value,
                "document_minus_raw": document_value - raw_value,
                "document_to_raw_ratio": (
                    document_value / raw_value if raw_value != 0 else None
                ),
            }
    released = ["jina-fineweb-2m-prompt-map-contrast-v1"]
    if decision["passed"]:
        released.append("jina-fineweb-2m-document-prompt-map-transfer-v1")
    body = {
        "schema": _schema("paired-prompt-decision"),
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "scores": score_signatures,
        "graphs": graph_signatures,
        "registered_decision": decision,
        "polish_ood_prompt_contrast": {
            "role": "diagnostic-only; excluded from registered_decision",
            "query_rows": POLISH_QUERY_ROWS,
            "matched_and_cross_convention": polish_contrast,
        },
        "topology_shift": {
            "anchors": GRAPH_QUALITY_ROWS,
            "k": GRAPH_K - 1,
            "exact_high_d_neighbor_overlap": {
                "mean": float(np.mean(exact_overlap)),
                "p10": float(np.percentile(exact_overlap, 10)),
                "min": float(np.min(exact_overlap)),
            },
            "qualified_ann_neighbor_overlap": {
                "mean": float(np.mean(ann_overlap)),
                "p10": float(np.percentile(ann_overlap, 10)),
                "min": float(np.min(ann_overlap)),
            },
        },
        "capabilities_produced": released,
        "production_ready": False,
        "complete_sae_corpus_ready": False,
        "one_seed_screen": True,
        "training_performed": True,
        "optimizer_updates_per_arm": SUCCESSFUL_UPDATES,
    }
    receipt = seal(body)
    path = os.path.join(output, "decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0113Error("R0113 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    action = selected.get("action")
    if action == "embed_query_reserve":
        return run_embed_queries(active, selected)
    if action == "assemble_compact_arrays":
        return run_assemble(active, selected)
    if action == "build_arm_graph":
        return run_build_graph(active, selected)
    if action == "select_matched_queries":
        return run_select_queries(active, selected)
    if action == "train_arm":
        return run_train(active, selected)
    if action == "evaluate_arm":
        return run_evaluate(active, selected)
    if action == "decide_prompt_contrast":
        return run_decide(active, selected)
    raise Round0113Error(f"unknown R0113 action: {action!r}")
