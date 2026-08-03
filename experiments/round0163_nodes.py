"""Execute the CPU-only prompted-English 8M representative census."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0162_prompted_english_staging import (
    DIMENSION,
    DTYPE,
    VIEW_ROWS,
)
from basemap.round0163_prompted_english_census import (
    CAPABILITY,
    FAMILY_SOURCES,
    HOST_CAPABILITY,
    PROJECTION_POSITIONS,
    ROUND_ID,
    SCHEMA,
    Round0163Error,
    embedding_text_relation,
    population_identity,
    seal,
    union_representatives,
    validate_seal,
)


@dataclass(frozen=True)
class Slice:
    canonical_start: int
    canonical_stop: int
    source_start: int
    source_stop: int
    signature: dict[str, Any]


def _read_sealed(signature: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    observed = expected_input_signature(str(signature.get("canonical_path") or ""))
    if observed != dict(signature):
        raise Round0163Error(f"{label} bytes changed")
    with open(observed["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0163Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value


def _validate_signature(signature: Mapping[str, Any], *, label: str) -> None:
    path = str(signature.get("canonical_path") or "")
    try:
        stat = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise Round0163Error(f"{label} is unavailable") from error
    if (
        os.path.islink(path)
        or not os.path.isfile(path)
        or stat.st_size != int(signature.get("bytes", -1))
        or stat.st_nlink != 1
    ):
        raise Round0163Error(f"{label} file identity changed")
    # Every selected source is below 1 GiB.  Rehash it here so the CPU node's
    # data read cannot silently consume a path that drifted after queue prep.
    expected = {
        key: signature[key]
        for key in ("kind", "canonical_path", "bytes", "sha256")
    }
    if expected_input_signature(path) != expected:
        raise Round0163Error(f"{label} payload changed")


class ChunkedFp16:
    def __init__(self, slices: Sequence[Slice], *, label: str):
        self.slices = list(slices)
        self.label = label
        cursor = 0
        for item in self.slices:
            if (
                item.canonical_start != cursor
                or item.canonical_stop <= item.canonical_start
                or item.source_stop - item.source_start
                != item.canonical_stop - item.canonical_start
            ):
                raise Round0163Error(f"{label} slices are not gap-free")
            cursor = item.canonical_stop
        if cursor != VIEW_ROWS:
            raise Round0163Error(f"{label} does not cover the first 8M rows")
        self.stops = np.asarray(
            [item.canonical_stop for item in self.slices], dtype=np.int64
        )
        self._arrays: dict[str, np.ndarray] = {}

    def _open(self, item: Slice) -> np.ndarray:
        path = str(item.signature["canonical_path"])
        values = self._arrays.get(path)
        if values is None:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            self._arrays[path] = values
        if (
            values.ndim != 2
            or values.shape[1] != DIMENSION
            or values.dtype != np.dtype(DTYPE)
            or item.source_start < 0
            or item.source_stop > len(values)
        ):
            raise Round0163Error(f"{self.label} source geometry changed")
        return values

    def projected_keys(self) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.monotonic()
        projection = np.empty(
            (VIEW_ROWS, len(PROJECTION_POSITIONS)), dtype=np.uint16
        )
        for item in self.slices:
            values = self._open(item)
            selected = np.asarray(
                values[item.source_start:item.source_stop, PROJECTION_POSITIONS],
                dtype=np.float16,
                order="C",
            ).view(np.uint16)
            projection[item.canonical_start:item.canonical_stop] = selected
            del values, selected
        return projection, {
            "candidate_projection_positions": PROJECTION_POSITIONS.tolist(),
            "projection_wall_seconds": time.monotonic() - started,
        }

    def rows(self, canonical_rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(canonical_rows, dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= VIEW_ROWS):
            raise Round0163Error(f"{self.label} gather rows are malformed")
        output = np.empty((len(rows), DIMENSION), dtype=np.float16)
        owners = np.searchsorted(self.stops, rows, side="right")
        for owner in np.unique(owners):
            positions = np.flatnonzero(owners == owner)
            item = self.slices[int(owner)]
            local = rows[positions] - item.canonical_start + item.source_start
            values = self._open(item)
            output[positions] = values[local]
            del values
        return output

    def write_compact(self, path: str, mapping: np.ndarray) -> None:
        rows = np.asarray(mapping, dtype=np.int64)
        output = np.memmap(
            path,
            mode="w+",
            dtype=np.dtype(DTYPE),
            shape=(len(rows), DIMENSION),
        )
        written = 0
        for item in self.slices:
            left = int(np.searchsorted(rows, item.canonical_start, side="left"))
            right = int(np.searchsorted(rows, item.canonical_stop, side="left"))
            if right <= left:
                continue
            local = rows[left:right] - item.canonical_start + item.source_start
            values = self._open(item)
            output[written:written + len(local)] = values[local]
            written += len(local)
            del values
        if written != len(rows):
            raise Round0163Error("prompted compact matrix did not close")
        output.flush()
        del output


def _families_from_fp16(
    source: ChunkedFp16,
) -> tuple[list[list[int]], dict[str, Any]]:
    started = time.monotonic()
    projection, report = source.projected_keys()
    key_dtype = np.dtype((np.void, projection.dtype.itemsize * projection.shape[1]))
    keys = projection.view(key_dtype).reshape(-1)
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    equal = ordered[1:] == ordered[:-1]
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(~equal).astype(np.int64) + 1)
    )
    stops = np.concatenate((starts[1:], np.asarray([VIEW_ROWS], dtype=np.int64)))
    repeated = np.flatnonzero(stops - starts > 1)
    families: list[list[int]] = []
    collision_splits = 0
    for group in repeated.tolist():
        members = np.sort(order[int(starts[group]):int(stops[group])]).astype(np.int64)
        rows = source.rows(members)
        exact: dict[bytes, list[int]] = {}
        for member, row in zip(members.tolist(), rows, strict=True):
            exact.setdefault(np.asarray(row).tobytes(order="C"), []).append(int(member))
        collision_splits += max(len(exact) - 1, 0)
        for family in exact.values():
            if len(family) >= 2:
                families.append(sorted(family))
        del rows
    families.sort(key=lambda family: (family[0], len(family), family))
    report.update({
        "identity": "complete stored fp16 row bytes",
        "candidate_repeated_groups": int(len(repeated)),
        "candidate_collision_splits": int(collision_splits),
        "exact_nontrivial_family_count": len(families),
        "rows_in_exact_nontrivial_families": sum(len(family) for family in families),
        "maximum_exact_family_size": max((len(family) for family in families), default=1),
        "family_examples": families[:32],
        "wall_seconds": time.monotonic() - started,
    })
    del projection, keys, order, ordered, equal, starts, stops, repeated
    gc.collect()
    return families, report


def _iter_texts(
    layout: Sequence[Mapping[str, Any]], rows: np.ndarray
) -> Iterator[tuple[int, str]]:
    import pyarrow.parquet as pq

    selected = np.asarray(rows, dtype=np.int64)
    yielded = 0
    for item in layout:
        start = int(item["canonical_start"])
        stop = int(item["canonical_stop"])
        left = int(np.searchsorted(selected, start, side="left"))
        right = int(np.searchsorted(selected, stop, side="left"))
        if right <= left:
            continue
        targets = selected[left:right] - start + int(item["shard_start"])
        parquet = pq.ParquetFile(item["text"]["canonical_path"])
        if int(parquet.metadata.num_rows) != int(item["shard_rows"]):
            raise Round0163Error("prompted-English text shard rows changed")
        cursor = 0
        target_cursor = 0
        for batch in parquet.iter_batches(batch_size=65_536, columns=[item["text_column"]]):
            batch_stop = cursor + len(batch)
            batch_left = int(np.searchsorted(targets, cursor, side="left"))
            batch_right = int(np.searchsorted(targets, batch_stop, side="left"))
            if batch_right > batch_left:
                local = targets[batch_left:batch_right] - cursor
                texts = batch.column(0).take(local.tolist()).to_pylist()
                for offset, text in enumerate(texts, start=batch_left):
                    if not isinstance(text, str):
                        raise Round0163Error("prompted-English source text is not a string")
                    yield int(selected[left + offset]), text
                    yielded += 1
                target_cursor = batch_right
            cursor = batch_stop
            if target_cursor == len(targets):
                break
        if target_cursor != len(targets):
            raise Round0163Error("prompted-English text fetch did not close")
    if yielded != len(selected):
        raise Round0163Error("prompted-English text layout did not cover selected rows")


def _text_families(
    layout: Sequence[Mapping[str, Any]],
) -> tuple[list[list[int]], dict[str, Any], np.ndarray]:
    started = time.monotonic()
    all_rows = np.arange(VIEW_ROWS, dtype=np.int64)
    hashes = np.empty(VIEW_ROWS, dtype="V32")
    for index, (row, text) in enumerate(_iter_texts(layout, all_rows)):
        if row != index:
            raise Round0163Error("prompted-English text order changed")
        hashes[index] = hashlib.sha256(text.encode("utf-8")).digest()
    order = np.argsort(hashes, kind="stable")
    ordered = hashes[order]
    equal = ordered[1:] == ordered[:-1]
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(~equal).astype(np.int64) + 1)
    )
    stops = np.concatenate((starts[1:], np.asarray([VIEW_ROWS], dtype=np.int64)))
    repeated = np.flatnonzero(stops - starts > 1)
    candidates = (
        np.sort(np.concatenate([
            order[int(starts[group]):int(stops[group])] for group in repeated.tolist()
        ])).astype(np.int64)
        if len(repeated)
        else np.empty(0, dtype=np.int64)
    )
    bytes_by_row = {
        row: text.encode("utf-8") for row, text in _iter_texts(layout, candidates)
    }
    families: list[list[int]] = []
    collision_splits = 0
    for group in repeated.tolist():
        members = order[int(starts[group]):int(stops[group])]
        exact: dict[bytes, list[int]] = {}
        for member in members.tolist():
            exact.setdefault(bytes_by_row[int(member)], []).append(int(member))
        collision_splits += max(len(exact) - 1, 0)
        for family in exact.values():
            if len(family) >= 2:
                families.append(sorted(family))
    families.sort(key=lambda family: (family[0], len(family), family))
    report = {
        "identity": "complete source-text UTF-8 bytes",
        "candidate_hash": "SHA-256 over each complete UTF-8 text",
        "candidate_repeated_groups": int(len(repeated)),
        "candidate_collision_splits": int(collision_splits),
        "exact_nontrivial_family_count": len(families),
        "rows_in_exact_nontrivial_families": sum(len(family) for family in families),
        "maximum_exact_family_size": max((len(family) for family in families), default=1),
        "family_examples": families[:32],
        "wall_seconds": time.monotonic() - started,
    }
    return families, report, hashes


def _document_slices(view: Mapping[str, Any]) -> list[Slice]:
    output: list[Slice] = []
    for item in view.get("staged_slices") or []:
        start, stop = (int(value) for value in item["canonical_row_range"])
        source_start, source_stop = (int(value) for value in item["source_array_row_slice"])
        output.append(Slice(start, stop, source_start, source_stop, dict(item["staged_output"])))
    return output


def _source_layouts(
    r0116: Mapping[str, Any], r0120: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[Slice], list[dict[str, Any]]]:
    text_layout: list[dict[str, Any]] = []
    raw_slices: list[Slice] = []
    inputs: list[dict[str, Any]] = []
    for round_id, manifest in (("0116", r0116), ("0120", r0120)):
        for item in manifest.get("source_layout") or []:
            if round_id == "0120":
                start = int(item["r0087_global_row_start"])
                stop = int(item["r0087_global_row_stop"])
            else:
                start = int(item["corpus_global_row_start"])
                stop = int(item["corpus_global_row_stop"])
            if start >= VIEW_ROWS:
                continue
            selected_stop = min(stop, VIEW_ROWS)
            selected_rows = selected_stop - start
            raw = dict(item["accepted_raw_embedding"])
            text = dict(item["text"])
            raw_slices.append(Slice(
                start,
                selected_stop,
                int(item["shard_row_start"]),
                int(item["shard_row_start"]) + selected_rows,
                raw,
            ))
            text_layout.append({
                "canonical_start": start,
                "canonical_stop": selected_stop,
                "shard_start": int(item["shard_row_start"]),
                "shard_stop": int(item["shard_row_start"]) + selected_rows,
                "shard_rows": int(item["shard_rows"]),
                "text": text,
                "text_column": str(item["text_column"]),
            })
            inputs.extend((raw, text))
    if [item.canonical_start for item in raw_slices] != [item["canonical_start"] for item in text_layout]:
        raise Round0163Error("raw/text source layouts disagree")
    return text_layout, raw_slices, inputs


def run_census(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0163Error("R0163 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0163Error("R0163 is CPU-only")
    started = time.monotonic()
    r0162 = _read_sealed(job["canonical_layout"], label="accepted R0162 layout")
    view = _read_sealed(job["first8m_view"], label="accepted R0162 first8m view")
    r0116 = _read_sealed(job["r0116_manifest"], label="accepted R0116 manifest")
    r0120 = _read_sealed(job["r0120_manifest"], label="accepted R0120 manifest")
    if (
        r0162.get("schema") != "jina-document-english-9p126m-canonical-layout-v1"
        or view.get("schema") != "jina-document-english-first8m-view-v1"
        or view.get("layout_identity") != r0162.get("layout_identity")
        or int(view.get("rows", -1)) != VIEW_ROWS
        or view.get("dtype") != DTYPE
    ):
        raise Round0163Error("accepted R0162 view contract changed")

    text_layout, raw_slices, source_inputs = _source_layouts(r0116, r0120)
    document_slices = _document_slices(view)
    expected_inputs = [*source_inputs, *[item.signature for item in document_slices]]
    expected_keys = {
        (item["canonical_path"], item["bytes"], item["sha256"]) for item in job["payload_inputs"]
    }
    observed_keys = {
        (item["canonical_path"], item["bytes"], item["sha256"]) for item in expected_inputs
    }
    if expected_keys != observed_keys:
        raise Round0163Error("R0163 payload input set changed after queue preparation")
    for index, signature in enumerate(job["payload_inputs"]):
        _validate_signature(signature, label=f"R0163 payload input {index}")

    raw = ChunkedFp16(raw_slices, label="raw fp16")
    document = ChunkedFp16(document_slices, label="Document fp16")
    text_families, text_report, text_hashes = _text_families(text_layout)
    raw_families, raw_report = _families_from_fp16(raw)
    document_families, document_report = _families_from_fp16(document)
    families = {
        "source_text": text_families,
        "raw_fp16": raw_families,
        "document_fp16": document_families,
    }
    mapping, excluded, union_report = union_representatives(families)
    relation = {
        "raw_fp16": embedding_text_relation(raw_families, text_families),
        "document_fp16": embedding_text_relation(document_families, text_families),
    }
    output = create_fresh_directory(str(job["outputs"][0]), label="R0163 census output")
    mapping_path = os.path.join(output, "compact-to-canonical.i64.npy")
    excluded_path = os.path.join(output, "excluded-canonical.i64.npy")
    text_hash_path = os.path.join(output, "source-text-sha256-sorted.v32.npy")
    atomic_save_new_npy(mapping_path, mapping, immutable=True)
    atomic_save_new_npy(excluded_path, excluded, immutable=True)
    retained_hashes = np.sort(text_hashes[mapping], kind="stable")
    if len(retained_hashes) > 1 and np.any(retained_hashes[1:] == retained_hashes[:-1]):
        raise Round0163Error("representative selector leaves exact source-text duplicates")
    atomic_save_new_npy(text_hash_path, retained_hashes, immutable=True)
    compact_path = os.path.join(output, "document-compact.f16")
    compact_started = time.monotonic()
    atomic_build_new_file(
        compact_path,
        lambda temporary: document.write_compact(temporary, mapping),
        immutable=True,
    )
    compact_wall = time.monotonic() - compact_started
    compact_signature = expected_input_signature(compact_path)
    if compact_signature["bytes"] != len(mapping) * DIMENSION * 2:
        raise Round0163Error("prompted compact fp16 byte count changed")

    def retained_members(family: Sequence[int]) -> int:
        values = np.asarray(family, dtype=np.int64)
        positions = np.searchsorted(mapping, values)
        present = positions < len(mapping)
        if np.any(present):
            present[present] = mapping[positions[present]] == values[present]
        return int(present.sum())

    audit = {
        source: {
            "families_before_selection": len(families[source]),
            "families_with_more_than_one_retained_member": sum(
                retained_members(family) > 1 for family in families[source]
            ),
        }
        for source in FAMILY_SOURCES
    }
    if any(value["families_with_more_than_one_retained_member"] for value in audit.values()):
        raise Round0163Error("representative selector leaves an exact family")
    observed_r0113_mapping = expected_input_signature(
        str(job["r0113_compact_mapping"]["canonical_path"])
    )
    if observed_r0113_mapping != dict(job["r0113_compact_mapping"]):
        raise Round0163Error("accepted R0113 compact mapping bytes changed")
    accepted_prefix = np.load(
        str(observed_r0113_mapping["canonical_path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    prefix = mapping[mapping < 2_000_000]
    r0113_prefix_exact = bool(
        accepted_prefix.dtype == np.int64
        and accepted_prefix.shape == (1_993_761,)
        and np.array_equal(prefix, accepted_prefix)
    )
    cross_text_collisions = {
        arm: int(relation[arm]["cross_source_text_families"])
        for arm in ("raw_fp16", "document_fp16")
    }
    q2_population_released = bool(
        r0113_prefix_exact
        and all(value == 0 for value in cross_text_collisions.values())
        and all(
            value["families_with_more_than_one_retained_member"] == 0
            for value in audit.values()
        )
    )
    identity = population_identity(
        view_identity=str(view["identity_sha256"]), mapping=mapping, excluded=excluded
    )
    receipt = seal({
        "schema": SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": (
            [CAPABILITY, HOST_CAPABILITY] if q2_population_released else []
        ),
        "population_identity": identity,
        "source_view": job["first8m_view"],
        "source_layout": job["canonical_layout"],
        "source_rows": VIEW_ROWS,
        "retained_rows": len(mapping),
        "excluded_rows": len(excluded),
        "dimension": DIMENSION,
        "dtype": DTYPE,
        "mapping": expected_input_signature(mapping_path),
        "excluded": expected_input_signature(excluded_path),
        "document_compact": compact_signature,
        "source_text_hash_index": expected_input_signature(text_hash_path),
        "mapping_ordered_sha256": ordered_array_sha256(mapping),
        "excluded_ordered_sha256": ordered_array_sha256(excluded),
        "family_census": {
            "source_text": text_report,
            "raw_fp16": raw_report,
            "document_fp16": document_report,
            "embedding_text_relation": relation,
            "union": union_report,
        },
        "retained_family_audit": audit,
        "selection_rule_matches_r0113": True,
        "accepted_r0113_prefix": {
            "mapping": job["r0113_compact_mapping"],
            "expected_rows": 1_993_761,
            "observed_rows": len(prefix),
            "exact": r0113_prefix_exact,
        },
        "cross_source_text_embedding_collisions": cross_text_collisions,
        "q2_population_released": q2_population_released,
        "outcome": (
            "prompted-english-8m-representative-population-qualified"
            if q2_population_released
            else "prompted-english-8m-population-confound-detected"
        ),
        "multiplicity_preserved_as_metadata": True,
        "training_performed": False,
        "graph_built": False,
        "performance": {
            "compact_write_seconds": compact_wall,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
        },
    })
    atomic_write_new_json(os.path.join(output, "representative-census.json"), receipt, immutable=True)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "census_prompted_english_representatives":
        raise Round0163Error("unknown R0163 action")
    run_census(active, job)
