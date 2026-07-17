"""CPU-only preparation utilities for the Round 0010 MiniLM input pack.

This module deliberately depends only on the Python standard library and
NumPy.  In particular it must remain importable without importing Torch or
probing CUDA.  The production contract is the ordered 30M row universe

    fineweb[:10M] | redpajama[:10M] | pile[:10M]

backed by raw, headerless little-endian float32 shards.  All large operations
are bounded, resumable, and content-addressed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


ROUND_ID = "0010"
PACK_SCHEMA = "30m-input-pack-v1"
INVENTORY_SCHEMA = "round0010-source-inventory-v1"
MATERIALIZATION_SCHEMA = "round0010-fp16-materialization-v1"
ENDPOINT_SCHEMA = "round0010-endpoints-v1"
FIXTURE_SCHEMA = "round0010-loader-fixtures-v1"
DIMENSION = 384
ROWS_PER_CORPUS = 10_000_000
TOTAL_ROWS = 30_000_000
GRAPH_K = 15
EDGE_COUNT = TOTAL_ROWS * GRAPH_K
RAW_DTYPE = np.dtype("<f4")
MATERIALIZED_DTYPE = np.dtype("<f2")
ENDPOINT_DTYPE = np.dtype("<i4")
EXPECTED_GRAPH_SHA256 = (
    "2fc30fc27ced442c5b69fde084ab41c054fcc1bf5e7913a5cee9d20f59baadca"
)
DEFAULT_GRAPH = Path("/data/checkpoints/pumap/edges_30m_k15.npz")
DEFAULT_SOURCE_SPECS: tuple[tuple[str, Path], ...] = (
    (
        "fineweb",
        Path(
            "/data/embeddings/"
            "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train"
        ),
    ),
    (
        "redpajama",
        Path(
            "/data/embeddings/"
            "RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train"
        ),
    ),
    (
        "pile",
        Path(
            "/data/embeddings/"
            "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train"
        ),
    ),
)

_SHARD_RE = re.compile(r"^data-(\d+)-of-(\d+)\.npy$")
_HASH_BLOCK_BYTES = 16 * 1024 * 1024
_ROW_BYTES = DIMENSION * RAW_DTYPE.itemsize


class PackError(RuntimeError):
    """A fail-closed input-pack validation error."""


class PlannedInterruption(PackError):
    """Fixture-only interruption used to exercise durable resume."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_record(value: Mapping[str, Any], field: str = "receipt_sha256") -> dict[str, Any]:
    if field in value:
        raise PackError(f"record already contains seal field {field!r}")
    sealed = dict(value)
    sealed[field] = canonical_sha256(value)
    return sealed


def verify_sealed_record(value: Mapping[str, Any], field: str = "receipt_sha256") -> None:
    expected = value.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise PackError(f"missing or malformed {field}")
    body = {key: item for key, item in value.items() if key != field}
    observed = canonical_sha256(body)
    if observed != expected:
        raise PackError(f"{field} mismatch: expected {expected}, observed {observed}")


def read_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PackError(f"expected JSON object at {path}")
    return value


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(
    path: os.PathLike[str] | str,
    value: Mapping[str, Any],
    *,
    replace: bool,
) -> None:
    """Write canonical JSON in the destination directory and fsync it.

    Immutable receipts use ``replace=False``.  Mutable resumable state is the
    only caller allowed to use ``replace=True``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    if destination.exists() and not replace:
        existing = destination.read_bytes()
        if existing == payload:
            return
        raise PackError(f"refusing to overwrite existing artifact {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not replace:
            raise PackError(f"refusing to overwrite existing artifact {destination}")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def file_identity(path: os.PathLike[str] | str) -> dict[str, Any]:
    candidate = Path(path)
    result = os.stat(candidate, follow_symlinks=False)
    if not stat.S_ISREG(result.st_mode):
        raise PackError(f"expected a regular file without symlink traversal: {candidate}")
    return {
        "kind": "regular",
        "mode_octal": format(stat.S_IMODE(result.st_mode), "04o"),
        "device": int(result.st_dev),
        "inode": int(result.st_ino),
        "size_bytes": int(result.st_size),
        "mtime_ns": int(result.st_mtime_ns),
        "ctime_ns": int(result.st_ctime_ns),
    }


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "kind",
        "mode_octal",
        "device",
        "inode",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def _open_regular_readonly(path: Path) -> tuple[int, dict[str, Any]]:
    before = file_identity(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    opened_identity = {
        "kind": "regular" if stat.S_ISREG(opened.st_mode) else "other",
        "mode_octal": format(stat.S_IMODE(opened.st_mode), "04o"),
        "device": int(opened.st_dev),
        "inode": int(opened.st_ino),
        "size_bytes": int(opened.st_size),
        "mtime_ns": int(opened.st_mtime_ns),
        "ctime_ns": int(opened.st_ctime_ns),
    }
    if not _same_identity(before, opened_identity):
        os.close(descriptor)
        raise PackError(f"file changed while opening {path}")
    return descriptor, before


def sha256_file(path: os.PathLike[str] | str) -> tuple[str, int, float]:
    candidate = Path(path)
    descriptor, before = _open_regular_readonly(candidate)
    digest = hashlib.sha256()
    count = 0
    started = time.monotonic()
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            while True:
                block = handle.read(_HASH_BLOCK_BYTES)
                if not block:
                    break
                digest.update(block)
                count += len(block)
    finally:
        # fdopen owns the descriptor in the normal path.  This branch is only
        # needed if fdopen itself raised.
        with contextlib.suppress(OSError):
            os.close(descriptor)
    after = file_identity(candidate)
    if not _same_identity(before, after) or count != before["size_bytes"]:
        raise PackError(f"file changed while hashing {candidate}")
    return digest.hexdigest(), count, time.monotonic() - started


def _parse_shard_name(path: Path) -> tuple[int, int]:
    match = _SHARD_RE.fullmatch(path.name)
    if match is None:
        raise PackError(f"unexpected source shard name {path.name!r}")
    return int(match.group(1)), int(match.group(2))


def discover_raw_shards(directory: os.PathLike[str] | str) -> list[Path]:
    source_dir = Path(directory)
    directory_stat = os.stat(source_dir, follow_symlinks=False)
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise PackError(f"source directory is not a real directory: {source_dir}")
    paths = list(source_dir.glob("*.npy"))
    parsed: list[tuple[int, int, Path]] = []
    for path in paths:
        index, declared_total = _parse_shard_name(path)
        parsed.append((index, declared_total, path))
    parsed.sort(key=lambda item: (item[0], item[2].name))
    if not parsed:
        raise PackError(f"no raw source shards found in {source_dir}")
    indices = [item[0] for item in parsed]
    if len(indices) != len(set(indices)):
        raise PackError(f"duplicate source shard index in {source_dir}")
    # Only the prefix needed by the issued selection is an input to this round.
    # Some later, unused local pull members may be incomplete; validate byte
    # geometry when (and only when) a member enters the selected prefix.
    return [item[2] for item in parsed]


@dataclasses.dataclass(frozen=True)
class SourceSelection:
    corpus: str
    corpus_order: int
    shard_order: int
    shard_index: int
    declared_shard_total: int
    path: Path
    full_rows: int
    selected_local_start: int
    selected_rows: int
    global_start: int

    @property
    def global_stop(self) -> int:
        return self.global_start + self.selected_rows

    @property
    def selected_local_stop(self) -> int:
        return self.selected_local_start + self.selected_rows


def select_source_rows(
    source_specs: Sequence[tuple[str, Path]],
    *,
    rows_per_corpus: int,
    dimension: int,
) -> list[SourceSelection]:
    if dimension != DIMENSION:
        raise PackError(f"Round 0010 dimension is fixed at {DIMENSION}, got {dimension}")
    selections: list[SourceSelection] = []
    global_cursor = 0
    names: set[str] = set()
    for corpus_order, (corpus, directory) in enumerate(source_specs):
        if corpus in names:
            raise PackError(f"duplicate corpus name {corpus!r}")
        names.add(corpus)
        remaining = rows_per_corpus
        for shard_order, path in enumerate(discover_raw_shards(directory)):
            shard_index, declared_total = _parse_shard_name(path)
            size = file_identity(path)["size_bytes"]
            if size <= 0 or size % (dimension * RAW_DTYPE.itemsize):
                raise PackError(
                    f"selected raw shard byte geometry is not an integral "
                    f"{dimension}D float32 matrix: {path}"
                )
            full_rows = size // (dimension * RAW_DTYPE.itemsize)
            take = min(remaining, full_rows)
            if take:
                selections.append(
                    SourceSelection(
                        corpus=corpus,
                        corpus_order=corpus_order,
                        shard_order=shard_order,
                        shard_index=shard_index,
                        declared_shard_total=declared_total,
                        path=path.resolve(),
                        full_rows=full_rows,
                        selected_local_start=0,
                        selected_rows=take,
                        global_start=global_cursor,
                    )
                )
                remaining -= take
                global_cursor += take
            if remaining == 0:
                break
        if remaining:
            raise PackError(
                f"{corpus} has {rows_per_corpus - remaining:,} rows, "
                f"needs {rows_per_corpus:,}"
            )
    expected_total = rows_per_corpus * len(source_specs)
    if global_cursor != expected_total:
        raise PackError(f"source selection has {global_cursor:,} rows, expected {expected_total:,}")
    return selections


def _scan_raw_source(selection: SourceSelection) -> dict[str, Any]:
    """Hash a complete source shard while validating every selected value."""

    descriptor, before = _open_regular_readonly(selection.path)
    digest = hashlib.sha256()
    selected_bytes = selection.selected_rows * DIMENSION * RAW_DTYPE.itemsize
    selected_start_bytes = selection.selected_local_start * DIMENSION * RAW_DTYPE.itemsize
    selected_stop_bytes = selected_start_bytes + selected_bytes
    bytes_read = 0
    finite_values = 0
    value_min = math.inf
    value_max = -math.inf
    started = time.monotonic()
    first_prefix = b""
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            while True:
                block = handle.read(_HASH_BLOCK_BYTES)
                if not block:
                    break
                if not first_prefix:
                    first_prefix = block[:8]
                digest.update(block)
                block_start = bytes_read
                block_stop = block_start + len(block)
                overlap_start = max(block_start, selected_start_bytes)
                overlap_stop = min(block_stop, selected_stop_bytes)
                if overlap_start < overlap_stop:
                    lo = overlap_start - block_start
                    hi = overlap_stop - block_start
                    selected = block[lo:hi]
                    if len(selected) % RAW_DTYPE.itemsize:
                        raise PackError(f"unaligned float32 bytes in {selection.path}")
                    values = np.frombuffer(selected, dtype=RAW_DTYPE)
                    finite = np.isfinite(values)
                    if not bool(np.all(finite)):
                        bad = int(values.size - np.count_nonzero(finite))
                        raise PackError(f"{selection.path} contains {bad} non-finite selected values")
                    finite_values += int(values.size)
                    if values.size:
                        value_min = min(value_min, float(np.min(values)))
                        value_max = max(value_max, float(np.max(values)))
                bytes_read = block_stop
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)
    after = file_identity(selection.path)
    if not _same_identity(before, after) or bytes_read != before["size_bytes"]:
        raise PackError(f"source changed while inventorying {selection.path}")
    if first_prefix.startswith(b"\x93NUMPY"):
        raise PackError(f"source unexpectedly has a NumPy header: {selection.path}")
    expected_values = selection.selected_rows * DIMENSION
    if finite_values != expected_values:
        raise PackError(
            f"selected finite value count mismatch for {selection.path}: "
            f"{finite_values} != {expected_values}"
        )
    return {
        "path": str(selection.path),
        "corpus": selection.corpus,
        "corpus_order": selection.corpus_order,
        "shard_order_within_corpus": selection.shard_order,
        "shard_index": selection.shard_index,
        "declared_shard_total": selection.declared_shard_total,
        "representation": "raw-headerless-little-endian-float32",
        "dimension": DIMENSION,
        "full_rows": selection.full_rows,
        "selected_local_row_start": selection.selected_local_start,
        "selected_local_row_stop": selection.selected_local_stop,
        "selected_rows": selection.selected_rows,
        "selected_byte_start": selected_start_bytes,
        "selected_byte_stop": selected_stop_bytes,
        "output_global_row_start": selection.global_start,
        "output_global_row_stop": selection.global_stop,
        "selected_values_finite": True,
        "selected_value_min": value_min,
        "selected_value_max": value_max,
        "sha256_full_file": digest.hexdigest(),
        "identity": before,
        "hash_and_finite_scan_wall_seconds": time.monotonic() - started,
    }


def _read_npy_header(handle: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise PackError(f"unsupported NPY header version {version}")
    return tuple(int(value) for value in shape), bool(fortran), np.dtype(dtype)


def inspect_graph_npz(
    path: os.PathLike[str] | str,
    *,
    expected_sha256: str = EXPECTED_GRAPH_SHA256,
) -> dict[str, Any]:
    graph = Path(path).resolve()
    graph_sha, size_bytes, hash_wall = sha256_file(graph)
    if graph_sha != expected_sha256:
        raise PackError(
            f"graph SHA-256 mismatch: expected {expected_sha256}, observed {graph_sha}"
        )
    expected_members = {
        "sources.npy",
        "targets.npy",
        "weights.npy",
        "n_nodes.npy",
        "k.npy",
        "nprobe.npy",
    }
    headers: dict[str, Any] = {}
    scalars: dict[str, int] = {}
    with zipfile.ZipFile(graph, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != expected_members:
            raise PackError(f"unexpected or duplicate graph members: {names}")
        for name in names:
            info = archive.getinfo(name)
            with archive.open(info, "r") as member:
                shape, fortran, dtype = _read_npy_header(member)
                headers[name] = {
                    "shape": list(shape),
                    "dtype": dtype.str,
                    "fortran_order": fortran,
                    "zip_crc32": format(info.CRC, "08x"),
                    "uncompressed_size_bytes": int(info.file_size),
                    "compressed_size_bytes": int(info.compress_size),
                }
                if shape == ():
                    data = member.read()
                    if len(data) != dtype.itemsize:
                        raise PackError(f"malformed scalar member {name}")
                    scalars[name.removesuffix(".npy")] = int(
                        np.frombuffer(data, dtype=dtype, count=1)[0]
                    )
    exact_headers = {
        "sources.npy": ([EDGE_COUNT], ENDPOINT_DTYPE.str),
        "targets.npy": ([EDGE_COUNT], ENDPOINT_DTYPE.str),
        "weights.npy": ([EDGE_COUNT], np.dtype("<f4").str),
    }
    for name, (shape, dtype) in exact_headers.items():
        observed = headers[name]
        if observed["shape"] != shape or observed["dtype"] != dtype or observed["fortran_order"]:
            raise PackError(f"graph member contract mismatch for {name}: {observed}")
    if scalars.get("n_nodes") != TOTAL_ROWS or scalars.get("k") != GRAPH_K:
        raise PackError(f"graph scalar contract mismatch: {scalars}")
    return {
        "path": str(graph),
        "identity": file_identity(graph),
        "sha256_full_file": graph_sha,
        "size_bytes": size_bytes,
        "hash_wall_seconds": hash_wall,
        "members": headers,
        "scalars": scalars,
    }


def _disk_forecast(root: Path, *, rows_per_output_shard: int) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    materialized = TOTAL_ROWS * DIMENSION * MATERIALIZED_DTYPE.itemsize
    endpoints = EDGE_COUNT * ENDPOINT_DTYPE.itemsize * 2
    max_material_temp = rows_per_output_shard * DIMENSION * MATERIALIZED_DTYPE.itemsize + 4096
    max_endpoint_temp = EDGE_COUNT * ENDPOINT_DTYPE.itemsize + 4096
    persistent = materialized + endpoints
    peak_incremental = persistent + max(max_material_temp, max_endpoint_temp)
    reserve = 4 * 1024**3
    required = peak_incremental + reserve
    if usage.free < required:
        raise PackError(
            f"insufficient free disk: {usage.free} bytes available, {required} required"
        )
    return {
        "filesystem_total_bytes": usage.total,
        "filesystem_used_bytes": usage.used,
        "filesystem_free_bytes_at_admission": usage.free,
        "materialized_payload_bytes": materialized,
        "endpoint_payload_bytes": endpoints,
        "largest_atomic_temporary_bytes": max(max_material_temp, max_endpoint_temp),
        "safety_reserve_bytes": reserve,
        "required_free_bytes": required,
        "admitted": True,
    }


def build_source_inventory(
    root: os.PathLike[str] | str,
    *,
    source_specs: Sequence[tuple[str, Path]] = DEFAULT_SOURCE_SPECS,
    graph_path: os.PathLike[str] | str = DEFAULT_GRAPH,
    rows_per_output_shard: int = 1_000_000,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if root_path != Path("/data/latent-basemap/runs/round-0010"):
        raise PackError(f"production inventory root must be the Round 0010 root, got {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)
    if rows_per_output_shard <= 0 or TOTAL_ROWS % rows_per_output_shard:
        raise PackError("rows_per_output_shard must be a positive divisor of 30,000,000")
    started_utc = utc_now()
    started = time.monotonic()
    selections = select_source_rows(
        source_specs, rows_per_corpus=ROWS_PER_CORPUS, dimension=DIMENSION
    )
    print(
        f"inventory: selected {len(selections)} source shards for {TOTAL_ROWS:,} rows",
        flush=True,
    )
    source_records: list[dict[str, Any]] = []
    for index, selection in enumerate(selections, start=1):
        print(
            f"inventory: [{index}/{len(selections)}] {selection.corpus} "
            f"{selection.path.name} rows={selection.selected_rows:,}",
            flush=True,
        )
        source_records.append(_scan_raw_source(selection))
    graph = inspect_graph_npz(graph_path)
    forecast = _disk_forecast(root_path, rows_per_output_shard=rows_per_output_shard)
    body = {
        "schema": INVENTORY_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": started_utc,
        "completed_utc": utc_now(),
        "cuda_visible_devices_required": "",
        "torch_imported": "torch" in __import__("sys").modules,
        "row_contract": {
            "order": [name for name, _ in source_specs],
            "rows_per_corpus": ROWS_PER_CORPUS,
            "total_rows": TOTAL_ROWS,
            "dimension": DIMENSION,
            "source_dtype": RAW_DTYPE.str,
            "source_representation": "raw-headerless",
            "materialized_dtype": MATERIALIZED_DTYPE.str,
            "rows_per_output_shard": rows_per_output_shard,
        },
        "sources": source_records,
        "graph": graph,
        "disk_forecast": forecast,
        "inventory_wall_seconds": time.monotonic() - started,
    }
    if body["torch_imported"]:
        raise PackError("Torch was imported in the CPU-only Round 0010 process")
    sealed = seal_record(body)
    destination = root_path / "receipts" / "source-inventory.json"
    atomic_write_json(destination, sealed, replace=False)
    print(f"inventory: sealed {destination} {sealed['receipt_sha256']}", flush=True)
    return sealed


def load_inventory(path: os.PathLike[str] | str) -> dict[str, Any]:
    inventory = read_json(path)
    verify_sealed_record(inventory)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise PackError(f"unexpected inventory schema {inventory.get('schema')!r}")
    validate_inventory_structure(inventory)
    return inventory


def validate_inventory_structure(inventory: Mapping[str, Any]) -> None:
    contract = inventory.get("row_contract")
    if not isinstance(contract, dict):
        raise PackError("inventory row_contract is missing")
    if contract.get("total_rows") != TOTAL_ROWS or contract.get("dimension") != DIMENSION:
        raise PackError("inventory row contract changed")
    if contract.get("order") != [name for name, _ in DEFAULT_SOURCE_SPECS]:
        raise PackError(f"inventory corpus order changed: {contract.get('order')}")
    records = inventory.get("sources")
    if not isinstance(records, list) or not records:
        raise PackError("inventory source list is empty")
    seen_paths: set[str] = set()
    cursor = 0
    corpus_rows = {name: 0 for name, _ in DEFAULT_SOURCE_SPECS}
    previous_key: tuple[int, int] | None = None
    for record in records:
        if not isinstance(record, dict):
            raise PackError("inventory source member is not an object")
        path = record.get("path")
        if not isinstance(path, str) or path in seen_paths:
            raise PackError(f"missing or duplicate inventory path: {path!r}")
        seen_paths.add(path)
        key = (int(record["corpus_order"]), int(record["shard_order_within_corpus"]))
        if previous_key is not None and key <= previous_key:
            raise PackError("inventory source records are reordered or duplicated")
        previous_key = key
        if int(record["output_global_row_start"]) != cursor:
            raise PackError("inventory global row ranges are not contiguous")
        stop = int(record["output_global_row_stop"])
        selected_rows = int(record["selected_rows"])
        if stop - cursor != selected_rows or selected_rows <= 0:
            raise PackError("inventory source row range is malformed")
        if record.get("representation") != "raw-headerless-little-endian-float32":
            raise PackError("inventory source representation changed")
        if record.get("dimension") != DIMENSION:
            raise PackError("inventory source dimension changed")
        digest = record.get("sha256_full_file")
        if not isinstance(digest, str) or len(digest) != 64:
            raise PackError("inventory source hash is malformed")
        corpus = str(record.get("corpus"))
        if corpus not in corpus_rows:
            raise PackError(f"unknown inventory corpus {corpus!r}")
        corpus_rows[corpus] += selected_rows
        cursor = stop
    if cursor != TOTAL_ROWS or any(value != ROWS_PER_CORPUS for value in corpus_rows.values()):
        raise PackError(f"inventory row totals changed: total={cursor}, corpora={corpus_rows}")
    graph = inventory.get("graph")
    if not isinstance(graph, dict) or graph.get("sha256_full_file") != EXPECTED_GRAPH_SHA256:
        raise PackError("inventory graph identity changed")


def verify_inventory_sources(
    inventory: Mapping[str, Any], *, full_hash: bool, check_identity: bool = True
) -> dict[str, Any]:
    validate_inventory_structure(inventory)
    verified: list[dict[str, Any]] = []
    started = time.monotonic()
    for record in inventory["sources"]:
        path = Path(record["path"])
        observed_identity = file_identity(path)
        if check_identity and not _same_identity(record["identity"], observed_identity):
            raise PackError(f"source identity is stale: {path}")
        item = {"path": str(path), "identity_match": True}
        if full_hash:
            digest, size, wall = sha256_file(path)
            if digest != record["sha256_full_file"]:
                raise PackError(f"source hash mismatch: {path}")
            item.update({"sha256": digest, "size_bytes": size, "hash_wall_seconds": wall})
        verified.append(item)
    return {
        "verified": verified,
        "full_hash": full_hash,
        "wall_seconds": time.monotonic() - started,
    }


@dataclasses.dataclass(frozen=True)
class RawMapMember:
    path: Path
    corpus: str
    global_start: int
    global_stop: int
    local_start: int
    full_rows: int
    identity: Mapping[str, Any]
    sha256: str


class RawSourceMap:
    """Bounded, deterministic view over an ordered raw-float32 row universe."""

    def __init__(
        self,
        members: Sequence[RawMapMember],
        *,
        total_rows: int,
        dimension: int,
        enforce_identity: bool = True,
    ) -> None:
        if not members:
            raise PackError("raw source map cannot be empty")
        self.members = tuple(members)
        self.total_rows = int(total_rows)
        self.dimension = int(dimension)
        self.enforce_identity = bool(enforce_identity)
        cursor = 0
        for member in self.members:
            if member.global_start != cursor or member.global_stop <= member.global_start:
                raise PackError("raw source map ranges must be positive and contiguous")
            cursor = member.global_stop
        if cursor != self.total_rows:
            raise PackError(f"raw source map ends at {cursor}, expected {self.total_rows}")

    @classmethod
    def from_inventory(cls, inventory: Mapping[str, Any]) -> "RawSourceMap":
        validate_inventory_structure(inventory)
        members = [
            RawMapMember(
                path=Path(record["path"]),
                corpus=str(record["corpus"]),
                global_start=int(record["output_global_row_start"]),
                global_stop=int(record["output_global_row_stop"]),
                local_start=int(record["selected_local_row_start"]),
                full_rows=int(record["full_rows"]),
                identity=record["identity"],
                sha256=str(record["sha256_full_file"]),
            )
            for record in inventory["sources"]
        ]
        return cls(members, total_rows=TOTAL_ROWS, dimension=DIMENSION)

    def _check_bounds(self, start: int, stop: int) -> None:
        if start < 0 or stop < start or stop > self.total_rows:
            raise IndexError(f"row range [{start},{stop}) outside [0,{self.total_rows})")

    def intersections(self, start: int, stop: int) -> Iterator[tuple[RawMapMember, int, int]]:
        self._check_bounds(start, stop)
        for member in self.members:
            if member.global_stop <= start:
                continue
            if member.global_start >= stop:
                break
            yield member, max(start, member.global_start), min(stop, member.global_stop)

    def source_segments(self, start: int, stop: int) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for member, global_lo, global_hi in self.intersections(start, stop):
            local_lo = member.local_start + global_lo - member.global_start
            local_hi = local_lo + global_hi - global_lo
            segments.append(
                {
                    "corpus": member.corpus,
                    "source_path": str(member.path),
                    "source_sha256": member.sha256,
                    "source_local_row_start": local_lo,
                    "source_local_row_stop": local_hi,
                    "output_global_row_start": global_lo,
                    "output_global_row_stop": global_hi,
                }
            )
        return segments

    def read(self, start: int, stop: int) -> np.ndarray:
        self._check_bounds(start, stop)
        output = np.empty((stop - start, self.dimension), dtype=RAW_DTYPE)
        self.read_into(start, stop, output)
        return output

    def read_into(self, start: int, stop: int, output: np.ndarray) -> None:
        self._check_bounds(start, stop)
        expected_shape = (stop - start, self.dimension)
        if output.shape != expected_shape:
            raise PackError(f"output shape {output.shape} does not match {expected_shape}")
        destination_cursor = 0
        for member, global_lo, global_hi in self.intersections(start, stop):
            if self.enforce_identity:
                observed = file_identity(member.path)
                if not _same_identity(member.identity, observed):
                    raise PackError(f"source identity changed before read: {member.path}")
            local_lo = member.local_start + global_lo - member.global_start
            count = global_hi - global_lo
            source = np.memmap(
                member.path,
                mode="r",
                dtype=RAW_DTYPE,
                offset=local_lo * self.dimension * RAW_DTYPE.itemsize,
                shape=(count, self.dimension),
                order="C",
            )
            output[destination_cursor : destination_cursor + count] = source
            destination_cursor += count
            del source
        if destination_cursor != stop - start:
            raise PackError("raw source map did not fill the requested slice")

    def take(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1:
            raise IndexError("only one-dimensional index arrays are supported")
        if np.any((requested < 0) | (requested >= self.total_rows)):
            raise IndexError("raw source map index out of bounds")
        result = np.empty((len(requested), self.dimension), dtype=RAW_DTYPE)
        for member in self.members:
            positions = np.flatnonzero(
                (requested >= member.global_start) & (requested < member.global_stop)
            )
            if positions.size == 0:
                continue
            if self.enforce_identity:
                observed = file_identity(member.path)
                if not _same_identity(member.identity, observed):
                    raise PackError(f"source identity changed before indexed read: {member.path}")
            local = member.local_start + requested[positions] - member.global_start
            source = np.memmap(
                member.path,
                mode="r",
                dtype=RAW_DTYPE,
                shape=(member.full_rows, self.dimension),
                order="C",
            )
            result[positions] = source[local]
            del source
        return result


@dataclasses.dataclass
class GeometryAccumulator:
    row_count: int = 0
    value_count: int = 0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    max_absolute_error: float = 0.0
    cosine_sum: float = 0.0
    min_cosine: float = 1.0
    norm_absolute_error_sum: float = 0.0
    max_norm_absolute_error: float = 0.0

    def update(self, original: np.ndarray, quantized: np.ndarray) -> None:
        if original.shape != quantized.shape:
            raise PackError("geometry arrays have different shapes")
        reconstructed = quantized.astype(np.float32)
        difference = reconstructed - original
        absolute = np.abs(difference)
        self.row_count += int(original.shape[0])
        self.value_count += int(original.size)
        self.absolute_error_sum += float(np.sum(absolute, dtype=np.float64))
        self.squared_error_sum += float(
            np.sum(np.multiply(difference, difference), dtype=np.float64)
        )
        if absolute.size:
            self.max_absolute_error = max(self.max_absolute_error, float(np.max(absolute)))
        original_norm = np.sqrt(np.einsum("ij,ij->i", original, original))
        reconstructed_norm = np.sqrt(
            np.einsum("ij,ij->i", reconstructed, reconstructed)
        )
        if np.any(original_norm <= 0) or np.any(reconstructed_norm <= 0):
            raise PackError("zero-norm row encountered during fp16 geometry validation")
        dot = np.einsum("ij,ij->i", original, reconstructed)
        cosine = dot / (original_norm * reconstructed_norm)
        if not bool(np.all(np.isfinite(cosine))):
            raise PackError("non-finite cosine in fp16 geometry validation")
        cosine = np.clip(cosine, -1.0, 1.0)
        self.cosine_sum += float(np.sum(cosine, dtype=np.float64))
        self.min_cosine = min(self.min_cosine, float(np.min(cosine)))
        norm_error = np.abs(reconstructed_norm - original_norm)
        self.norm_absolute_error_sum += float(np.sum(norm_error, dtype=np.float64))
        self.max_norm_absolute_error = max(
            self.max_norm_absolute_error, float(np.max(norm_error))
        )

    def merge_summary(self, summary: Mapping[str, Any]) -> None:
        self.row_count += int(summary["row_count"])
        self.value_count += int(summary["value_count"])
        self.absolute_error_sum += float(summary["absolute_error_sum"])
        self.squared_error_sum += float(summary["squared_error_sum"])
        self.max_absolute_error = max(
            self.max_absolute_error, float(summary["max_absolute_error"])
        )
        self.cosine_sum += float(summary["cosine_sum"])
        self.min_cosine = min(self.min_cosine, float(summary["min_cosine"]))
        self.norm_absolute_error_sum += float(summary["norm_absolute_error_sum"])
        self.max_norm_absolute_error = max(
            self.max_norm_absolute_error,
            float(summary["max_norm_absolute_error"]),
        )

    def summary(self) -> dict[str, Any]:
        if self.row_count <= 0 or self.value_count <= 0:
            raise PackError("empty fp16 geometry summary")
        return {
            "scope": "all-selected-values-and-rows",
            "row_count": self.row_count,
            "value_count": self.value_count,
            "absolute_error_sum": self.absolute_error_sum,
            "squared_error_sum": self.squared_error_sum,
            "mean_absolute_error": self.absolute_error_sum / self.value_count,
            "root_mean_squared_error": math.sqrt(self.squared_error_sum / self.value_count),
            "max_absolute_error": self.max_absolute_error,
            "cosine_sum": self.cosine_sum,
            "mean_cosine": self.cosine_sum / self.row_count,
            "min_cosine": self.min_cosine,
            "norm_absolute_error_sum": self.norm_absolute_error_sum,
            "mean_norm_absolute_error": self.norm_absolute_error_sum / self.row_count,
            "max_norm_absolute_error": self.max_norm_absolute_error,
        }


def _scan_npy(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
    require_finite: bool,
    block_rows: int = 131_072,
) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != expected_shape or array.dtype != expected_dtype:
        raise PackError(
            f"array contract mismatch for {path}: shape={array.shape}, dtype={array.dtype}, "
            f"expected shape={expected_shape}, dtype={expected_dtype}"
        )
    if not array.flags.c_contiguous:
        raise PackError(f"array is not C contiguous: {path}")
    value_min = math.inf
    value_max = -math.inf
    if array.ndim == 1:
        outer = array.shape[0]
    else:
        outer = array.shape[0]
    for start in range(0, outer, block_rows):
        block = np.asarray(array[start : min(start + block_rows, outer)])
        if require_finite and not bool(np.all(np.isfinite(block))):
            raise PackError(f"non-finite value in {path}")
        if block.size:
            value_min = min(value_min, float(np.min(block)))
            value_max = max(value_max, float(np.max(block)))
    del array
    digest, size, hash_wall = sha256_file(path)
    return {
        "path": str(path.resolve()),
        "shape": list(expected_shape),
        "dtype": expected_dtype.str,
        "c_contiguous": True,
        "finite": require_finite,
        "value_min": value_min,
        "value_max": value_max,
        "sha256": digest,
        "size_bytes": size,
        "hash_wall_seconds": hash_wall,
        "identity": file_identity(path),
    }


def _materialization_plan(
    inventory: Mapping[str, Any], *, rows_per_output_shard: int, block_rows: int
) -> dict[str, Any]:
    return {
        "schema": MATERIALIZATION_SCHEMA,
        "source_inventory_receipt_sha256": inventory["receipt_sha256"],
        "total_rows": TOTAL_ROWS,
        "dimension": DIMENSION,
        "source_dtype": RAW_DTYPE.str,
        "output_dtype": MATERIALIZED_DTYPE.str,
        "rows_per_output_shard": rows_per_output_shard,
        "output_shard_count": math.ceil(TOTAL_ROWS / rows_per_output_shard),
        "bounded_working_rows": block_rows,
        "atomic_unit": "one content-addressed chunk directory",
    }


def _sealed_state(body: Mapping[str, Any]) -> dict[str, Any]:
    return seal_record(body, field="state_sha256")


def _verify_state(state: Mapping[str, Any]) -> None:
    verify_sealed_record(state, field="state_sha256")


def _clean_owned_partial_directories(parent: Path) -> list[str]:
    removed: list[str] = []
    if not parent.exists():
        return removed
    for candidate in sorted(parent.iterdir()):
        if not candidate.name.startswith(".chunk-") or ".partial-" not in candidate.name:
            continue
        if not candidate.is_dir() or candidate.is_symlink():
            raise PackError(f"unexpected partial artifact kind: {candidate}")
        shutil.rmtree(candidate)
        removed.append(str(candidate))
    if removed:
        _fsync_directory(parent)
    return removed


def _load_chunk_receipt(chunk_dir: Path) -> dict[str, Any]:
    receipt = read_json(chunk_dir / "receipt.json")
    verify_sealed_record(receipt)
    return receipt


def _validate_materialized_chunk(
    chunk_dir: Path,
    *,
    index: int,
    start: int,
    stop: int,
    plan_sha256: str,
) -> dict[str, Any]:
    receipt = _load_chunk_receipt(chunk_dir)
    if (
        receipt.get("schema") != "round0010-fp16-chunk-v1"
        or receipt.get("chunk_index") != index
        or receipt.get("global_row_start") != start
        or receipt.get("global_row_stop") != stop
        or receipt.get("materialization_plan_sha256") != plan_sha256
    ):
        raise PackError(f"materialized chunk receipt mismatch: {chunk_dir}")
    data_path = chunk_dir / "embeddings.npy"
    observed = _scan_npy(
        data_path,
        expected_shape=(stop - start, DIMENSION),
        expected_dtype=MATERIALIZED_DTYPE,
        require_finite=True,
    )
    expected = receipt.get("artifact")
    if not isinstance(expected, dict):
        raise PackError(f"missing artifact record in {chunk_dir}")
    if observed["sha256"] != expected.get("sha256") or observed["size_bytes"] != expected.get(
        "size_bytes"
    ):
        raise PackError(f"materialized chunk hash mismatch: {chunk_dir}")
    return receipt


def materialize_fp16(
    root: os.PathLike[str] | str,
    inventory_path: os.PathLike[str] | str,
    *,
    rows_per_output_shard: int = 1_000_000,
    block_rows: int = 32_768,
    interrupt_after_chunks: int | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if root_path != Path("/data/latent-basemap/runs/round-0010"):
        raise PackError(f"production materialization root must be Round 0010, got {root_path}")
    inventory = load_inventory(inventory_path)
    if rows_per_output_shard <= 0 or TOTAL_ROWS % rows_per_output_shard:
        raise PackError("rows_per_output_shard must divide 30,000,000")
    if block_rows <= 0 or block_rows > rows_per_output_shard:
        raise PackError("invalid bounded block_rows")
    verify_inventory_sources(inventory, full_hash=False)
    source = RawSourceMap.from_inventory(inventory)
    plan = _materialization_plan(
        inventory, rows_per_output_shard=rows_per_output_shard, block_rows=block_rows
    )
    plan_sha = canonical_sha256(plan)
    materialized_root = root_path / "materialized"
    materialized_root.mkdir(parents=True, exist_ok=True)
    removed_partials = _clean_owned_partial_directories(materialized_root)
    state_path = root_path / "receipts" / "materialization-state.json"
    if state_path.exists():
        state = read_json(state_path)
        _verify_state(state)
        if state.get("plan_sha256") != plan_sha or state.get("plan") != plan:
            raise PackError("existing materialization state belongs to another plan")
    else:
        state = _sealed_state(
            {
                "schema": "round0010-materialization-state-v1",
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
                "plan": plan,
                "plan_sha256": plan_sha,
                "completed_chunks": [],
                "recovered_partial_directories": removed_partials,
            }
        )
        atomic_write_json(state_path, state, replace=False)
    started_utc = utc_now()
    started = time.monotonic()
    completed_receipts: list[dict[str, Any]] = []
    created_this_call = 0
    shard_count = int(plan["output_shard_count"])
    for index in range(shard_count):
        start = index * rows_per_output_shard
        stop = min(start + rows_per_output_shard, TOTAL_ROWS)
        chunk_dir = materialized_root / f"chunk-{index:05d}"
        if chunk_dir.exists():
            if not chunk_dir.is_dir() or chunk_dir.is_symlink():
                raise PackError(f"materialized chunk path has wrong kind: {chunk_dir}")
            receipt = _validate_materialized_chunk(
                chunk_dir, index=index, start=start, stop=stop, plan_sha256=plan_sha
            )
            print(f"materialize: resume verified chunk {index + 1}/{shard_count}", flush=True)
        else:
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".chunk-{index:05d}.partial-", dir=materialized_root
                )
            )
            try:
                output_path = temporary / "embeddings.npy"
                output = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=MATERIALIZED_DTYPE,
                    shape=(stop - start, DIMENSION),
                )
                geometry = GeometryAccumulator()
                chunk_started = time.monotonic()
                for block_start in range(start, stop, block_rows):
                    block_stop = min(block_start + block_rows, stop)
                    original = source.read(block_start, block_stop)
                    if not bool(np.all(np.isfinite(original))):
                        raise PackError(f"source became non-finite at rows {block_start}:{block_stop}")
                    quantized = original.astype(MATERIALIZED_DTYPE)
                    if not bool(np.all(np.isfinite(quantized))):
                        raise PackError(f"fp16 overflow at rows {block_start}:{block_stop}")
                    output[block_start - start : block_stop - start] = quantized
                    geometry.update(original, quantized)
                output.flush()
                del output
                with output_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                artifact = _scan_npy(
                    output_path,
                    expected_shape=(stop - start, DIMENSION),
                    expected_dtype=MATERIALIZED_DTYPE,
                    require_finite=True,
                )
                artifact["path"] = str((chunk_dir / "embeddings.npy").resolve())
                chunk_body = {
                    "schema": "round0010-fp16-chunk-v1",
                    "created_utc": utc_now(),
                    "chunk_index": index,
                    "global_row_start": start,
                    "global_row_stop": stop,
                    "row_count": stop - start,
                    "dimension": DIMENSION,
                    "dtype": MATERIALIZED_DTYPE.str,
                    "materialization_plan_sha256": plan_sha,
                    "source_segments": source.source_segments(start, stop),
                    "geometry": geometry.summary(),
                    "artifact": artifact,
                    "materialization_wall_seconds": time.monotonic() - chunk_started,
                    "atomic_completion": "directory rename after data fsync, reopen, hash, and receipt fsync",
                }
                receipt = seal_record(chunk_body)
                atomic_write_json(temporary / "receipt.json", receipt, replace=False)
                if chunk_dir.exists():
                    raise PackError(f"chunk collision before atomic commit: {chunk_dir}")
                os.rename(temporary, chunk_dir)
                _fsync_directory(materialized_root)
                receipt = _validate_materialized_chunk(
                    chunk_dir, index=index, start=start, stop=stop, plan_sha256=plan_sha
                )
                created_this_call += 1
                print(
                    f"materialize: committed chunk {index + 1}/{shard_count} "
                    f"rows={stop - start:,} sha256={receipt['artifact']['sha256']}",
                    flush=True,
                )
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        completed_receipts.append(receipt)
        state_body = {
            key: value
            for key, value in state.items()
            if key not in {"state_sha256", "updated_utc", "completed_chunks"}
        }
        state_body["updated_utc"] = utc_now()
        state_body["completed_chunks"] = [
            {
                "chunk_index": int(item["chunk_index"]),
                "receipt_sha256": str(item["receipt_sha256"]),
                "artifact_sha256": str(item["artifact"]["sha256"]),
            }
            for item in completed_receipts
        ]
        state = _sealed_state(state_body)
        atomic_write_json(state_path, state, replace=True)
        if interrupt_after_chunks is not None and created_this_call >= interrupt_after_chunks:
            raise PlannedInterruption(
                f"planned interruption after {created_this_call} newly committed chunks"
            )
    aggregate = GeometryAccumulator()
    ordered_shards: list[dict[str, Any]] = []
    total_bytes = 0
    for receipt in completed_receipts:
        aggregate.merge_summary(receipt["geometry"])
        artifact = receipt["artifact"]
        total_bytes += int(artifact["size_bytes"])
        ordered_shards.append(
            {
                "chunk_index": receipt["chunk_index"],
                "path": str(
                    materialized_root
                    / f"chunk-{int(receipt['chunk_index']):05d}"
                    / "embeddings.npy"
                ),
                "global_row_start": receipt["global_row_start"],
                "global_row_stop": receipt["global_row_stop"],
                "shape": artifact["shape"],
                "dtype": artifact["dtype"],
                "sha256": artifact["sha256"],
                "size_bytes": artifact["size_bytes"],
                "receipt_sha256": receipt["receipt_sha256"],
                "source_segments": receipt["source_segments"],
            }
        )
    if sum(int(item["shape"][0]) for item in ordered_shards) != TOTAL_ROWS:
        raise PackError("materialized shard row total is not exactly 30,000,000")
    body = {
        "schema": MATERIALIZATION_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": started_utc,
        "completed_utc": utc_now(),
        "source_inventory_path": str(Path(inventory_path).resolve()),
        "source_inventory_receipt_sha256": inventory["receipt_sha256"],
        "materialization_plan": plan,
        "materialization_plan_sha256": plan_sha,
        "ordered_shards": ordered_shards,
        "total_rows": TOTAL_ROWS,
        "dimension": DIMENSION,
        "dtype": MATERIALIZED_DTYPE.str,
        "total_storage_bytes": total_bytes,
        "geometry": aggregate.summary(),
        "reopen_verified_all_shards": True,
        "created_chunks_this_invocation": created_this_call,
        "phase_wall_seconds": time.monotonic() - started,
    }
    sealed = seal_record(body)
    destination = root_path / "receipts" / "materialization-reopen.json"
    atomic_write_json(destination, sealed, replace=False)
    print(f"materialize: sealed {destination} {sealed['receipt_sha256']}", flush=True)
    return sealed


def _copy_zip_member_to_file(
    archive: zipfile.ZipFile, member_name: str, destination: Path
) -> dict[str, Any]:
    info = archive.getinfo(member_name)
    copied = 0
    started = time.monotonic()
    with archive.open(info, "r") as source, destination.open("xb") as output:
        while True:
            block = source.read(_HASH_BLOCK_BYTES)
            if not block:
                break
            output.write(block)
            copied += len(block)
        output.flush()
        os.fsync(output.fileno())
    if copied != info.file_size:
        raise PackError(
            f"truncated graph member {member_name}: copied {copied}, expected {info.file_size}"
        )
    return {
        "zip_member": member_name,
        "zip_crc32": format(info.CRC, "08x"),
        "zip_uncompressed_size_bytes": int(info.file_size),
        "zip_compressed_size_bytes": int(info.compress_size),
        "copy_wall_seconds": time.monotonic() - started,
    }


def _validate_endpoint_array(
    path: Path,
    *,
    role: str,
    block_values: int = 4_000_000,
    edge_count: int = EDGE_COUNT,
    n_nodes: int = TOTAL_ROWS,
    k: int = GRAPH_K,
) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != (edge_count,) or array.dtype != ENDPOINT_DTYPE:
        raise PackError(f"{role} endpoint contract mismatch: {array.shape} {array.dtype}")
    minimum = n_nodes
    maximum = -1
    source_order_exact = False
    started = time.monotonic()
    for start in range(0, edge_count, block_values):
        stop = min(start + block_values, edge_count)
        block = np.asarray(array[start:stop])
        minimum = min(minimum, int(np.min(block)))
        maximum = max(maximum, int(np.max(block)))
        if role == "sources":
            expected = (np.arange(start, stop, dtype=np.int64) // k).astype(
                ENDPOINT_DTYPE
            )
            if not bool(np.array_equal(block, expected)):
                mismatch = int(np.flatnonzero(block != expected)[0])
                raise PackError(
                    f"source endpoint row order mismatch at edge {start + mismatch}: "
                    f"{int(block[mismatch])} != {int(expected[mismatch])}"
                )
            source_order_exact = True
    del array
    if minimum < 0 or maximum >= n_nodes:
        raise PackError(f"{role} endpoints out of bounds: min={minimum}, max={maximum}")
    artifact = _scan_npy(
        path,
        expected_shape=(edge_count,),
        expected_dtype=ENDPOINT_DTYPE,
        require_finite=False,
        block_rows=block_values,
    )
    artifact.update(
        {
            "endpoint_role": role,
            "minimum": minimum,
            "maximum": maximum,
            "all_in_bounds": True,
            "validation_wall_seconds": time.monotonic() - started,
        }
    )
    if role == "sources":
        artifact["source_repeat_k_row_order_exact"] = source_order_exact
    return artifact


def _extract_endpoint_atomic(
    graph_path: Path,
    endpoint_root: Path,
    *,
    role: str,
) -> dict[str, Any]:
    member_name = f"{role}.npy"
    final_dir = endpoint_root / role
    final_path = final_dir / member_name
    if final_dir.exists():
        if not final_dir.is_dir() or final_dir.is_symlink():
            raise PackError(f"endpoint output has wrong kind: {final_dir}")
        receipt = _load_chunk_receipt(final_dir)
        if receipt.get("schema") != "round0010-endpoint-member-v1" or receipt.get(
            "endpoint_role"
        ) != role:
            raise PackError(f"endpoint receipt mismatch: {final_dir}")
        observed = _validate_endpoint_array(final_path, role=role)
        if observed["sha256"] != receipt["artifact"]["sha256"]:
            raise PackError(f"endpoint output hash mismatch: {final_path}")
        return receipt
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{role}.partial-", dir=endpoint_root)
    )
    try:
        temporary_path = temporary / member_name
        with zipfile.ZipFile(graph_path, "r") as archive:
            zip_record = _copy_zip_member_to_file(archive, member_name, temporary_path)
        artifact = _validate_endpoint_array(temporary_path, role=role)
        artifact["path"] = str(final_path.resolve())
        body = {
            "schema": "round0010-endpoint-member-v1",
            "created_utc": utc_now(),
            "endpoint_role": role,
            "graph_path": str(graph_path),
            "graph_sha256": EXPECTED_GRAPH_SHA256,
            "conversion": "lossless decompression of ordered int32 NPY member",
            "zip_member": zip_record,
            "artifact": artifact,
        }
        receipt = seal_record(body)
        atomic_write_json(temporary / "receipt.json", receipt, replace=False)
        if final_dir.exists():
            raise PackError(f"endpoint output collision: {final_dir}")
        os.rename(temporary, final_dir)
        _fsync_directory(endpoint_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    observed = _validate_endpoint_array(final_path, role=role)
    if observed["sha256"] != receipt["artifact"]["sha256"]:
        raise PackError(f"endpoint changed after atomic commit: {final_path}")
    return receipt


def _verify_constant_weights(
    graph_path: Path, *, edge_count: int = EDGE_COUNT, k: int = GRAPH_K
) -> dict[str, Any]:
    payload_digest = hashlib.sha256()
    count = 0
    constant_bits: int | None = None
    started = time.monotonic()
    with zipfile.ZipFile(graph_path, "r") as archive:
        info = archive.getinfo("weights.npy")
        with archive.open(info, "r") as member:
            shape, fortran, dtype = _read_npy_header(member)
            if shape != (edge_count,) or fortran or dtype != np.dtype("<f4"):
                raise PackError(
                    f"weight member contract mismatch: shape={shape}, fortran={fortran}, dtype={dtype}"
                )
            remainder = b""
            while True:
                block = member.read(_HASH_BLOCK_BYTES)
                if not block:
                    break
                data = remainder + block
                aligned = len(data) - (len(data) % dtype.itemsize)
                payload = data[:aligned]
                remainder = data[aligned:]
                if not payload:
                    continue
                payload_digest.update(payload)
                values = np.frombuffer(payload, dtype=dtype)
                bits = values.view(np.uint32)
                if constant_bits is None:
                    constant_bits = int(bits[0])
                if not bool(np.all(bits == constant_bits)):
                    mismatch = int(np.flatnonzero(bits != constant_bits)[0])
                    raise PackError(f"graph weight is nonconstant at edge {count + mismatch}")
                count += int(values.size)
            if remainder:
                raise PackError("truncated float32 weight payload")
    if count != edge_count or constant_bits is None:
        raise PackError(f"weight count mismatch: {count} != {edge_count}")
    constant = np.array([constant_bits], dtype=np.uint32).view(np.float32)[0]
    expected = np.float32(1.0 / k)
    if int(np.array([expected], dtype=np.float32).view(np.uint32)[0]) != constant_bits:
        raise PackError(
            f"constant graph weight {float(constant)} does not equal float32(1/{k})"
        )
    return {
        "edge_count": count,
        "dtype": np.dtype("<f4").str,
        "constant_value_decimal": float(constant),
        "constant_value_float32_hex": float(constant).hex(),
        "constant_value_bits_hex": f"0x{constant_bits:08x}",
        "expected_semantics": f"float32(1/{k})",
        "payload_sha256": payload_digest.hexdigest(),
        "all_values_bitwise_equal": True,
        "sampling_semantics": "uniform-over-directed-edges",
        "cdf_required": False,
        "verification_wall_seconds": time.monotonic() - started,
    }


def _alignment_sample(
    source: RawSourceMap,
    target_path: Path,
    *,
    samples_per_corpus: int = 2_048,
    seed: int = 10_010,
) -> dict[str, Any]:
    targets = np.load(target_path, mmap_mode="r", allow_pickle=False)
    if tuple(targets.shape) != (EDGE_COUNT,) or targets.dtype != ENDPOINT_DTYPE:
        raise PackError("target endpoint contract changed before alignment sample")
    rng = np.random.default_rng(seed)
    reports: list[dict[str, Any]] = []
    all_sample_ids: list[np.ndarray] = []
    for corpus_order, corpus in enumerate([name for name, _ in DEFAULT_SOURCE_SPECS]):
        block_start = corpus_order * ROWS_PER_CORPUS
        nodes = rng.choice(ROWS_PER_CORPUS, size=samples_per_corpus, replace=False).astype(
            np.int64
        )
        nodes += block_start
        slots = rng.integers(0, GRAPH_K, size=samples_per_corpus, dtype=np.int64)
        edge_positions = nodes * GRAPH_K + slots
        neighbors = np.asarray(targets[edge_positions], dtype=np.int64)
        random_neighbors = rng.integers(0, TOTAL_ROWS, size=samples_per_corpus, dtype=np.int64)
        random_neighbors[random_neighbors == nodes] = (
            random_neighbors[random_neighbors == nodes] + 1
        ) % TOTAL_ROWS
        source_vectors = source.take(nodes)
        neighbor_vectors = source.take(neighbors)
        random_vectors = source.take(random_neighbors)

        def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            numerator = np.einsum("ij,ij->i", left, right)
            denominator = np.sqrt(np.einsum("ij,ij->i", left, left)) * np.sqrt(
                np.einsum("ij,ij->i", right, right)
            )
            if np.any(denominator <= 0):
                raise PackError("zero norm in graph alignment sample")
            return numerator / denominator

        edge_cosine = cosine(source_vectors, neighbor_vectors)
        random_cosine = cosine(source_vectors, random_vectors)
        edge_mean = float(np.mean(edge_cosine))
        random_mean = float(np.mean(random_cosine))
        if edge_mean < 0.40 or edge_mean - random_mean < 0.25:
            raise PackError(
                f"graph/data alignment failed for {corpus}: edge_mean={edge_mean}, "
                f"random_mean={random_mean}"
            )
        sample_matrix = np.column_stack((nodes, slots, neighbors, random_neighbors)).astype(
            "<i8", copy=False
        )
        all_sample_ids.append(sample_matrix)
        reports.append(
            {
                "corpus": corpus,
                "sample_count": samples_per_corpus,
                "node_min": int(np.min(nodes)),
                "node_max": int(np.max(nodes)),
                "edge_cosine_mean": edge_mean,
                "edge_cosine_min": float(np.min(edge_cosine)),
                "edge_cosine_p05": float(np.quantile(edge_cosine, 0.05)),
                "random_cosine_mean": random_mean,
                "mean_margin": edge_mean - random_mean,
                "pass": True,
            }
        )
    del targets
    sample_bytes = np.concatenate(all_sample_ids).tobytes(order="C")
    return {
        "method": "deterministic source-node/edge-slot sample against random-pair control",
        "seed": seed,
        "samples_per_corpus": samples_per_corpus,
        "sample_tuple_layout": ["source_node", "edge_slot", "target_node", "random_node"],
        "sample_tuples_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "acceptance": "edge cosine mean >=0.40 and edge-minus-random mean >=0.25 per corpus",
        "reports": reports,
        "aligned": True,
    }


def convert_and_verify_endpoints(
    root: os.PathLike[str] | str,
    inventory_path: os.PathLike[str] | str,
    materialization_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if root_path != Path("/data/latent-basemap/runs/round-0010"):
        raise PackError(f"production endpoint root must be Round 0010, got {root_path}")
    inventory = load_inventory(inventory_path)
    materialization = read_json(materialization_path)
    verify_sealed_record(materialization)
    if materialization.get("schema") != MATERIALIZATION_SCHEMA:
        raise PackError("endpoint conversion requires the complete materialization receipt")
    if materialization.get("source_inventory_receipt_sha256") != inventory["receipt_sha256"]:
        raise PackError("materialization and source inventory are not bound")
    graph_path = Path(inventory["graph"]["path"])
    graph_sha, _, _ = sha256_file(graph_path)
    if graph_sha != inventory["graph"]["sha256_full_file"]:
        raise PackError("graph changed after source inventory")
    endpoint_root = root_path / "endpoints"
    endpoint_root.mkdir(parents=True, exist_ok=True)
    _clean_owned_partial_directories(endpoint_root)
    started_utc = utc_now()
    started = time.monotonic()
    print("endpoints: extracting and validating ordered sources", flush=True)
    source_receipt = _extract_endpoint_atomic(
        graph_path, endpoint_root, role="sources"
    )
    print("endpoints: extracting and validating ordered targets", flush=True)
    target_receipt = _extract_endpoint_atomic(
        graph_path, endpoint_root, role="targets"
    )
    print("endpoints: proving all k15 weights are one exact constant", flush=True)
    weights = _verify_constant_weights(graph_path)
    print("endpoints: running deterministic graph/data alignment check", flush=True)
    alignment = _alignment_sample(
        RawSourceMap.from_inventory(inventory), endpoint_root / "targets" / "targets.npy"
    )
    body = {
        "schema": ENDPOINT_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": started_utc,
        "completed_utc": utc_now(),
        "source_inventory_receipt_sha256": inventory["receipt_sha256"],
        "materialization_receipt_sha256": materialization["receipt_sha256"],
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": graph_sha,
        "n_nodes": TOTAL_ROWS,
        "k": GRAPH_K,
        "edge_count": EDGE_COUNT,
        "source_endpoints": source_receipt,
        "target_endpoints": target_receipt,
        "weights": weights,
        "row_alignment": alignment,
        "absence_of_truncation": True,
        "endpoint_bounds_verified": True,
        "graph_row_identity_preserved": True,
        "phase_wall_seconds": time.monotonic() - started,
    }
    sealed = seal_record(body)
    destination = root_path / "receipts" / "endpoints-reopen.json"
    atomic_write_json(destination, sealed, replace=False)
    print(f"endpoints: sealed {destination} {sealed['receipt_sha256']}", flush=True)
    return sealed


def correct_endpoint_validation_receipt(
    root: os.PathLike[str] | str,
    original_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Additively correct the original target receipt's source-only field.

    The first endpoint receipt remains immutable history.  This function
    independently reopens both endpoint arrays and emits new member validation
    receipts plus a superseding aggregate without changing any endpoint bytes.
    """

    root_path = Path(root).resolve()
    original = _load_required_receipt(original_path, ENDPOINT_SCHEMA)
    started = time.monotonic()
    corrected_members: dict[str, dict[str, Any]] = {}
    for role in ("sources", "targets"):
        path = root_path / "endpoints" / role / f"{role}.npy"
        artifact = _validate_endpoint_array(path, role=role)
        original_member = original[
            "source_endpoints" if role == "sources" else "target_endpoints"
        ]
        if artifact["sha256"] != original_member["artifact"]["sha256"]:
            raise PackError(f"{role} changed before additive validation correction")
        body = {
            "schema": "round0010-endpoint-member-validation-v2",
            "created_utc": utc_now(),
            "endpoint_role": role,
            "supersedes_receipt_sha256": original_member["receipt_sha256"],
            "correction": (
                "source_repeat_k_row_order_exact is asserted only for the sources array; "
                "target ordering is represented by the lossless member hash and endpoint bounds"
            ),
            "artifact": artifact,
        }
        member = seal_record(body)
        atomic_write_json(
            root_path / "receipts" / f"endpoint-{role}-validation-v2.json",
            member,
            replace=False,
        )
        corrected_members[role] = member
    body = {
        key: value
        for key, value in original.items()
        if key
        not in {
            "receipt_sha256",
            "source_endpoints",
            "target_endpoints",
            "completed_utc",
            "phase_wall_seconds",
        }
    }
    body.update(
        {
            "completed_utc": utc_now(),
            "supersedes_receipt_sha256": original["receipt_sha256"],
            "correction": (
                "withdraw the inapplicable source-repeat-k field from target endpoint "
                "metadata; endpoint bytes, hashes, bounds, weights, and alignment are unchanged"
            ),
            "source_endpoints": corrected_members["sources"],
            "target_endpoints": corrected_members["targets"],
            "phase_wall_seconds": float(original["phase_wall_seconds"])
            + (time.monotonic() - started),
        }
    )
    receipt = seal_record(body)
    destination = root_path / "receipts" / "endpoints-reopen-correction-1.json"
    atomic_write_json(destination, receipt, replace=False)
    print(
        f"endpoints: additive validation correction sealed {destination} "
        f"{receipt['receipt_sha256']}",
        flush=True,
    )
    return receipt


class NpyShardMap:
    """Vectorized contiguous access to ordered materialized NPY shards."""

    def __init__(
        self,
        members: Sequence[Mapping[str, Any]],
        *,
        total_rows: int,
        dimension: int,
        dtype: np.dtype[Any],
    ) -> None:
        self.members = tuple(dict(member) for member in members)
        self.total_rows = int(total_rows)
        self.dimension = int(dimension)
        self.dtype = np.dtype(dtype)
        cursor = 0
        seen: set[str] = set()
        for member in self.members:
            path = str(member["path"])
            if path in seen:
                raise PackError(f"duplicate materialized shard {path}")
            seen.add(path)
            if int(member["global_row_start"]) != cursor:
                raise PackError("materialized shard ranges are not ordered and contiguous")
            stop = int(member["global_row_stop"])
            if stop <= cursor:
                raise PackError("materialized shard has an empty range")
            cursor = stop
        if cursor != self.total_rows:
            raise PackError(f"materialized shard map ends at {cursor}, expected {self.total_rows}")

    @classmethod
    def from_materialization(cls, receipt: Mapping[str, Any]) -> "NpyShardMap":
        verify_sealed_record(receipt)
        if receipt.get("schema") != MATERIALIZATION_SCHEMA:
            raise PackError("unexpected materialization receipt schema")
        return cls(
            receipt["ordered_shards"],
            total_rows=int(receipt["total_rows"]),
            dimension=int(receipt["dimension"]),
            dtype=np.dtype(receipt["dtype"]),
        )

    def read(self, start: int, stop: int) -> np.ndarray:
        if start < 0 or stop < start or stop > self.total_rows:
            raise IndexError("materialized row slice out of bounds")
        output = np.empty((stop - start, self.dimension), dtype=self.dtype)
        destination = 0
        for member in self.members:
            member_start = int(member["global_row_start"])
            member_stop = int(member["global_row_stop"])
            if member_stop <= start:
                continue
            if member_start >= stop:
                break
            global_lo = max(start, member_start)
            global_hi = min(stop, member_stop)
            local_lo = global_lo - member_start
            count = global_hi - global_lo
            array = np.load(member["path"], mmap_mode="r", allow_pickle=False)
            if array.dtype != self.dtype or tuple(array.shape) != (
                member_stop - member_start,
                self.dimension,
            ):
                raise PackError(f"materialized shard shape/dtype drift: {member['path']}")
            output[destination : destination + count] = array[local_lo : local_lo + count]
            destination += count
            del array
        if destination != stop - start:
            raise PackError("materialized shard map did not fill contiguous slice")
        return output


def verify_raw_map_members(
    members: Sequence[RawMapMember],
    *,
    total_rows: int,
    dimension: int,
    full_hash: bool,
    check_identity: bool = True,
) -> None:
    RawSourceMap(
        members,
        total_rows=total_rows,
        dimension=dimension,
        enforce_identity=check_identity,
    )
    paths = [str(member.path) for member in members]
    if len(paths) != len(set(paths)):
        raise PackError("raw source map contains duplicate paths")
    for member in members:
        observed = file_identity(member.path)
        if observed["size_bytes"] != member.full_rows * dimension * RAW_DTYPE.itemsize:
            raise PackError(f"raw source byte geometry changed: {member.path}")
        if check_identity and not _same_identity(member.identity, observed):
            raise PackError(f"raw source identity is stale: {member.path}")
        if full_hash:
            digest, _, _ = sha256_file(member.path)
            if digest != member.sha256:
                raise PackError(f"raw source hash mismatch: {member.path}")


def _stream_chunk_receipt(
    chunk_dir: Path,
    *,
    index: int,
    start: int,
    stop: int,
    output_dim: int,
    output_dtype: np.dtype[Any],
    transform_id: str,
) -> dict[str, Any]:
    receipt = _load_chunk_receipt(chunk_dir)
    if (
        receipt.get("schema") != "round0010-stream-output-chunk-v1"
        or receipt.get("chunk_index") != index
        or receipt.get("global_row_start") != start
        or receipt.get("global_row_stop") != stop
        or receipt.get("transform_id") != transform_id
    ):
        raise PackError(f"stream output receipt mismatch: {chunk_dir}")
    observed = _scan_npy(
        chunk_dir / "coordinates.npy",
        expected_shape=(stop - start, output_dim),
        expected_dtype=output_dtype,
        require_finite=True,
    )
    if observed["sha256"] != receipt["artifact"]["sha256"]:
        raise PackError(f"stream output hash mismatch: {chunk_dir}")
    return receipt


def stream_transform_to_npy_chunks(
    source: RawSourceMap,
    output_root: os.PathLike[str] | str,
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    transform_id: str,
    output_dim: int,
    output_dtype: np.dtype[Any] | str = np.dtype("<f4"),
    rows_per_chunk: int = 8,
    read_block_rows: int = 4,
    interrupt_after_chunks: int | None = None,
) -> dict[str, Any]:
    """Stream a CPU transform directly into atomically committed NPY chunks.

    This is the persistence/resume contract exercised in Round 0010.  The
    transform callback receives bounded float32 blocks; no output block list,
    concatenate, DataFrame, Torch, or CUDA object is constructed.
    """

    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    if rows_per_chunk <= 0 or read_block_rows <= 0 or output_dim <= 0:
        raise PackError("invalid stream transform geometry")
    dtype = np.dtype(output_dtype)
    receipts: list[dict[str, Any]] = []
    created = 0
    chunk_count = math.ceil(source.total_rows / rows_per_chunk)
    for index in range(chunk_count):
        start = index * rows_per_chunk
        stop = min(start + rows_per_chunk, source.total_rows)
        final_dir = destination / f"chunk-{index:05d}"
        if final_dir.exists():
            receipt = _stream_chunk_receipt(
                final_dir,
                index=index,
                start=start,
                stop=stop,
                output_dim=output_dim,
                output_dtype=dtype,
                transform_id=transform_id,
            )
        else:
            temporary = Path(
                tempfile.mkdtemp(prefix=f".chunk-{index:05d}.partial-", dir=destination)
            )
            try:
                temporary_path = temporary / "coordinates.npy"
                output = np.lib.format.open_memmap(
                    temporary_path,
                    mode="w+",
                    dtype=dtype,
                    shape=(stop - start, output_dim),
                )
                for block_start in range(start, stop, read_block_rows):
                    block_stop = min(block_start + read_block_rows, stop)
                    transformed = np.asarray(transform(source.read(block_start, block_stop)))
                    if transformed.shape != (block_stop - block_start, output_dim):
                        raise PackError(
                            f"stream transform returned {transformed.shape}, expected "
                            f"{(block_stop - block_start, output_dim)}"
                        )
                    transformed = transformed.astype(dtype, copy=False)
                    if not bool(np.all(np.isfinite(transformed))):
                        raise PackError("stream transform returned non-finite coordinates")
                    output[block_start - start : block_stop - start] = transformed
                output.flush()
                del output
                with temporary_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                artifact = _scan_npy(
                    temporary_path,
                    expected_shape=(stop - start, output_dim),
                    expected_dtype=dtype,
                    require_finite=True,
                )
                artifact["path"] = str((final_dir / "coordinates.npy").resolve())
                receipt = seal_record(
                    {
                        "schema": "round0010-stream-output-chunk-v1",
                        "created_utc": utc_now(),
                        "chunk_index": index,
                        "global_row_start": start,
                        "global_row_stop": stop,
                        "transform_id": transform_id,
                        "source_segments": source.source_segments(start, stop),
                        "direct_persistence": "numpy-open-memmap",
                        "artifact": artifact,
                    }
                )
                atomic_write_json(temporary / "receipt.json", receipt, replace=False)
                if final_dir.exists():
                    raise PackError(f"stream output collision: {final_dir}")
                os.rename(temporary, final_dir)
                _fsync_directory(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            receipt = _stream_chunk_receipt(
                final_dir,
                index=index,
                start=start,
                stop=stop,
                output_dim=output_dim,
                output_dtype=dtype,
                transform_id=transform_id,
            )
            created += 1
        receipts.append(receipt)
        if interrupt_after_chunks is not None and created >= interrupt_after_chunks:
            raise PlannedInterruption(
                f"planned stream interruption after {created} committed chunks"
            )
    capability_payload = {
        "schema": "round0010-stream-output-v1",
        "source_total_rows": source.total_rows,
        "source_dimension": source.dimension,
        "source_members": [
            {
                "path_basename": member.path.name,
                "global_start": member.global_start,
                "global_stop": member.global_stop,
                "local_start": member.local_start,
                "sha256": member.sha256,
            }
            for member in source.members
        ],
        "transform_id": transform_id,
        "output_dimension": output_dim,
        "output_dtype": dtype.str,
        "rows_per_chunk": rows_per_chunk,
        "ordered_chunks": [
            {
                "chunk_index": receipt["chunk_index"],
                "global_row_start": receipt["global_row_start"],
                "global_row_stop": receipt["global_row_stop"],
                "sha256": receipt["artifact"]["sha256"],
                "size_bytes": receipt["artifact"]["size_bytes"],
            }
            for receipt in receipts
        ],
    }
    return {
        "capability_payload": capability_payload,
        "capability_sha256": canonical_sha256(capability_payload),
        "created_chunks_this_invocation": created,
        "resumed_chunks_this_invocation": len(receipts) - created,
        "ordered_receipt_sha256": [receipt["receipt_sha256"] for receipt in receipts],
    }


def _write_raw_fixture(path: Path, array: np.ndarray) -> RawMapMember:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PackError(f"fixture output collision: {path}")
    contiguous = np.ascontiguousarray(array, dtype=RAW_DTYPE)
    with path.open("xb") as handle:
        handle.write(contiguous.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    digest, _, _ = sha256_file(path)
    return RawMapMember(
        path=path.resolve(),
        corpus="fixture",
        global_start=0,
        global_stop=len(contiguous),
        local_start=0,
        full_rows=len(contiguous),
        identity=file_identity(path),
        sha256=digest,
    )


def _expect_pack_error(name: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        operation()
    except (PackError, FileNotFoundError, IndexError, ValueError) as error:
        return {
            "case": name,
            "rejected": True,
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    raise PackError(f"mutation fixture {name!r} unexpectedly passed")


def run_fixture_suite(root: os.PathLike[str] | str) -> dict[str, Any]:
    root_path = Path(root).resolve()
    production_root = Path("/data/latent-basemap/runs/round-0010")
    try:
        root_path.relative_to(production_root)
    except ValueError as error:
        raise PackError(f"fixture root must remain under Round 0010: {root_path}") from error
    fixture_root = root_path / "fixtures"
    receipt_path = root_path / "receipts" / "fixture-receipt.json"
    if receipt_path.exists():
        existing = read_json(receipt_path)
        verify_sealed_record(existing)
        if existing.get("schema") != FIXTURE_SCHEMA:
            raise PackError("existing fixture receipt has wrong schema")
        return existing
    if fixture_root.exists() and any(fixture_root.iterdir()):
        raise PackError(f"refusing unreceipted pre-existing fixture root {fixture_root}")
    fixture_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    dimension = 4
    arrays = [
        np.arange(0, 28, dtype=np.float32).reshape(7, dimension) / 10,
        np.arange(28, 48, dtype=np.float32).reshape(5, dimension) / 10,
        np.arange(48, 84, dtype=np.float32).reshape(9, dimension) / 10,
    ]
    members: list[RawMapMember] = []
    cursor = 0
    for index, array in enumerate(arrays):
        raw_member = _write_raw_fixture(
            fixture_root / "source" / f"data-{index:05d}-of-00003.npy", array
        )
        members.append(
            dataclasses.replace(
                raw_member,
                corpus=f"fixture-{index}",
                global_start=cursor,
                global_stop=cursor + len(array),
            )
        )
        cursor += len(array)
    source = RawSourceMap(members, total_rows=cursor, dimension=dimension)
    expected = np.concatenate(arrays)
    cross_shard = source.read(5, 16)
    if not np.array_equal(cross_shard, expected[5:16]):
        raise PackError("contiguous multi-shard fixture returned reordered rows")
    indexed = np.array([20, 1, 11, 7, 20, 0], dtype=np.int64)
    if not np.array_equal(source.take(indexed), expected[indexed]):
        raise PackError("indexed multi-shard fixture returned incorrect rows")

    transform_id = "fixture-linear-v1: [x0+x1, x2-x3, row-sum]"

    def transform(block: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (block[:, 0] + block[:, 1], block[:, 2] - block[:, 3], np.sum(block, axis=1))
        ).astype(np.float32)

    interrupted = False
    try:
        stream_transform_to_npy_chunks(
            source,
            fixture_root / "stream-resume",
            transform,
            transform_id=transform_id,
            output_dim=3,
            rows_per_chunk=5,
            read_block_rows=2,
            interrupt_after_chunks=2,
        )
    except PlannedInterruption:
        interrupted = True
    if not interrupted:
        raise PackError("stream fixture did not interrupt at the registered boundary")
    resumed = stream_transform_to_npy_chunks(
        source,
        fixture_root / "stream-resume",
        transform,
        transform_id=transform_id,
        output_dim=3,
        rows_per_chunk=5,
        read_block_rows=2,
    )
    clean = stream_transform_to_npy_chunks(
        source,
        fixture_root / "stream-clean",
        transform,
        transform_id=transform_id,
        output_dim=3,
        rows_per_chunk=5,
        read_block_rows=2,
    )
    if resumed["capability_sha256"] != clean["capability_sha256"]:
        raise PackError("interrupted/resumed stream differs from clean stream")

    mutation_root = fixture_root / "mutations"
    mutation_root.mkdir()
    mutations: list[dict[str, Any]] = []
    mutations.append(
        _expect_pack_error(
            "reordered",
            lambda: verify_raw_map_members(
                list(reversed(members)),
                total_rows=cursor,
                dimension=dimension,
                full_hash=False,
            ),
        )
    )
    mutations.append(
        _expect_pack_error(
            "missing",
            lambda: verify_raw_map_members(
                [members[0], members[2]],
                total_rows=cursor,
                dimension=dimension,
                full_hash=False,
            ),
        )
    )
    mutations.append(
        _expect_pack_error(
            "duplicated",
            lambda: verify_raw_map_members(
                [members[0], members[0], *members[1:]],
                total_rows=cursor,
                dimension=dimension,
                full_hash=False,
            ),
        )
    )

    def copied_member(case: str) -> RawMapMember:
        source_path = members[0].path
        destination = mutation_root / case / source_path.name
        destination.parent.mkdir()
        shutil.copyfile(source_path, destination)
        digest, _, _ = sha256_file(destination)
        return dataclasses.replace(
            members[0],
            path=destination.resolve(),
            identity=file_identity(destination),
            sha256=digest,
        )

    truncated = copied_member("truncated")
    with truncated.path.open("r+b") as handle:
        handle.truncate(truncated.path.stat().st_size - RAW_DTYPE.itemsize)
        handle.flush()
        os.fsync(handle.fileno())
    mutations.append(
        _expect_pack_error(
            "truncated",
            lambda: verify_raw_map_members(
                [truncated],
                total_rows=truncated.global_stop,
                dimension=dimension,
                full_hash=False,
            ),
        )
    )
    stale = copied_member("stale")
    os.utime(stale.path, ns=(stale.identity["mtime_ns"] + 1, stale.identity["mtime_ns"] + 1))
    mutations.append(
        _expect_pack_error(
            "stale",
            lambda: verify_raw_map_members(
                [stale],
                total_rows=stale.global_stop,
                dimension=dimension,
                full_hash=False,
            ),
        )
    )
    mismatched = copied_member("hash-mismatched")
    with mismatched.path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
        handle.flush()
        os.fsync(handle.fileno())
    mutations.append(
        _expect_pack_error(
            "hash-mismatched",
            lambda: verify_raw_map_members(
                [mismatched],
                total_rows=mismatched.global_stop,
                dimension=dimension,
                full_hash=True,
                check_identity=False,
            ),
        )
    )
    body = {
        "schema": FIXTURE_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "torch_imported": "torch" in __import__("sys").modules,
        "headerless_loader": {
            "members": len(members),
            "rows": cursor,
            "dimension": dimension,
            "contiguous_cross_shard_slice": "pass",
            "vectorized_reordered_and_duplicate_index_take": "pass",
        },
        "streamed_transform_and_direct_memmap_persistence": {
            "interruption_observed": interrupted,
            "resume_capability_sha256": resumed["capability_sha256"],
            "clean_capability_sha256": clean["capability_sha256"],
            "byte_identity_match": True,
            "resume_created_chunks": resumed["created_chunks_this_invocation"],
            "resume_reused_chunks": resumed["resumed_chunks_this_invocation"],
            "atomic_chunk_receipts_verified": True,
        },
        "mutation_matrix": mutations,
        "all_mutations_rejected": all(item["rejected"] for item in mutations),
        "phase_wall_seconds": time.monotonic() - started,
    }
    if body["torch_imported"]:
        raise PackError("Torch was imported during CPU-only fixture execution")
    sealed = seal_record(body)
    atomic_write_json(receipt_path, sealed, replace=False)
    print(f"fixtures: sealed {receipt_path} {sealed['receipt_sha256']}", flush=True)
    return sealed


def _load_required_receipt(path: os.PathLike[str] | str, schema: str) -> dict[str, Any]:
    receipt = read_json(path)
    verify_sealed_record(receipt)
    if receipt.get("schema") != schema:
        raise PackError(f"receipt {path} has schema {receipt.get('schema')!r}, expected {schema!r}")
    return receipt


def _capability_payload(
    inventory: Mapping[str, Any],
    materialization: Mapping[str, Any],
    endpoints: Mapping[str, Any],
    fixtures: Mapping[str, Any],
) -> dict[str, Any]:
    source_members = [
        {
            "path": item["path"],
            "corpus": item["corpus"],
            "corpus_order": item["corpus_order"],
            "shard_order_within_corpus": item["shard_order_within_corpus"],
            "shard_index": item["shard_index"],
            "full_rows": item["full_rows"],
            "selected_local_row_start": item["selected_local_row_start"],
            "selected_local_row_stop": item["selected_local_row_stop"],
            "output_global_row_start": item["output_global_row_start"],
            "output_global_row_stop": item["output_global_row_stop"],
            "sha256_full_file": item["sha256_full_file"],
            "identity": item["identity"],
        }
        for item in inventory["sources"]
    ]
    materialized_members = [
        {
            "chunk_index": item["chunk_index"],
            "path": item["path"],
            "global_row_start": item["global_row_start"],
            "global_row_stop": item["global_row_stop"],
            "shape": item["shape"],
            "dtype": item["dtype"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
            "receipt_sha256": item["receipt_sha256"],
            "source_segments": item["source_segments"],
        }
        for item in materialization["ordered_shards"]
    ]
    # The endpoint receipts use singular keys in the aggregate and plural NPY
    # filenames internally.  Keep the capability's consumer-facing keys exact.
    endpoint_members = {
        "sources": {
            "path": endpoints["source_endpoints"]["artifact"]["path"],
            "shape": endpoints["source_endpoints"]["artifact"]["shape"],
            "dtype": endpoints["source_endpoints"]["artifact"]["dtype"],
            "sha256": endpoints["source_endpoints"]["artifact"]["sha256"],
            "size_bytes": endpoints["source_endpoints"]["artifact"]["size_bytes"],
            "receipt_sha256": endpoints["source_endpoints"]["receipt_sha256"],
        },
        "targets": {
            "path": endpoints["target_endpoints"]["artifact"]["path"],
            "shape": endpoints["target_endpoints"]["artifact"]["shape"],
            "dtype": endpoints["target_endpoints"]["artifact"]["dtype"],
            "sha256": endpoints["target_endpoints"]["artifact"]["sha256"],
            "size_bytes": endpoints["target_endpoints"]["artifact"]["size_bytes"],
            "receipt_sha256": endpoints["target_endpoints"]["receipt_sha256"],
        },
    }
    return {
        "schema": PACK_SCHEMA,
        "round_id": ROUND_ID,
        "row_universe": {
            "total_rows": TOTAL_ROWS,
            "dimension": DIMENSION,
            "blocked_corpus_order": [name for name, _ in DEFAULT_SOURCE_SPECS],
            "rows_per_corpus": ROWS_PER_CORPUS,
        },
        "raw_source": {
            "dtype": RAW_DTYPE.str,
            "representation": "raw-headerless",
            "ordered_members": source_members,
            "inventory_receipt_sha256": inventory["receipt_sha256"],
        },
        "materialized_fp16": {
            "dtype": MATERIALIZED_DTYPE.str,
            "ordered_members": materialized_members,
            "geometry": materialization["geometry"],
            "receipt_sha256": materialization["receipt_sha256"],
        },
        "graph": {
            "path": inventory["graph"]["path"],
            "sha256": inventory["graph"]["sha256_full_file"],
            "n_nodes": TOTAL_ROWS,
            "k": GRAPH_K,
            "edge_count": EDGE_COUNT,
            "ordered_int32_endpoints": endpoint_members,
            "weights": endpoints["weights"],
            "row_alignment": endpoints["row_alignment"],
            "receipt_sha256": endpoints["receipt_sha256"],
        },
        "loader_contract": {
            "raw": {
                "member_order": "ordered_members exactly as listed",
                "row_address": "byte_offset=(selected_local_row_start+local_row)*384*4",
                "dtype": "little-endian float32",
                "shape": "(full_file_bytes/(384*4),384)",
                "contiguous_slice": "vectorized member intersections; never per-row iteration",
            },
            "materialized": {
                "format": "NumPy NPY v1/v2, C order",
                "open": "numpy.load(path,mmap_mode='r',allow_pickle=False)",
                "member_order": "ordered_members exactly as listed",
                "contiguous_slice": "vectorized member intersections",
                "dtype": "little-endian float16",
            },
            "endpoints": {
                "format": "separate NumPy NPY arrays",
                "dtype": "little-endian int32",
                "source_row_rule": "sources[e] == floor(e/15)",
                "target_bounds": "0 <= target < 30000000",
            },
            "weights": {
                "storage": "do not materialize a CDF or redundant output array",
                "semantics": "uniform-over-directed-edges",
                "constant_bits_hex": endpoints["weights"]["constant_value_bits_hex"],
            },
            "streamed_output": {
                "persistence": "direct open_memmap chunks",
                "atomic_unit": "fsynced data+receipt directory rename",
                "resume": "reopen and hash every completed chunk before reuse",
            },
        },
        "fixture_receipt_sha256": fixtures["receipt_sha256"],
    }


def assemble_input_pack(
    root: os.PathLike[str] | str,
    *,
    inventory_path: os.PathLike[str] | str,
    materialization_path: os.PathLike[str] | str,
    endpoints_path: os.PathLike[str] | str,
    fixtures_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if root_path != Path("/data/latent-basemap/runs/round-0010"):
        raise PackError(f"production pack root must be Round 0010, got {root_path}")
    inventory = _load_required_receipt(inventory_path, INVENTORY_SCHEMA)
    validate_inventory_structure(inventory)
    materialization = _load_required_receipt(materialization_path, MATERIALIZATION_SCHEMA)
    endpoints = _load_required_receipt(endpoints_path, ENDPOINT_SCHEMA)
    fixtures = _load_required_receipt(fixtures_path, FIXTURE_SCHEMA)
    if materialization["source_inventory_receipt_sha256"] != inventory["receipt_sha256"]:
        raise PackError("materialization is not bound to the selected inventory")
    if endpoints["source_inventory_receipt_sha256"] != inventory["receipt_sha256"]:
        raise PackError("endpoints are not bound to the selected inventory")
    if endpoints["materialization_receipt_sha256"] != materialization["receipt_sha256"]:
        raise PackError("endpoints are not bound to the selected materialization")
    payload = _capability_payload(inventory, materialization, endpoints, fixtures)
    capability_hash = canonical_sha256(payload)
    source_selected_bytes = TOTAL_ROWS * DIMENSION * RAW_DTYPE.itemsize
    materialized_bytes = sum(int(item["size_bytes"]) for item in materialization["ordered_shards"])
    endpoint_bytes = sum(
        int(endpoints[key]["artifact"]["size_bytes"])
        for key in ("source_endpoints", "target_endpoints")
    )
    phase_walls = {
        "source_inventory": float(inventory["inventory_wall_seconds"]),
        "materialization": float(materialization["phase_wall_seconds"]),
        "endpoints_and_alignment": float(endpoints["phase_wall_seconds"]),
        "fixtures": float(fixtures["phase_wall_seconds"]),
    }
    body = {
        "schema": PACK_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "capability_name": PACK_SCHEMA,
        "capability_payload": payload,
        "capability_hash_sha256": capability_hash,
        "source_to_materialized_row_map": materialization["ordered_shards"],
        "storage": {
            "forecast": inventory["disk_forecast"],
            "actual_materialized_bytes": materialized_bytes,
            "actual_endpoint_bytes": endpoint_bytes,
            "actual_pack_payload_bytes": materialized_bytes + endpoint_bytes,
        },
        "measured_phase_wall_seconds": phase_walls,
        "measured_io_rates": {
            "inventory_full_source_plus_graph_bytes_per_second": (
                sum(int(item["identity"]["size_bytes"]) for item in inventory["sources"])
                + int(inventory["graph"]["size_bytes"])
            )
            / max(float(inventory["inventory_wall_seconds"]), 1e-9),
            "materialization_selected_input_plus_output_bytes_per_second": (
                source_selected_bytes + materialized_bytes
            )
            / max(float(materialization["phase_wall_seconds"]), 1e-9),
            "endpoint_output_bytes_per_second": endpoint_bytes
            / max(float(endpoints["phase_wall_seconds"]), 1e-9),
        },
        "receipt_hashes": {
            "source_inventory": inventory["receipt_sha256"],
            "materialization": materialization["receipt_sha256"],
            "endpoints": endpoints["receipt_sha256"],
            "fixtures": fixtures["receipt_sha256"],
        },
        "claims": {
            "cpu_only_input_pack": True,
            "training_readiness": False,
            "gpu_performance": False,
            "panel_quality": False,
            "execution_seal": False,
            "scale_result": False,
        },
    }
    manifest = seal_record(body, field="manifest_receipt_sha256")
    destination = root_path / f"{PACK_SCHEMA}.json"
    atomic_write_json(destination, manifest, replace=False)
    print(f"pack: assembled {destination} capability={capability_hash}", flush=True)
    return manifest


def reopen_input_pack(
    root: os.PathLike[str] | str,
    manifest_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest = read_json(manifest_path)
    verify_sealed_record(manifest, field="manifest_receipt_sha256")
    if manifest.get("schema") != PACK_SCHEMA:
        raise PackError("unexpected input-pack manifest schema")
    payload = manifest.get("capability_payload")
    if not isinstance(payload, dict):
        raise PackError("input-pack capability payload is missing")
    capability_hash = canonical_sha256(payload)
    if capability_hash != manifest.get("capability_hash_sha256"):
        raise PackError("input-pack capability hash does not reproduce")
    started_utc = utc_now()
    started = time.monotonic()
    source_inventory_path = root_path / "receipts" / "source-inventory.json"
    inventory = load_inventory(source_inventory_path)
    print("reopen: verifying complete raw-source identities", flush=True)
    source_verification = verify_inventory_sources(inventory, full_hash=True)
    graph_path = Path(payload["graph"]["path"])
    graph_sha, graph_size, graph_hash_wall = sha256_file(graph_path)
    if graph_sha != payload["graph"]["sha256"]:
        raise PackError("graph hash mismatch during complete pack reopen")
    materialization = _load_required_receipt(
        root_path / "receipts" / "materialization-reopen.json", MATERIALIZATION_SCHEMA
    )
    materialized_map = NpyShardMap.from_materialization(materialization)
    print("reopen: verifying every materialized fp16 shard", flush=True)
    materialized_verified: list[dict[str, Any]] = []
    for item in payload["materialized_fp16"]["ordered_members"]:
        observed = _scan_npy(
            Path(item["path"]),
            expected_shape=tuple(int(value) for value in item["shape"]),
            expected_dtype=MATERIALIZED_DTYPE,
            require_finite=True,
        )
        if observed["sha256"] != item["sha256"] or observed["size_bytes"] != item["size_bytes"]:
            raise PackError(f"materialized member mismatch during reopen: {item['path']}")
        materialized_verified.append(observed)
    print("reopen: verifying endpoint identities, bounds, order, and weights", flush=True)
    source_endpoint = _validate_endpoint_array(
        Path(payload["graph"]["ordered_int32_endpoints"]["sources"]["path"]),
        role="sources",
    )
    target_endpoint = _validate_endpoint_array(
        Path(payload["graph"]["ordered_int32_endpoints"]["targets"]["path"]),
        role="targets",
    )
    for role, observed in (("sources", source_endpoint), ("targets", target_endpoint)):
        if observed["sha256"] != payload["graph"]["ordered_int32_endpoints"][role]["sha256"]:
            raise PackError(f"{role} endpoint hash mismatch during reopen")
    weights = _verify_constant_weights(graph_path)
    if weights["payload_sha256"] != payload["graph"]["weights"]["payload_sha256"]:
        raise PackError("weight payload hash mismatch during reopen")
    raw_map = RawSourceMap.from_inventory(inventory)
    boundary_checks: list[dict[str, Any]] = []
    boundaries = sorted(
        {
            int(item["global_row_stop"])
            for item in payload["materialized_fp16"]["ordered_members"][:-1]
        }
        | {ROWS_PER_CORPUS, ROWS_PER_CORPUS * 2}
    )
    for boundary in boundaries:
        start = max(0, boundary - 2)
        stop = min(TOTAL_ROWS, boundary + 2)
        raw = raw_map.read(start, stop).astype(MATERIALIZED_DTYPE)
        materialized = materialized_map.read(start, stop)
        if not np.array_equal(raw, materialized):
            raise PackError(f"source/materialized row order mismatch around boundary {boundary}")
        boundary_checks.append(
            {
                "boundary": boundary,
                "row_start": start,
                "row_stop": stop,
                "fp16_bytes_sha256": hashlib.sha256(raw.tobytes(order="C")).hexdigest(),
                "match": True,
            }
        )
    fixture = _load_required_receipt(
        root_path / "receipts" / "fixture-receipt.json", FIXTURE_SCHEMA
    )
    if fixture["receipt_sha256"] != payload["fixture_receipt_sha256"]:
        raise PackError("fixture receipt hash mismatch during pack reopen")
    reproduced = canonical_sha256(manifest["capability_payload"])
    if reproduced != capability_hash:
        raise PackError("capability hash changed after artifact reopen")
    body = {
        "schema": "30m-input-pack-v1-reopen-receipt",
        "round_id": ROUND_ID,
        "created_utc": started_utc,
        "completed_utc": utc_now(),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_receipt_sha256": manifest["manifest_receipt_sha256"],
        "capability_hash_sha256": reproduced,
        "source_verification": source_verification,
        "graph_verification": {
            "sha256": graph_sha,
            "size_bytes": graph_size,
            "hash_wall_seconds": graph_hash_wall,
        },
        "materialized_members_verified": materialized_verified,
        "source_endpoint": source_endpoint,
        "target_endpoint": target_endpoint,
        "weights": weights,
        "source_to_materialized_boundary_checks": boundary_checks,
        "fixture_receipt_sha256": fixture["receipt_sha256"],
        "complete_reopen": True,
        "phase_wall_seconds": time.monotonic() - started,
    }
    receipt = seal_record(body)
    destination = root_path / "receipts" / "pack-reopen.json"
    atomic_write_json(destination, receipt, replace=False)
    print(f"reopen: sealed {destination} capability={reproduced}", flush=True)
    return receipt


def seal_verification_logs(
    root: os.PathLike[str] | str,
    records: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    parsed: list[dict[str, Any]] = []
    passed_pattern = re.compile(r"(?P<count>\d+) passed")
    for record in records:
        name = record.get("name")
        command = record.get("command")
        path_value = record.get("path")
        if not name or not command or not path_value:
            raise PackError("verification log record needs name, command, and path")
        path = Path(path_value).resolve()
        try:
            path.relative_to(root_path)
        except ValueError as error:
            raise PackError(f"verification log lies outside Round 0010 root: {path}") from error
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(passed_pattern.finditer(text))
        if not matches or re.search(r"(?:^|\s)(?:failed|error)(?:s|\s|$)", text, re.IGNORECASE):
            raise PackError(f"verification log does not record a clean pytest pass: {path}")
        digest, size, hash_wall = sha256_file(path)
        parsed.append(
            {
                "name": name,
                "command": command,
                "path": str(path),
                "sha256": digest,
                "size_bytes": size,
                "hash_wall_seconds": hash_wall,
                "passed": int(matches[-1].group("count")),
            }
        )
    body = {
        "schema": "round0010-cuda-hidden-verification-v1",
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX"),
        "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "torch_imported": "torch" in __import__("sys").modules,
        "records": parsed,
    }
    if body["cuda_visible_devices"] != "" or body["torch_imported"]:
        raise PackError("verification sealing process was not CUDA-hidden and Torch-free")
    receipt = seal_record(body)
    destination = root_path / "receipts" / "verification.json"
    atomic_write_json(destination, receipt, replace=False)
    print(f"verification: sealed {destination} {receipt['receipt_sha256']}", flush=True)
    return receipt
