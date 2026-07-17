"""Round 0012 correction and read-only qualification for the 30M MiniLM pack.

This module is CPU-only by contract.  It may reuse Round-0010 bytes read-only,
but every receipt it emits lives under the fresh Round-0012 root.  Torch is not
imported and CUDA visibility must be empty for every entry point.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .minilm_input_pack import (
    DEFAULT_SOURCE_SPECS,
    DIMENSION,
    EDGE_COUNT,
    ENDPOINT_SCHEMA,
    GRAPH_K,
    INVENTORY_SCHEMA,
    MATERIALIZATION_SCHEMA,
    MATERIALIZED_DTYPE,
    PACK_SCHEMA,
    RAW_DTYPE,
    ROWS_PER_CORPUS,
    TOTAL_ROWS,
    NpyShardMap,
    PackError,
    PlannedInterruption,
    RawMapMember,
    RawSourceMap,
    _alignment_sample,
    _load_required_receipt,
    _scan_npy,
    _validate_endpoint_array,
    _verify_constant_weights,
    _write_raw_fixture,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    file_identity,
    inspect_graph_npz,
    load_inventory,
    read_json,
    seal_record,
    sha256_file,
    stream_transform_to_npy_chunks,
    utc_now,
    validate_inventory_structure,
    verify_raw_map_members,
    verify_sealed_record,
)


ROUND_ID = "0012"
ROUND_ROOT = Path("/data/latent-basemap/runs/round-0012")
UPSTREAM_ROOT = Path("/data/latent-basemap/runs/round-0010")
FIXTURE_SCHEMA = "round0012-loader-fixtures-v1"
GRAPH_PROVENANCE_SCHEMA = "round0012-graph-provenance-v1"
FULL_REOPEN_SCHEMA = "round0012-30m-full-reopen-v1"
CAPABILITY_REOPEN_SCHEMA = "round0012-capability-reopen-v1"
EXPECTED_UPSTREAM_CAPABILITY = (
    "3a9e9f173a0d65dd0725fe9a23e2913f4dfb577ebd8fb1a7ad572f2868b722ba"
)
DEFAULT_BLOCK_ROWS = 32_768


def require_cuda_hidden() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PackError("Round 0012 requires CUDA_VISIBLE_DEVICES to be empty")
    if "torch" in sys.modules:
        raise PackError("Torch was imported in the Round 0012 CPU-only process")


def _round_root(root: os.PathLike[str] | str) -> Path:
    candidate = Path(root).resolve()
    if candidate != ROUND_ROOT:
        raise PackError(f"Round 0012 output root must be {ROUND_ROOT}, got {candidate}")
    return candidate


def _upstream_root(root: os.PathLike[str] | str) -> Path:
    candidate = Path(root).resolve()
    if candidate != UPSTREAM_ROOT:
        raise PackError(
            f"Round 0012 may reopen only the registered Round 0010 root, got {candidate}"
        )
    return candidate


def _stable_file(path: Path) -> dict[str, Any]:
    digest, size_bytes, hash_wall_seconds = sha256_file(path)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": size_bytes,
        "identity": file_identity(path),
        "hash_wall_seconds": hash_wall_seconds,
    }


def _expect_pack_error(case: str, operation) -> dict[str, Any]:
    try:
        operation()
    except PackError as error:
        return {
            "case": case,
            "rejected": True,
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    raise PackError(f"adversarial fixture {case!r} unexpectedly succeeded")


def _fixture_source(
    root: Path, arrays: Sequence[np.ndarray], *, prefix: str
) -> tuple[RawSourceMap, list[RawMapMember], np.ndarray]:
    members: list[RawMapMember] = []
    cursor = 0
    count = len(arrays)
    for index, values in enumerate(arrays):
        member = _write_raw_fixture(
            root / f"{prefix}-{index:05d}-of-{count:05d}.npy", values
        )
        selected = dataclasses.replace(
            member,
            corpus=f"{prefix}-{index}",
            global_start=cursor,
            global_stop=cursor + len(values),
        )
        members.append(selected)
        cursor += len(values)
    return (
        RawSourceMap(members, total_rows=cursor, dimension=arrays[0].shape[1]),
        members,
        np.concatenate(arrays),
    )


def _linear_transform(block: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (block[:, 0] + block[:, 1], block[:, 2] - block[:, 3])
    ).astype("<f4")


_FIXTURE_TRANSFORM_ID = "round0012-fixture-linear-v1"
_FIXTURE_TRANSFORM_CONFIG = {
    "expressions": ["x0+x1", "x2-x3"],
    "input_dtype": RAW_DTYPE.str,
    "output_dtype": np.dtype("<f4").str,
}
_FIXTURE_IMPLEMENTATION_SHA256 = canonical_sha256(
    {
        "implementation": "numpy.column_stack",
        "expressions": _FIXTURE_TRANSFORM_CONFIG["expressions"],
        "astype": "<f4",
    }
)


def _stream_kwargs() -> dict[str, Any]:
    return {
        "transform_id": _FIXTURE_TRANSFORM_ID,
        "transform_implementation_sha256": _FIXTURE_IMPLEMENTATION_SHA256,
        "transform_config": _FIXTURE_TRANSFORM_CONFIG,
        "output_dim": 2,
        "output_dtype": np.dtype("<f4"),
        "rows_per_chunk": 4,
        "read_block_rows": 2,
    }


def _stream_payload_files(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            item = _stable_file(path)
            item["relative_path"] = str(path.relative_to(root))
            records.append(item)
    return records


def run_round0012_fixtures(root: os.PathLike[str] | str = ROUND_ROOT) -> dict[str, Any]:
    """Run the corrected unchanged/changed-source and mutation fixtures once."""

    require_cuda_hidden()
    root_path = _round_root(root)
    receipt_path = root_path / "receipts" / "fixture-receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        verify_sealed_record(receipt)
        if receipt.get("schema") != FIXTURE_SCHEMA:
            raise PackError("existing Round 0012 fixture receipt has the wrong schema")
        return receipt

    fixture_root = root_path / "fixtures"
    if fixture_root.exists() and any(fixture_root.iterdir()):
        raise PackError(f"refusing unreceipted Round 0012 fixture root {fixture_root}")
    fixture_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    arrays = [
        np.arange(0, 20, dtype=np.float32).reshape(5, 4) / 10,
        np.arange(20, 32, dtype=np.float32).reshape(3, 4) / 10,
        np.arange(32, 56, dtype=np.float32).reshape(6, 4) / 10,
    ]
    source, members, expected = _fixture_source(
        fixture_root / "unchanged-source", arrays, prefix="unchanged"
    )
    interrupted = False
    try:
        stream_transform_to_npy_chunks(
            source,
            fixture_root / "unchanged-resume",
            _linear_transform,
            interrupt_after_chunks=2,
            **_stream_kwargs(),
        )
    except PlannedInterruption:
        interrupted = True
    if not interrupted:
        raise PackError("unchanged-source fixture did not interrupt")
    resumed = stream_transform_to_npy_chunks(
        source,
        fixture_root / "unchanged-resume",
        _linear_transform,
        **_stream_kwargs(),
    )
    clean = stream_transform_to_npy_chunks(
        source,
        fixture_root / "unchanged-clean",
        _linear_transform,
        **_stream_kwargs(),
    )
    if resumed["capability_sha256"] != clean["capability_sha256"]:
        raise PackError("unchanged interrupted/resumed capability differs from clean")
    resume_files = _stream_payload_files(fixture_root / "unchanged-resume")
    clean_files = _stream_payload_files(fixture_root / "unchanged-clean")
    resume_identity = {
        value["relative_path"]: (value["sha256"], value["size_bytes"])
        for value in resume_files
    }
    clean_identity = {
        value["relative_path"]: (value["sha256"], value["size_bytes"])
        for value in clean_files
    }
    if resume_identity != clean_identity:
        raise PackError("unchanged interrupted/resumed stream bytes differ from clean")
    observed = np.concatenate([
        np.load(path / "coordinates.npy", allow_pickle=False)
        for path in sorted((fixture_root / "unchanged-resume").glob("chunk-*"))
    ])
    if not np.array_equal(observed, _linear_transform(expected)):
        raise PackError("unchanged stream payload differs from the registered transform")

    changed_arrays = [
        np.arange(0, 16, dtype=np.float32).reshape(4, 4),
        np.arange(16, 32, dtype=np.float32).reshape(4, 4),
    ]
    changed_source, changed_members, _ = _fixture_source(
        fixture_root / "changed-source", changed_arrays, prefix="changed"
    )
    changed_output = fixture_root / "changed-resume"
    try:
        stream_transform_to_npy_chunks(
            changed_source,
            changed_output,
            _linear_transform,
            interrupt_after_chunks=1,
            **_stream_kwargs(),
        )
    except PlannedInterruption:
        pass
    else:
        raise PackError("changed-source fixture did not commit exactly one chunk")
    committed_before = _stream_payload_files(changed_output)
    before_identity = {
        value["relative_path"]: (value["sha256"], value["size_bytes"])
        for value in committed_before
    }

    changed_path = changed_members[0].path
    replacement = changed_path.with_name(f".{changed_path.name}.replacement")
    replacement_values = np.ascontiguousarray(changed_arrays[0] + 0.5, dtype=RAW_DTYPE)
    with replacement.open("xb") as handle:
        handle.write(replacement_values.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(replacement, changed_path)
    digest, _, _ = sha256_file(changed_path)
    replacement_member = dataclasses.replace(
        changed_members[0], identity=file_identity(changed_path), sha256=digest
    )
    changed_source_v2 = RawSourceMap(
        [replacement_member, *changed_members[1:]], total_rows=8, dimension=4
    )
    transform_calls = 0

    def counted_transform(block: np.ndarray) -> np.ndarray:
        nonlocal transform_calls
        transform_calls += 1
        return _linear_transform(block)

    rejection_message = None
    try:
        stream_transform_to_npy_chunks(
            changed_source_v2,
            changed_output,
            counted_transform,
            **_stream_kwargs(),
        )
    except PackError as error:
        rejection_message = str(error)
    if rejection_message is None or "stream plan mismatch" not in rejection_message:
        raise PackError("changed-source resume did not fail on the complete plan")
    committed_after = _stream_payload_files(changed_output)
    after_identity = {
        value["relative_path"]: (value["sha256"], value["size_bytes"])
        for value in committed_after
    }
    if transform_calls != 0 or before_identity != after_identity:
        raise PackError("changed-source rejection reused, produced, or changed stream output")
    if (changed_output / "chunk-00001").exists():
        raise PackError("changed-source rejection produced a second chunk")

    mutation_root = fixture_root / "mutations"
    mutation_root.mkdir()
    mutations = [
        _expect_pack_error(
            "reordered",
            lambda: verify_raw_map_members(
                list(reversed(members)),
                total_rows=source.total_rows,
                dimension=source.dimension,
                full_hash=False,
            ),
        ),
        _expect_pack_error(
            "missing",
            lambda: verify_raw_map_members(
                [members[0], members[2]],
                total_rows=source.total_rows,
                dimension=source.dimension,
                full_hash=False,
            ),
        ),
    ]

    truncated_path = mutation_root / "truncated.npy"
    shutil.copyfile(members[0].path, truncated_path)
    truncated_digest, _, _ = sha256_file(truncated_path)
    truncated = dataclasses.replace(
        members[0], path=truncated_path.resolve(),
        identity=file_identity(truncated_path), sha256=truncated_digest,
    )
    with truncated_path.open("r+b") as handle:
        handle.truncate(truncated_path.stat().st_size - RAW_DTYPE.itemsize)
        handle.flush()
        os.fsync(handle.fileno())
    mutations.append(_expect_pack_error(
        "truncated",
        lambda: verify_raw_map_members(
            [truncated], total_rows=truncated.global_stop,
            dimension=source.dimension, full_hash=False),
    ))

    mismatch_path = mutation_root / "hash-mismatch.npy"
    shutil.copyfile(members[0].path, mismatch_path)
    with mismatch_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
        handle.flush()
        os.fsync(handle.fileno())
    mismatched = dataclasses.replace(
        members[0], path=mismatch_path.resolve(), identity=file_identity(mismatch_path)
    )
    mutations.append(_expect_pack_error(
        "hash-mismatch",
        lambda: verify_raw_map_members(
            [mismatched], total_rows=mismatched.global_stop,
            dimension=source.dimension, full_hash=True),
    ))

    body = {
        "schema": FIXTURE_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "canonical_stream_plan": {
            "schema": resumed["capability_payload"]["plan"]["schema"],
            "stream_plan_sha256": resumed["stream_plan_sha256"],
            "stream_plan_receipt_sha256": resumed["stream_plan_receipt_sha256"],
            "complete_source_identity_sha256": resumed["capability_payload"][
                "complete_source_identity_sha256"
            ],
            "ordered_chunk_receipt_sha256": resumed["ordered_receipt_sha256"],
        },
        "unchanged_source_resume": {
            "interruption_observed": True,
            "capability_sha256": resumed["capability_sha256"],
            "clean_capability_sha256": clean["capability_sha256"],
            "byte_identical_to_clean": True,
            "created_chunks_after_resume": resumed["created_chunks_this_invocation"],
            "reused_chunks_after_resume": resumed["resumed_chunks_this_invocation"],
            "payload_files": resume_files,
        },
        "changed_source_resume": {
            "correctly_rehashed_replacement_sha256": digest,
            "rejected_before_transform_call": transform_calls == 0,
            "rejected_before_chunk_reuse_or_production": before_identity == after_identity,
            "capability_returned": False,
            "second_chunk_absent": True,
            "error": rejection_message,
            "committed_files_before": committed_before,
            "committed_files_after": committed_after,
        },
        "mutation_matrix": mutations,
        "all_mutations_rejected": all(value["rejected"] for value in mutations),
        "phase_wall_seconds": time.monotonic() - started,
    }
    if body["torch_imported"] or body["cuda_visible_devices"] != "":
        raise PackError("Round 0012 fixtures did not remain CUDA-hidden and Torch-free")
    receipt = seal_record(body)
    atomic_write_json(receipt_path, receipt, replace=False)
    print(f"round0012 fixtures: sealed {receipt_path} {receipt['receipt_sha256']}", flush=True)
    return receipt


def record_graph_provenance(
    root: os.PathLike[str] | str = ROUND_ROOT,
    upstream_root: os.PathLike[str] | str = UPSTREAM_ROOT,
) -> dict[str, Any]:
    """Search only registered Round-0010 JSON evidence for graph construction."""

    require_cuda_hidden()
    root_path = _round_root(root)
    upstream = _upstream_root(upstream_root)
    destination = root_path / "receipts" / "graph-provenance.json"
    if destination.exists():
        receipt = read_json(destination)
        verify_sealed_record(receipt)
        if receipt.get("schema") != GRAPH_PROVENANCE_SCHEMA:
            raise PackError("existing graph-provenance receipt has the wrong schema")
        return receipt

    inventory = load_inventory(upstream / "receipts" / "source-inventory.json")
    endpoints = _load_required_receipt(
        upstream / "receipts" / "endpoints-reopen-correction-1.json",
        ENDPOINT_SCHEMA,
    )
    registered_paths = [upstream / f"{PACK_SCHEMA}.json"] + sorted(
        (upstream / "receipts").glob("*.json")
    )
    inspected: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for path in registered_paths:
        value = read_json(path)
        schema = value.get("schema")
        descriptor = _stable_file(path)
        descriptor["schema"] = schema
        inspected.append(descriptor)
        if isinstance(schema, str) and "graph-construction" in schema:
            try:
                verify_sealed_record(value)
            except PackError:
                continue
            if (
                value.get("source_inventory_receipt_sha256")
                == inventory["receipt_sha256"]
                and value.get("graph_sha256") == inventory["graph"]["sha256_full_file"]
            ):
                eligible.append({"descriptor": descriptor, "receipt": value})
    if len(eligible) > 1:
        raise PackError("multiple registered graph-construction receipts are ambiguous")

    if eligible:
        status = "registered-complete-provenance-found"
        claim = "registered graph construction is bound to the complete source inventory"
        registered = eligible[0]
        diagnostic_only = False
    else:
        status = "no-registered-complete-provenance"
        claim = (
            "graph/source row semantics are supported only by the reproduced registered "
            "alignment diagnostic; no complete construction provenance is claimed"
        )
        registered = None
        diagnostic_only = True
    body = {
        "schema": GRAPH_PROVENANCE_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "search_scope": "only JSON evidence registered under the Round-0010 root",
        "inspected_registered_evidence": inspected,
        "source_inventory_receipt_sha256": inventory["receipt_sha256"],
        "graph_sha256": inventory["graph"]["sha256_full_file"],
        "status": status,
        "registered_graph_construction_receipt": registered,
        "diagnostic_only": diagnostic_only,
        "claim": claim,
        "registered_alignment": endpoints["row_alignment"],
        "registered_alignment_sha256": canonical_sha256(endpoints["row_alignment"]),
    }
    receipt = seal_record(body)
    atomic_write_json(destination, receipt, replace=False)
    print(f"round0012 graph provenance: sealed {destination}", flush=True)
    return receipt


def _load_upstream_closure(upstream: Path) -> dict[str, Any]:
    inventory_path = upstream / "receipts" / "source-inventory.json"
    materialization_path = upstream / "receipts" / "materialization-reopen.json"
    endpoints_path = upstream / "receipts" / "endpoints-reopen-correction-1.json"
    fixture_path = upstream / "receipts" / "fixture-receipt.json"
    manifest_path = upstream / f"{PACK_SCHEMA}.json"
    inventory = load_inventory(inventory_path)
    validate_inventory_structure(inventory)
    materialization = _load_required_receipt(
        materialization_path, MATERIALIZATION_SCHEMA
    )
    endpoints = _load_required_receipt(endpoints_path, ENDPOINT_SCHEMA)
    upstream_fixture = _load_required_receipt(
        fixture_path, "round0010-loader-fixtures-v1"
    )
    manifest = read_json(manifest_path)
    verify_sealed_record(manifest, field="manifest_receipt_sha256")
    if manifest.get("schema") != PACK_SCHEMA or manifest.get("round_id") != "0010":
        raise PackError("unexpected upstream input-pack manifest identity")
    payload = manifest.get("capability_payload")
    if not isinstance(payload, dict):
        raise PackError("upstream capability payload is missing")
    if (
        canonical_sha256(payload) != manifest.get("capability_hash_sha256")
        or manifest.get("capability_hash_sha256") != EXPECTED_UPSTREAM_CAPABILITY
    ):
        raise PackError("upstream provisional capability does not reproduce")
    receipt_hashes = {
        "source_inventory": inventory["receipt_sha256"],
        "materialization": materialization["receipt_sha256"],
        "endpoints": endpoints["receipt_sha256"],
        "fixtures": upstream_fixture["receipt_sha256"],
    }
    if manifest.get("receipt_hashes") != receipt_hashes:
        raise PackError("upstream manifest receipt closure does not reproduce")
    if (
        payload["raw_source"].get("inventory_receipt_sha256")
        != inventory["receipt_sha256"]
        or payload["materialized_fp16"].get("receipt_sha256")
        != materialization["receipt_sha256"]
        or payload["graph"].get("receipt_sha256") != endpoints["receipt_sha256"]
        or payload.get("fixture_receipt_sha256") != upstream_fixture["receipt_sha256"]
    ):
        raise PackError("upstream capability receipt bindings do not close")
    if (
        materialization.get("source_inventory_receipt_sha256")
        != inventory["receipt_sha256"]
        or endpoints.get("source_inventory_receipt_sha256")
        != inventory["receipt_sha256"]
        or endpoints.get("materialization_receipt_sha256")
        != materialization["receipt_sha256"]
    ):
        raise PackError("upstream receipt dependency chain does not close")
    if manifest.get("source_to_materialized_row_map") != materialization["ordered_shards"]:
        raise PackError("upstream manifest row map differs from materialization receipt")

    chunk_receipts: list[dict[str, Any]] = []
    for item in materialization["ordered_shards"]:
        path = Path(item["path"])
        receipt_path = path.parent / "receipt.json"
        receipt = read_json(receipt_path)
        verify_sealed_record(receipt)
        if (
            receipt.get("schema") != "round0010-fp16-chunk-v1"
            or receipt.get("receipt_sha256") != item["receipt_sha256"]
            or receipt.get("chunk_index") != item["chunk_index"]
            or receipt.get("global_row_start") != item["global_row_start"]
            or receipt.get("global_row_stop") != item["global_row_stop"]
            or receipt.get("source_segments") != item["source_segments"]
            or receipt.get("materialization_plan_sha256")
            != materialization["materialization_plan_sha256"]
            or receipt.get("artifact", {}).get("path") != item["path"]
            or receipt.get("artifact", {}).get("sha256") != item["sha256"]
            or receipt.get("artifact", {}).get("size_bytes") != item["size_bytes"]
        ):
            raise PackError(f"upstream materialized chunk receipt does not close: {path}")
        chunk_receipts.append(
            {
                "chunk_index": item["chunk_index"],
                "path": str(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
                "file": _stable_file(receipt_path),
            }
        )
    for member in (endpoints["source_endpoints"], endpoints["target_endpoints"]):
        verify_sealed_record(member)

    receipt_files = []
    for label, path in (
        ("source_inventory", inventory_path),
        ("materialization", materialization_path),
        ("endpoints", endpoints_path),
        ("fixtures", fixture_path),
    ):
        receipt_files.append({"label": label, **_stable_file(path)})
    return {
        "inventory": inventory,
        "materialization": materialization,
        "endpoints": endpoints,
        "upstream_fixture": upstream_fixture,
        "manifest": manifest,
        "manifest_file": _stable_file(manifest_path),
        "receipt_files": receipt_files,
        "chunk_receipts": chunk_receipts,
    }


def _update_progress(
    progress_path: Path,
    *,
    phase: str,
    current_log: str,
    live_child_pid: int | None,
    completed_item: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    progress = read_json(progress_path)
    now = utc_now()
    progress.update(
        sequence=int(progress.get("sequence", 0)) + 1,
        updated_utc=now,
        last_progress_utc=now,
        phase=phase,
        current_log=current_log,
        live_child_pid=live_child_pid,
        **fields,
    )
    if completed_item is not None:
        items = list(progress.get("completed_contract_items", []))
        if completed_item not in items:
            items.append(completed_item)
        progress["completed_contract_items"] = items
    atomic_write_json(progress_path, progress, replace=True)
    return progress


def _require_heavy_authority(progress_path: Path, *, command_authorized: bool) -> None:
    progress = read_json(progress_path)
    if not command_authorized or progress.get("heavy_io_authorized") is not True:
        raise PackError(
            "full Round 0012 reopen requires both the explicit CLI flag and the "
            "manager-authored heavy_io_authorized progress transition"
        )


def full_read_only_reopen(
    root: os.PathLike[str] | str = ROUND_ROOT,
    upstream_root: os.PathLike[str] | str = UPSTREAM_ROOT,
    *,
    progress_path: os.PathLike[str] | str = ROUND_ROOT / "management" / "progress.json",
    log_path: os.PathLike[str] | str = ROUND_ROOT / "logs" / "full-read-only-reopen.log",
    block_rows: int = DEFAULT_BLOCK_ROWS,
    heavy_io_authorized: bool = False,
) -> dict[str, Any]:
    """Complete one every-byte/every-row qualification without modifying upstream."""

    require_cuda_hidden()
    root_path = _round_root(root)
    upstream = _upstream_root(upstream_root)
    progress = Path(progress_path).resolve()
    if progress != root_path / "management" / "progress.json":
        raise PackError("full reopen progress path must be the Round 0012 durable cursor")
    _require_heavy_authority(progress, command_authorized=heavy_io_authorized)
    if not isinstance(block_rows, int) or isinstance(block_rows, bool) or block_rows <= 0:
        raise PackError("full reopen block_rows must be a positive integer")
    destination = root_path / "receipts" / "full-read-only-reopen.json"
    if destination.exists():
        receipt = read_json(destination)
        verify_sealed_record(receipt)
        if receipt.get("schema") != FULL_REOPEN_SCHEMA:
            raise PackError("existing full reopen receipt has the wrong schema")
        return receipt

    log_text = str(Path(log_path).resolve())
    started_utc = utc_now()
    started = time.monotonic()
    _update_progress(
        progress,
        phase="full-reopen-upstream-receipt-closure",
        current_log=log_text,
        live_child_pid=os.getpid(),
    )
    closure = _load_upstream_closure(upstream)
    inventory = closure["inventory"]
    materialization = closure["materialization"]
    endpoints = closure["endpoints"]
    provenance = _load_required_receipt(
        root_path / "receipts" / "graph-provenance.json", GRAPH_PROVENANCE_SCHEMA
    )

    print("round0012 reopen: hashing all 36 registered raw source files", flush=True)
    raw_verified: list[dict[str, Any]] = []
    for index, record in enumerate(inventory["sources"]):
        path = Path(record["path"])
        before = file_identity(path)
        if before != record["identity"]:
            raise PackError(f"raw source identity changed since Round 0010: {path}")
        digest, size_bytes, wall = sha256_file(path)
        after = file_identity(path)
        if (
            digest != record["sha256_full_file"]
            or size_bytes != record["identity"]["size_bytes"]
            or after != before
        ):
            raise PackError(f"raw source full hash/identity mismatch: {path}")
        raw_verified.append(
            {
                "source_index": index,
                "path": str(path),
                "corpus": record["corpus"],
                "selected_global_row_start": record["output_global_row_start"],
                "selected_global_row_stop": record["output_global_row_stop"],
                "sha256_full_file": digest,
                "size_bytes": size_bytes,
                "identity_before": before,
                "identity_after": after,
                "hash_wall_seconds": wall,
            }
        )
        _update_progress(
            progress,
            phase=f"full-reopen-raw-{index + 1:02d}-of-{len(inventory['sources']):02d}",
            current_log=log_text,
            live_child_pid=os.getpid(),
            full_reopen_raw_completed=index + 1,
        )

    raw_map = RawSourceMap.from_inventory(inventory)
    global_converted_digest = hashlib.sha256()
    global_persisted_digest = hashlib.sha256()
    materialized_verified: list[dict[str, Any]] = []
    conversion_rows = 0
    conversion_blocks = 0
    print("round0012 reopen: comparing every one of 30M fp32 rows to persisted fp16", flush=True)
    for index, item in enumerate(materialization["ordered_shards"]):
        path = Path(item["path"])
        before = file_identity(path)
        digest, size_bytes, hash_wall = sha256_file(path)
        if digest != item["sha256"] or size_bytes != item["size_bytes"]:
            raise PackError(f"materialized full-file hash mismatch: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(int(value) for value in item["shape"])
        if (
            tuple(array.shape) != expected_shape
            or array.dtype != MATERIALIZED_DTYPE
            or not array.flags.c_contiguous
        ):
            raise PackError(f"materialized shape/dtype/layout mismatch: {path}")
        start = int(item["global_row_start"])
        stop = int(item["global_row_stop"])
        if raw_map.source_segments(start, stop) != item["source_segments"]:
            raise PackError(f"materialized source-segment binding mismatch: {path}")
        converted_digest = hashlib.sha256()
        persisted_digest = hashlib.sha256()
        shard_blocks = 0
        for block_start in range(start, stop, block_rows):
            block_stop = min(block_start + block_rows, stop)
            raw = raw_map.read(block_start, block_stop)
            if not bool(np.all(np.isfinite(raw))):
                raise PackError(f"non-finite raw value in rows [{block_start},{block_stop})")
            converted = np.ascontiguousarray(raw.astype(MATERIALIZED_DTYPE))
            persisted = np.ascontiguousarray(
                array[block_start - start:block_stop - start]
            )
            if not bool(np.all(np.isfinite(persisted))):
                raise PackError(f"non-finite persisted fp16 value in {path}")
            if not np.array_equal(converted, persisted):
                mismatch = np.argwhere(converted != persisted)[0]
                raise PackError(
                    f"source-to-fp16 mismatch at global row "
                    f"{block_start + int(mismatch[0])}, column {int(mismatch[1])}"
                )
            converted_bytes = converted.tobytes(order="C")
            persisted_bytes = persisted.tobytes(order="C")
            converted_digest.update(converted_bytes)
            persisted_digest.update(persisted_bytes)
            global_converted_digest.update(converted_bytes)
            global_persisted_digest.update(persisted_bytes)
            rows = block_stop - block_start
            conversion_rows += rows
            conversion_blocks += 1
            shard_blocks += 1
        del array
        after = file_identity(path)
        if after != before:
            raise PackError(f"materialized artifact changed during reopen: {path}")
        converted_sha = converted_digest.hexdigest()
        persisted_sha = persisted_digest.hexdigest()
        if converted_sha != persisted_sha:
            raise PackError(f"materialized conversion digest mismatch: {path}")
        materialized_verified.append(
            {
                "chunk_index": item["chunk_index"],
                "path": str(path),
                "global_row_start": start,
                "global_row_stop": stop,
                "shape": list(expected_shape),
                "dtype": MATERIALIZED_DTYPE.str,
                "sha256_full_npy": digest,
                "size_bytes": size_bytes,
                "source_fp16_payload_sha256": converted_sha,
                "persisted_fp16_payload_sha256": persisted_sha,
                "blocks": shard_blocks,
                "identity_before": before,
                "identity_after": after,
                "hash_wall_seconds": hash_wall,
            }
        )
        _update_progress(
            progress,
            phase=f"full-reopen-fp16-{index + 1:02d}-of-{len(materialization['ordered_shards']):02d}",
            current_log=log_text,
            live_child_pid=os.getpid(),
            full_reopen_materialized_completed=index + 1,
            full_reopen_conversion_rows=conversion_rows,
        )

    converted_global = global_converted_digest.hexdigest()
    persisted_global = global_persisted_digest.hexdigest()
    if (
        conversion_rows != TOTAL_ROWS
        or converted_global != persisted_global
        or len(raw_verified) != 36
        or len(materialized_verified) != 30
    ):
        raise PackError("complete 30M source-to-fp16 conversion closure did not complete")

    _update_progress(
        progress,
        phase="full-reopen-graph-endpoints-weights",
        current_log=log_text,
        live_child_pid=os.getpid(),
    )
    print("round0012 reopen: verifying graph, endpoints, weights, and diagnostic alignment", flush=True)
    graph_path = Path(inventory["graph"]["path"])
    graph = inspect_graph_npz(
        graph_path, expected_sha256=inventory["graph"]["sha256_full_file"]
    )
    if graph["identity"] != inventory["graph"]["identity"]:
        raise PackError("graph identity changed since source inventory")
    source_endpoint = _validate_endpoint_array(
        Path(endpoints["source_endpoints"]["artifact"]["path"]), role="sources"
    )
    target_endpoint = _validate_endpoint_array(
        Path(endpoints["target_endpoints"]["artifact"]["path"]), role="targets"
    )
    for role, observed, expected in (
        ("sources", source_endpoint, endpoints["source_endpoints"]["artifact"]),
        ("targets", target_endpoint, endpoints["target_endpoints"]["artifact"]),
    ):
        if (
            observed["sha256"] != expected["sha256"]
            or observed["size_bytes"] != expected["size_bytes"]
            or observed["minimum"] != expected["minimum"]
            or observed["maximum"] != expected["maximum"]
        ):
            raise PackError(f"{role} endpoint closure mismatch")
    weights = _verify_constant_weights(graph_path)
    if (
        weights["payload_sha256"] != endpoints["weights"]["payload_sha256"]
        or weights["constant_value_bits_hex"]
        != endpoints["weights"]["constant_value_bits_hex"]
        or weights["edge_count"] != endpoints["weights"]["edge_count"]
    ):
        raise PackError("graph weight closure mismatch")
    alignment = _alignment_sample(raw_map, Path(target_endpoint["path"]))
    if canonical_sha256(alignment) != canonical_sha256(endpoints["row_alignment"]):
        raise PackError("registered graph/source alignment diagnostic did not reproduce")

    artifact_closure: list[dict[str, Any]] = []
    artifact_closure.extend(
        {
            "role": "raw-source",
            "path": value["path"],
            "sha256": value["sha256_full_file"],
            "size_bytes": value["size_bytes"],
            "identity_after": value["identity_after"],
        }
        for value in raw_verified
    )
    artifact_closure.extend(
        {
            "role": "materialized-fp16",
            "path": value["path"],
            "sha256": value["sha256_full_npy"],
            "size_bytes": value["size_bytes"],
            "identity_after": value["identity_after"],
        }
        for value in materialized_verified
    )
    artifact_closure.extend([
        {
            "role": "graph",
            "path": graph["path"],
            "sha256": graph["sha256_full_file"],
            "size_bytes": graph["size_bytes"],
            "identity_after": graph["identity"],
        },
        {
            "role": "source-endpoints",
            "path": source_endpoint["path"],
            "sha256": source_endpoint["sha256"],
            "size_bytes": source_endpoint["size_bytes"],
            "identity_after": source_endpoint["identity"],
        },
        {
            "role": "target-endpoints",
            "path": target_endpoint["path"],
            "sha256": target_endpoint["sha256"],
            "size_bytes": target_endpoint["size_bytes"],
            "identity_after": target_endpoint["identity"],
        },
    ])
    body = {
        "schema": FULL_REOPEN_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": started_utc,
        "completed_utc": utc_now(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "upstream_root": str(upstream),
        "upstream_capability_hash_sha256": EXPECTED_UPSTREAM_CAPABILITY,
        "upstream_manifest_file": closure["manifest_file"],
        "upstream_receipt_files": closure["receipt_files"],
        "upstream_chunk_receipts": closure["chunk_receipts"],
        "upstream_receipt_hashes": closure["manifest"]["receipt_hashes"],
        "raw_sources_verified": raw_verified,
        "materialized_shards_verified": materialized_verified,
        "source_to_fp16_conversion": {
            "scope": "every selected row and every dimension in canonical row order",
            "rows": conversion_rows,
            "dimensions": DIMENSION,
            "values": conversion_rows * DIMENSION,
            "blocks": conversion_blocks,
            "block_rows": block_rows,
            "source_cast_fp16_payload_sha256": converted_global,
            "persisted_fp16_payload_sha256": persisted_global,
            "byte_exact": True,
        },
        "graph": graph,
        "source_endpoint": source_endpoint,
        "target_endpoint": target_endpoint,
        "weights": weights,
        "reproduced_alignment_diagnostic": alignment,
        "graph_provenance_receipt_sha256": provenance["receipt_sha256"],
        "graph_semantic_claim_diagnostic_only": provenance["diagnostic_only"],
        "artifact_closure": artifact_closure,
        "complete_raw_file_count": len(raw_verified),
        "complete_materialized_shard_count": len(materialized_verified),
        "complete_reopen": True,
        "phase_wall_seconds": time.monotonic() - started,
    }
    if body["cuda_visible_devices"] != "" or body["torch_imported"]:
        raise PackError("full reopen process was not CUDA-hidden and Torch-free")
    receipt = seal_record(body)
    atomic_write_json(destination, receipt, replace=False)
    _update_progress(
        progress,
        phase="full-reopen-complete",
        current_log=log_text,
        live_child_pid=None,
        current_receipt=str(destination),
        completed_item="complete every-row 30M source-to-fp16 read-only reopen sealed",
        full_reopen_conversion_rows=conversion_rows,
    )
    print(f"round0012 reopen: sealed {destination} {receipt['receipt_sha256']}", flush=True)
    return receipt


def _round0012_capability_payload(
    *,
    upstream_manifest: Mapping[str, Any],
    fixture: Mapping[str, Any],
    full_reopen: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(upstream_manifest["capability_payload"])
    payload["round_id"] = ROUND_ID
    payload["raw_source"]["receipt_origin"] = "round0010-read-only-upstream"
    payload["materialized_fp16"]["receipt_origin"] = "round0010-read-only-upstream"
    payload["graph"]["receipt_origin"] = "round0010-read-only-upstream"
    payload["graph"]["provenance"] = {
        "receipt_sha256": provenance["receipt_sha256"],
        "status": provenance["status"],
        "diagnostic_only": provenance["diagnostic_only"],
        "claim": provenance["claim"],
        "registered_alignment_sha256": provenance["registered_alignment_sha256"],
    }
    payload["loader_contract"]["streamed_output"] = {
        "schema": fixture["canonical_stream_plan"]["schema"],
        "persistence": "direct open_memmap chunks",
        "atomic_unit": "fsynced data+receipt directory rename",
        "canonical_plan": (
            "exact ordered source members/full hashes and identities, transform "
            "implementation/config, output shape/dtype, chunk geometry, and naming"
        ),
        "resume": "compare the complete sealed plan before any chunk reuse or production",
        "stream_plan_sha256": fixture["canonical_stream_plan"]["stream_plan_sha256"],
        "stream_plan_receipt_sha256": fixture["canonical_stream_plan"][
            "stream_plan_receipt_sha256"
        ],
        "changed_source_rejected_before_reuse": fixture["changed_source_resume"][
            "rejected_before_chunk_reuse_or_production"
        ],
    }
    payload["fixture_receipt_sha256"] = fixture["receipt_sha256"]
    payload["qualification"] = {
        "schema": "round0012-30m-qualification-v1",
        "full_reopen_receipt_sha256": full_reopen["receipt_sha256"],
        "graph_provenance_receipt_sha256": provenance["receipt_sha256"],
        "rows_compared_source_fp32_to_persisted_fp16": full_reopen[
            "source_to_fp16_conversion"
        ]["rows"],
        "conversion_payload_sha256": full_reopen["source_to_fp16_conversion"][
            "persisted_fp16_payload_sha256"
        ],
        "complete_reopen": full_reopen["complete_reopen"],
    }
    payload["lineage"] = {
        "supersedes_round": "0010",
        "upstream_provisional_capability_sha256": EXPECTED_UPSTREAM_CAPABILITY,
        "upstream_bytes_reused_read_only": True,
        "upstream_artifacts_overwritten": False,
    }
    return payload


def assemble_round0012_capability(
    root: os.PathLike[str] | str = ROUND_ROOT,
    upstream_root: os.PathLike[str] | str = UPSTREAM_ROOT,
) -> dict[str, Any]:
    require_cuda_hidden()
    root_path = _round_root(root)
    upstream = _upstream_root(upstream_root)
    destination = root_path / f"{PACK_SCHEMA}.json"
    if destination.exists():
        manifest = read_json(destination)
        verify_sealed_record(manifest, field="manifest_receipt_sha256")
        return manifest
    closure = _load_upstream_closure(upstream)
    fixture = _load_required_receipt(
        root_path / "receipts" / "fixture-receipt.json", FIXTURE_SCHEMA
    )
    full_reopen = _load_required_receipt(
        root_path / "receipts" / "full-read-only-reopen.json", FULL_REOPEN_SCHEMA
    )
    provenance = _load_required_receipt(
        root_path / "receipts" / "graph-provenance.json", GRAPH_PROVENANCE_SCHEMA
    )
    if (
        full_reopen.get("upstream_capability_hash_sha256")
        != EXPECTED_UPSTREAM_CAPABILITY
        or full_reopen.get("upstream_receipt_hashes")
        != closure["manifest"]["receipt_hashes"]
        or full_reopen.get("graph_provenance_receipt_sha256")
        != provenance["receipt_sha256"]
        or full_reopen.get("complete_reopen") is not True
        or full_reopen.get("source_to_fp16_conversion", {}).get("rows") != TOTAL_ROWS
        or full_reopen.get("source_to_fp16_conversion", {}).get("byte_exact") is not True
    ):
        raise PackError("full reopen receipt does not qualify capability assembly")
    payload = _round0012_capability_payload(
        upstream_manifest=closure["manifest"], fixture=fixture,
        full_reopen=full_reopen, provenance=provenance,
    )
    capability_hash = canonical_sha256(payload)
    body = {
        "schema": PACK_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "capability_name": PACK_SCHEMA,
        "capability_payload": payload,
        "capability_hash_sha256": capability_hash,
        "own_receipt_hashes": {
            "fixtures": fixture["receipt_sha256"],
            "full_read_only_reopen": full_reopen["receipt_sha256"],
            "graph_provenance": provenance["receipt_sha256"],
        },
        "upstream_read_only_receipt_hashes": closure["manifest"]["receipt_hashes"],
        "upstream_manifest": closure["manifest_file"],
        "claims": {
            "cpu_only_input_pack": True,
            "content_bound_stream_resume": True,
            "complete_30m_source_to_fp16_reopen": True,
            "graph_semantic_provenance": not provenance["diagnostic_only"],
            "graph_alignment_diagnostic": True,
            "training_readiness": False,
            "gpu_performance": False,
            "execution_seal": False,
            "scale_result": False,
        },
    }
    manifest = seal_record(body, field="manifest_receipt_sha256")
    atomic_write_json(destination, manifest, replace=False)
    print(
        f"round0012 capability: assembled {destination} capability={capability_hash}",
        flush=True,
    )
    return manifest


def reopen_round0012_capability(
    root: os.PathLike[str] | str = ROUND_ROOT,
    upstream_root: os.PathLike[str] | str = UPSTREAM_ROOT,
) -> dict[str, Any]:
    """Separate metadata/identity reopen after the complete content scan."""

    require_cuda_hidden()
    root_path = _round_root(root)
    upstream = _upstream_root(upstream_root)
    destination = root_path / "receipts" / "capability-reopen.json"
    if destination.exists():
        receipt = read_json(destination)
        verify_sealed_record(receipt)
        if receipt.get("schema") != CAPABILITY_REOPEN_SCHEMA:
            raise PackError("existing capability reopen receipt has the wrong schema")
        return receipt
    manifest_path = root_path / f"{PACK_SCHEMA}.json"
    manifest = read_json(manifest_path)
    verify_sealed_record(manifest, field="manifest_receipt_sha256")
    if manifest.get("schema") != PACK_SCHEMA or manifest.get("round_id") != ROUND_ID:
        raise PackError("Round 0012 capability manifest identity is invalid")
    payload = manifest.get("capability_payload")
    if (
        not isinstance(payload, dict)
        or canonical_sha256(payload) != manifest.get("capability_hash_sha256")
    ):
        raise PackError("Round 0012 capability hash does not reproduce")
    fixture = _load_required_receipt(
        root_path / "receipts" / "fixture-receipt.json", FIXTURE_SCHEMA
    )
    full_reopen = _load_required_receipt(
        root_path / "receipts" / "full-read-only-reopen.json", FULL_REOPEN_SCHEMA
    )
    provenance = _load_required_receipt(
        root_path / "receipts" / "graph-provenance.json", GRAPH_PROVENANCE_SCHEMA
    )
    expected_own = {
        "fixtures": fixture["receipt_sha256"],
        "full_read_only_reopen": full_reopen["receipt_sha256"],
        "graph_provenance": provenance["receipt_sha256"],
    }
    if manifest.get("own_receipt_hashes") != expected_own:
        raise PackError("Round 0012 own-receipt closure does not reproduce")
    if (
        payload.get("fixture_receipt_sha256") != fixture["receipt_sha256"]
        or payload.get("qualification", {}).get("full_reopen_receipt_sha256")
        != full_reopen["receipt_sha256"]
        or payload.get("qualification", {}).get("graph_provenance_receipt_sha256")
        != provenance["receipt_sha256"]
    ):
        raise PackError("capability payload receipt closure does not reproduce")

    artifact_identities: list[dict[str, Any]] = []
    for artifact in full_reopen["artifact_closure"]:
        path = Path(artifact["path"])
        observed = file_identity(path)
        if observed != artifact["identity_after"]:
            raise PackError(f"qualified artifact identity changed after full reopen: {path}")
        artifact_identities.append(
            {"role": artifact["role"], "path": str(path), "identity": observed}
        )
    upstream_files: list[dict[str, Any]] = []
    for descriptor in [
        full_reopen["upstream_manifest_file"],
        *full_reopen["upstream_receipt_files"],
    ]:
        path = Path(descriptor["path"])
        digest, size_bytes, _ = sha256_file(path)
        if digest != descriptor["sha256"] or size_bytes != descriptor["size_bytes"]:
            raise PackError(f"upstream receipt/manifest changed after full reopen: {path}")
        upstream_files.append(
            {"path": str(path), "sha256": digest, "size_bytes": size_bytes}
        )
    closure = _load_upstream_closure(upstream)
    if (
        manifest.get("upstream_read_only_receipt_hashes")
        != closure["manifest"]["receipt_hashes"]
        or manifest.get("upstream_manifest", {}).get("sha256")
        != closure["manifest_file"]["sha256"]
    ):
        raise PackError("final capability upstream closure changed")
    manifest_file = _stable_file(manifest_path)
    body = {
        "schema": CAPABILITY_REOPEN_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "manifest_file": manifest_file,
        "manifest_receipt_sha256": manifest["manifest_receipt_sha256"],
        "capability_hash_sha256": manifest["capability_hash_sha256"],
        "own_receipt_hashes": expected_own,
        "upstream_receipt_hashes": closure["manifest"]["receipt_hashes"],
        "artifact_identities_reopened": artifact_identities,
        "upstream_receipt_files_reopened": upstream_files,
        "complete_content_scan_receipt_reopened": True,
        "capability_hash_reproduced": True,
        "torch_imported": "torch" in sys.modules,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if body["torch_imported"] or body["cuda_visible_devices"] != "":
        raise PackError("capability reopen was not CUDA-hidden and Torch-free")
    receipt = seal_record(body)
    atomic_write_json(destination, receipt, replace=False)
    print(
        f"round0012 capability: reopened {manifest['capability_hash_sha256']}",
        flush=True,
    )
    return receipt
