from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from basemap.minilm_input_pack import (
    PackError,
    PlannedInterruption,
    RawMapMember,
    RawSourceMap,
    canonical_sha256,
    file_identity,
    read_json,
    sha256_file,
    stream_transform_to_npy_chunks,
    verify_sealed_record,
)
from basemap.minilm_input_pack_round0012 import (
    GRAPH_PROVENANCE_SCHEMA,
    _require_heavy_authority,
    _round0012_capability_payload,
)


def _write_member(path: Path, values: np.ndarray, *, start: int) -> RawMapMember:
    array = np.ascontiguousarray(values, dtype="<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes(order="C"))
    digest, _, _ = sha256_file(path)
    return RawMapMember(
        path=path.resolve(),
        corpus=path.stem,
        global_start=start,
        global_stop=start + len(array),
        local_start=0,
        full_rows=len(array),
        identity=file_identity(path),
        sha256=digest,
    )


def _source(tmp_path: Path) -> tuple[RawSourceMap, list[RawMapMember]]:
    arrays = [
        np.arange(0, 16, dtype=np.float32).reshape(4, 4),
        np.arange(16, 32, dtype=np.float32).reshape(4, 4),
    ]
    members = []
    cursor = 0
    for index, array in enumerate(arrays):
        member = _write_member(tmp_path / f"source-{index}.raw", array, start=cursor)
        members.append(member)
        cursor = member.global_stop
    return RawSourceMap(members, total_rows=cursor, dimension=4), members


def _transform(block: np.ndarray) -> np.ndarray:
    return np.column_stack((block[:, 0] + block[:, 1], block[:, 2] - block[:, 3]))


_KWARGS = {
    "transform_id": "round0012-test-linear-v1",
    "transform_implementation_sha256": hashlib.sha256(
        b"round0012-test-linear-numpy-v1"
    ).hexdigest(),
    "transform_config": {"expressions": ["x0+x1", "x2-x3"]},
    "output_dim": 2,
    "output_dtype": np.dtype("<f4"),
    "rows_per_chunk": 4,
    "read_block_rows": 2,
}


def _hash_tree(root: Path) -> dict[str, tuple[str, int]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest, size, _ = sha256_file(path)
            result[str(path.relative_to(root))] = (digest, size)
    return result


def test_module_is_cuda_hidden_and_torch_free() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert "torch" not in sys.modules


def test_canonical_plan_and_every_chunk_bind_complete_source_transform_and_layout(
    tmp_path: Path,
) -> None:
    source, members = _source(tmp_path / "source")
    output = tmp_path / "output"
    capability = stream_transform_to_npy_chunks(
        source, output, _transform, **_KWARGS
    )
    plan_receipt = read_json(output / "stream-plan.json")
    verify_sealed_record(plan_receipt)
    plan = plan_receipt["plan"]
    assert plan_receipt["stream_plan_sha256"] == canonical_sha256(plan)
    assert plan["ordered_source_members"] == [
        {
            "path": str(member.path),
            "corpus": member.corpus,
            "global_row_start": member.global_start,
            "global_row_stop": member.global_stop,
            "selected_local_row_start": member.local_start,
            "selected_local_row_stop": member.local_start + 4,
            "full_rows": member.full_rows,
            "sha256_full_file": member.sha256,
            "identity": dict(member.identity),
        }
        for member in members
    ]
    assert plan["transform"] == {
        "transform_id": _KWARGS["transform_id"],
        "implementation_sha256": _KWARGS["transform_implementation_sha256"],
        "config": _KWARGS["transform_config"],
    }
    assert plan["output"] == {"shape": [8, 2], "dtype": "<f4", "c_contiguous": True}
    assert plan["chunk_geometry"] == {
        "rows_per_chunk": 4,
        "read_block_rows": 2,
        "chunk_count": 2,
    }
    assert plan["destination_layout"]["chunk_directory_template"] == \
        "chunk-{chunk_index:05d}"
    for index in range(2):
        receipt = read_json(output / f"chunk-{index:05d}" / "receipt.json")
        verify_sealed_record(receipt)
        assert receipt["stream_plan_sha256"] == plan_receipt["stream_plan_sha256"]
        assert receipt["stream_plan_receipt_sha256"] == plan_receipt["receipt_sha256"]
        assert receipt["complete_source_identity_sha256"] == \
            plan["complete_source_identity_sha256"]
        assert receipt["source_segments_sha256"] == canonical_sha256(
            receipt["source_segments"]
        )
        assert capability["ordered_receipt_sha256"][index] == receipt["receipt_sha256"]


def test_correctly_rehashed_changed_source_rejects_before_reuse_or_production(
    tmp_path: Path,
) -> None:
    source, members = _source(tmp_path / "source")
    output = tmp_path / "interrupted"
    with pytest.raises(PlannedInterruption):
        stream_transform_to_npy_chunks(
            source, output, _transform, interrupt_after_chunks=1, **_KWARGS
        )
    before = _hash_tree(output)
    changed = np.arange(100, 116, dtype=np.float32).reshape(4, 4)
    members[0].path.write_bytes(np.ascontiguousarray(changed, dtype="<f4").tobytes())
    digest, _, _ = sha256_file(members[0].path)
    replacement = dataclasses.replace(
        members[0], identity=file_identity(members[0].path), sha256=digest
    )
    changed_source = RawSourceMap(
        [replacement, members[1]], total_rows=8, dimension=4
    )
    calls = 0

    def counted(block: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return _transform(block)

    with pytest.raises(PackError, match="stream plan mismatch"):
        stream_transform_to_npy_chunks(
            changed_source, output, counted, **_KWARGS
        )
    assert calls == 0
    assert _hash_tree(output) == before
    assert not (output / "chunk-00001").exists()


@pytest.mark.parametrize(
    "mutation",
    ["implementation", "config", "output-shape", "chunk-geometry"],
)
def test_any_complete_plan_mutation_rejects_before_existing_chunk_reuse(
    tmp_path: Path, mutation: str
) -> None:
    source, _ = _source(tmp_path / "source")
    output = tmp_path / "output"
    stream_transform_to_npy_chunks(source, output, _transform, **_KWARGS)
    kwargs = dict(_KWARGS)
    if mutation == "implementation":
        kwargs["transform_implementation_sha256"] = "f" * 64
    elif mutation == "config":
        kwargs["transform_config"] = {"expressions": ["different"]}
    elif mutation == "output-shape":
        kwargs["output_dim"] = 3
    else:
        kwargs["rows_per_chunk"] = 2
    calls = 0

    def counted(block: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return _transform(block)

    with pytest.raises(PackError, match="stream plan mismatch"):
        stream_transform_to_npy_chunks(source, output, counted, **kwargs)
    assert calls == 0


def test_destination_entries_outside_plan_fail_closed(tmp_path: Path) -> None:
    source, _ = _source(tmp_path / "source")
    output = tmp_path / "output"
    stream_transform_to_npy_chunks(source, output, _transform, **_KWARGS)
    (output / "unregistered-output.bin").write_bytes(b"user-owned")
    with pytest.raises(PackError, match="outside the canonical plan"):
        stream_transform_to_npy_chunks(source, output, _transform, **_KWARGS)
    assert (output / "unregistered-output.bin").read_bytes() == b"user-owned"


def test_full_reopen_requires_dual_manager_and_cli_authority(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text('{"heavy_io_authorized":false}\n')
    with pytest.raises(PackError, match="explicit CLI flag"):
        _require_heavy_authority(progress, command_authorized=True)
    progress.write_text('{"heavy_io_authorized":true}\n')
    with pytest.raises(PackError, match="explicit CLI flag"):
        _require_heavy_authority(progress, command_authorized=False)
    _require_heavy_authority(progress, command_authorized=True)


def test_corrected_capability_payload_names_own_receipts_and_diagnostic_limit() -> None:
    upstream_payload = {
        "round_id": "0010",
        "raw_source": {},
        "materialized_fp16": {},
        "graph": {},
        "loader_contract": {},
        "fixture_receipt_sha256": "a" * 64,
    }
    fixture = {
        "receipt_sha256": "b" * 64,
        "canonical_stream_plan": {
            "schema": "round0012-stream-plan-v1",
            "stream_plan_sha256": "c" * 64,
            "stream_plan_receipt_sha256": "d" * 64,
        },
        "changed_source_resume": {"rejected_before_chunk_reuse_or_production": True},
    }
    reopen = {
        "receipt_sha256": "e" * 64,
        "complete_reopen": True,
        "source_to_fp16_conversion": {
            "rows": 30_000_000,
            "persisted_fp16_payload_sha256": "f" * 64,
        },
    }
    provenance = {
        "schema": GRAPH_PROVENANCE_SCHEMA,
        "receipt_sha256": "1" * 64,
        "status": "no-registered-complete-provenance",
        "diagnostic_only": True,
        "claim": "diagnostic only",
        "registered_alignment_sha256": "2" * 64,
    }
    payload = _round0012_capability_payload(
        upstream_manifest={"capability_payload": upstream_payload},
        fixture=fixture,
        full_reopen=reopen,
        provenance=provenance,
    )
    assert payload["round_id"] == "0012"
    assert payload["fixture_receipt_sha256"] == "b" * 64
    assert payload["qualification"]["full_reopen_receipt_sha256"] == "e" * 64
    assert payload["graph"]["provenance"]["diagnostic_only"] is True
    assert payload["lineage"]["upstream_artifacts_overwritten"] is False
