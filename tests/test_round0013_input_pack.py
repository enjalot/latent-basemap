from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from basemap.minilm_input_pack import (
    TRANSFORM_EXECUTION_SPEC_SEAL,
    PackError,
    PlannedInterruption,
    RawMapMember,
    RawSourceMap,
    atomic_write_json,
    build_transform_execution_spec,
    canonical_sha256,
    file_identity,
    read_json,
    seal_record,
    sha256_file,
    stream_transform_to_npy_chunks,
)


_RELEASE_ROOT = Path(__file__).resolve().parents[1]
_RELEASE_COMMIT = "c" * 40
_TRANSFORM_CONFIG = {"expressions": ["x0+x1", "x2-x3"]}
_DECLARED_TRANSFORM_ID = "round0013-adversarial-linear-v1"
_DECLARED_IMPLEMENTATION = hashlib.sha256(
    b"unchanged-caller-declared-implementation-label"
).hexdigest()
_A_CALLS = 0
_B_CALLS = 0


def _transform_a(block: np.ndarray) -> np.ndarray:
    global _A_CALLS
    _A_CALLS += 1
    return np.column_stack((block[:, 0] + block[:, 1], block[:, 2] - block[:, 3]))


def _transform_b(block: np.ndarray) -> np.ndarray:
    global _B_CALLS
    _B_CALLS += 1
    return np.column_stack((block[:, 0] + block[:, 1] + 100, block[:, 2] - block[:, 3]))


class _DestinationSentinel:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fspath_calls = 0

    def __fspath__(self) -> str:
        self.fspath_calls += 1
        return os.fspath(self.path)


class _CallableObject:
    def __call__(self, block: np.ndarray) -> np.ndarray:
        return _transform_a(block)


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


def _source(root: Path) -> RawSourceMap:
    arrays = [
        np.arange(0, 16, dtype=np.float32).reshape(4, 4),
        np.arange(16, 32, dtype=np.float32).reshape(4, 4),
    ]
    members = []
    cursor = 0
    for index, array in enumerate(arrays):
        member = _write_member(root / f"source-{index}.raw", array, start=cursor)
        members.append(member)
        cursor = member.global_stop
    return RawSourceMap(members, total_rows=cursor, dimension=4)


def _spec() -> dict[str, object]:
    return build_transform_execution_spec(
        _transform_a,
        release_root=_RELEASE_ROOT,
        release_commit=_RELEASE_COMMIT,
        transform_config=_TRANSFORM_CONFIG,
    )


def _kwargs() -> dict[str, object]:
    return {
        "transform_id": _DECLARED_TRANSFORM_ID,
        "transform_implementation_sha256": _DECLARED_IMPLEMENTATION,
        "transform_config": _TRANSFORM_CONFIG,
        "transform_execution_spec": _spec(),
        "release_root": _RELEASE_ROOT,
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


def test_round0013_module_is_cuda_hidden_and_torch_free() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import basemap.minilm_input_pack_round0013; "
            "print('torch_imported=' + str('torch' in sys.modules).lower())",
        ],
        cwd=_RELEASE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.stdout.strip() == "torch_imported=false"


def test_transform_a_then_b_same_declared_metadata_rejects_before_destination(
    tmp_path: Path,
) -> None:
    global _B_CALLS
    source = _source(tmp_path / "source")
    output = tmp_path / "resume"
    kwargs = _kwargs()
    with pytest.raises(PlannedInterruption):
        stream_transform_to_npy_chunks(
            source,
            output,
            _transform_a,
            interrupt_after_chunks=1,
            **kwargs,
        )
    assert (output / "chunk-00000").is_dir()
    assert not (output / "chunk-00001").exists()
    before = _hash_tree(output)
    destination = _DestinationSentinel(output)
    _B_CALLS = 0

    with pytest.raises(PackError, match="transform execution authentication failed"):
        stream_transform_to_npy_chunks(
            source,
            destination,
            _transform_b,
            **kwargs,
        )

    assert destination.fspath_calls == 0
    assert _B_CALLS == 0
    assert _hash_tree(output) == before
    assert (output / "chunk-00000").is_dir()
    assert not (output / "chunk-00001").exists()


def test_well_formed_supplied_code_digest_is_recomputed_before_destination(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source")
    bad_spec = copy.deepcopy(_spec())
    bad_spec["code"]["sha256"] = "f" * 64
    bad_spec.pop(TRANSFORM_EXECUTION_SPEC_SEAL)
    bad_spec[TRANSFORM_EXECUTION_SPEC_SEAL] = canonical_sha256(bad_spec)
    destination = _DestinationSentinel(tmp_path / "must-not-exist")
    kwargs = _kwargs()
    kwargs["transform_execution_spec"] = bad_spec
    with pytest.raises(PackError, match="transform execution authentication failed"):
        stream_transform_to_npy_chunks(
            source,
            destination,
            _transform_a,
            **kwargs,
        )
    assert destination.fspath_calls == 0
    assert not destination.path.exists()


def test_unauthenticatable_callable_object_is_rejected() -> None:
    with pytest.raises(PackError, match="unauthenticatable"):
        build_transform_execution_spec(
            _CallableObject(),
            release_root=_RELEASE_ROOT,
            release_commit=_RELEASE_COMMIT,
            transform_config=_TRANSFORM_CONFIG,
        )


def test_stale_chunk_receipt_rejects_before_callback_or_new_output(
    tmp_path: Path,
) -> None:
    global _A_CALLS
    source = _source(tmp_path / "source")
    output = tmp_path / "output"
    kwargs = _kwargs()
    stream_transform_to_npy_chunks(source, output, _transform_a, **kwargs)
    receipt_path = output / "chunk-00000" / "receipt.json"
    stale = read_json(receipt_path)
    stale.pop("receipt_sha256")
    stale["stream_plan_sha256"] = "e" * 64
    atomic_write_json(receipt_path, seal_record(stale), replace=True)
    before = _hash_tree(output)
    _A_CALLS = 0
    with pytest.raises(PackError, match="stream output receipt mismatch"):
        stream_transform_to_npy_chunks(source, output, _transform_a, **kwargs)
    assert _A_CALLS == 0
    assert _hash_tree(output) == before
