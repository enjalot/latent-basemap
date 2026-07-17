"""Round 0013 observed-transform correction and 30M read-only qualification."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .minilm_input_pack import (
    DIMENSION,
    PACK_SCHEMA,
    RAW_DTYPE,
    STREAM_PLAN_SCHEMA,
    TRANSFORM_EXECUTION_SPEC_SEAL,
    PackError,
    PlannedInterruption,
    RawMapMember,
    RawSourceMap,
    _load_required_receipt,
    _write_raw_fixture,
    atomic_write_json,
    build_transform_execution_spec,
    canonical_sha256,
    file_identity,
    read_json,
    seal_record,
    sha256_file,
    stream_transform_to_npy_chunks,
    utc_now,
    verify_raw_map_members,
    verify_sealed_record,
)
from .minilm_input_pack_round0012 import (
    DEFAULT_BLOCK_ROWS,
    UPSTREAM_ROOT,
    _load_upstream_closure,
    _stable_file,
    full_read_only_reopen as _full_read_only_reopen,
    record_graph_provenance as _record_graph_provenance,
)


ROUND_ID = "0013"
ROUND_ROOT = Path("/data/latent-basemap/runs/round-0013")
ROUND0012_ROOT = Path("/data/latent-basemap/runs/round-0012")
ISSUED_BASE = "fe4da5a0ac283955d8d45389c2ab7efe8fee8d2a"
EXPECTED_ROUND0012_MANIFEST_FILE_SHA256 = (
    "898f29e6c203d24e5ca2708c939e0d795fc1dc7ed024735c020cc2fb0ac7c460"
)
EXPECTED_ROUND0012_MANIFEST_RECEIPT = (
    "b58797e0407e93f1b1b454dc44ce018c9edb03bc1bfc4ac2b11bb990cbc6633c"
)
EXPECTED_ROUND0012_CAPABILITY = (
    "99a71782d892de89d8d0462d8acd679dcdceb8f7093ef50e15c32f152cd7a233"
)
INTAKE_SCHEMA = "round0013-admission-intake-v1"
PROGRESS_SCHEMA = "round0013-progress-v1"
TRANSFORM_SPEC_RECEIPT_SCHEMA = "round0013-transform-spec-receipt-v1"
FIXTURE_SCHEMA = "round0013-observed-transform-fixtures-v1"
GRAPH_PROVENANCE_SCHEMA = "round0013-graph-provenance-v1"
FULL_REOPEN_SCHEMA = "round0013-30m-full-reopen-v1"
CAPABILITY_REOPEN_SCHEMA = "round0013-capability-reopen-v1"
_CALLER_TRANSFORM_ID = "round0013-fixture-linear-v1"
_CALLER_IMPLEMENTATION_LABEL = canonical_sha256(
    {"caller_declared_label": "numpy-linear-v1", "round": ROUND_ID}
)
_TRANSFORM_CONFIG = {
    "expressions": ["x0+x1", "x2-x3"],
    "input_dtype": RAW_DTYPE.str,
    "output_dtype": np.dtype("<f4").str,
}
_TRANSFORM_A_CALLS = 0
_TRANSFORM_B_CALLS = 0


def require_cuda_hidden() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PackError("Round 0013 requires CUDA_VISIBLE_DEVICES to be empty")
    if "torch" in sys.modules:
        raise PackError("Torch was imported in the Round 0013 CPU-only process")


def _round_root(root: os.PathLike[str] | str) -> Path:
    candidate = Path(root).resolve()
    if candidate != ROUND_ROOT:
        raise PackError(f"Round 0013 output root must be {ROUND_ROOT}, got {candidate}")
    return candidate


def _release_root(root: os.PathLike[str] | str) -> Path:
    argument = Path(root)
    if (
        not argument.is_absolute()
        or argument.is_symlink()
        or argument.resolve() != argument
        or not argument.is_dir()
    ):
        raise PackError("release root must be a canonical absolute source directory")
    return argument


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if result.returncode != 0:
        error = result.stderr if isinstance(result.stderr, str) else result.stderr.decode()
        raise PackError(f"Git verification failed for {' '.join(arguments)}: {error.strip()}")
    return result.stdout


def _verify_release_source(release_root: Path, release_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", release_commit):
        raise PackError("release commit must be a full lowercase Git SHA")
    head = str(_git(release_root, "rev-parse", "HEAD")).strip()
    if head != release_commit:
        raise PackError(f"release checkout HEAD {head} differs from {release_commit}")
    status = str(
        _git(release_root, "status", "--porcelain=v1", "--untracked-files=all")
    )
    if status:
        raise PackError("release source checkout is not clean")
    parent = str(_git(release_root, "rev-parse", "HEAD^")).strip()
    if parent != ISSUED_BASE:
        raise PackError("Round 0013 release is not the issued base's direct child")
    forbidden: list[str] = []
    large: list[dict[str, Any]] = []
    for path in release_root.rglob("*"):
        if path.name in {".venv", "__pycache__", ".pytest_cache"} or path.suffix == ".pyc":
            forbidden.append(str(path.relative_to(release_root)))
        if path.is_file() and path.stat().st_size > 100 * 1024 * 1024:
            large.append(
                {"path": str(path.relative_to(release_root)), "size_bytes": path.stat().st_size}
            )
    if forbidden or large:
        raise PackError("release checkout contains a forbidden cache/environment/large file")
    return {
        "head": head,
        "parent": parent,
        "tree": str(_git(release_root, "rev-parse", "HEAD^{tree}")).strip(),
        "status_porcelain": status.splitlines(),
        "detached": subprocess.run(
            ["git", "-C", str(release_root), "symbolic-ref", "-q", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        != 0,
        "forbidden_source_entries": forbidden,
        "files_over_100mb": large,
    }


def _round0012_diagnostic() -> dict[str, Any]:
    path = ROUND0012_ROOT / f"{PACK_SCHEMA}.json"
    digest, size_bytes, _ = sha256_file(path)
    if digest != EXPECTED_ROUND0012_MANIFEST_FILE_SHA256:
        raise PackError("Round 0012 diagnostic manifest file hash changed")
    manifest = read_json(path)
    verify_sealed_record(manifest, field="manifest_receipt_sha256")
    payload = manifest.get("capability_payload")
    if (
        manifest.get("round_id") != "0012"
        or manifest.get("manifest_receipt_sha256")
        != EXPECTED_ROUND0012_MANIFEST_RECEIPT
        or manifest.get("capability_hash_sha256") != EXPECTED_ROUND0012_CAPABILITY
        or not isinstance(payload, dict)
        or canonical_sha256(payload) != EXPECTED_ROUND0012_CAPABILITY
    ):
        raise PackError("Round 0012 diagnostic manifest/capability does not reproduce")
    return {
        "status": "diagnostic-only-not-released",
        "path": str(path),
        "file_sha256": digest,
        "size_bytes": size_bytes,
        "manifest_receipt_sha256": manifest["manifest_receipt_sha256"],
        "capability_hash_sha256": manifest["capability_hash_sha256"],
        "verified_release_commit": ISSUED_BASE,
    }


def record_intake(
    release_commit: str,
    release_root: os.PathLike[str] | str,
    *,
    codex_child_pid: int,
    root: os.PathLike[str] | str = ROUND_ROOT,
) -> dict[str, Any]:
    require_cuda_hidden()
    root_path = _round_root(root)
    source_root = _release_root(release_root)
    destination = root_path / "receipts" / "admission-intake.json"
    if destination.exists():
        receipt = read_json(destination)
        verify_sealed_record(receipt)
        if receipt.get("schema") != INTAKE_SCHEMA:
            raise PackError("existing Round 0013 intake receipt has the wrong schema")
        return receipt
    dependency_paths = {
        "round0013": Path(
            "/home/enjalot/code/latent-labs/basemap-100m/round-0013-2026-07-17.md"
        ),
        "round0012": Path(
            "/home/enjalot/code/latent-labs/basemap-100m/round-0012-2026-07-17.md"
        ),
        "result0012": Path(
            "/home/enjalot/code/latent-labs/basemap-100m/result-0012-2026-07-17.md"
        ),
        "review0012": Path(
            "/home/enjalot/code/latent-labs/basemap-100m/review-0012-2026-07-17.md"
        ),
        "review0005": Path(
            "/home/enjalot/code/latent-labs/basemap-100m/review-0005-2026-07-17.md"
        ),
    }
    expected = {
        "round0013": "1b7cfafaf4df77139d05785ec96203092db18e2e95086ee8ffd29870bc48d27b",
        "round0012": "f0139bd5950b220cf42f4a8900c0e8bec1e7c5b8496593dc62f0e0ed63c7cffc",
        "result0012": "efb3893bb86ae8b9523ecf2ca982cfbe18719e18dea2552219428951769c7269",
        "review0012": "aa110ae8a7fba22aad4f5ba063f9a7fc660b606692e8eedb8532a3b82796b447",
        "review0005": "da8c9b2aae81980e0c02654eaa565f986fab940651baf7fef6388b476a3aa236",
    }
    dependencies: dict[str, Any] = {}
    for label, path in dependency_paths.items():
        digest, size_bytes, _ = sha256_file(path)
        if digest != expected[label]:
            raise PackError(f"issued dependency hash changed: {label}")
        dependencies[label] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": size_bytes,
        }
    progress_path = root_path / "management" / "progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_path.exists():
        raise PackError("Round 0013 progress cursor unexpectedly already exists")
    progress = {
        "schema": PROGRESS_SCHEMA,
        "round_id": ROUND_ID,
        "sequence": 0,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "last_progress_utc": utc_now(),
        "phase": "post-suite-production-intake",
        "current_log": str(root_path / "logs"),
        "live_child_pid": codex_child_pid,
        "heavy_io_authorized": True,
        "completed_contract_items": ["protocol/hash/admission intake"],
    }
    atomic_write_json(progress_path, progress, replace=False)
    body = {
        "schema": INTAKE_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "codex_child_pid": codex_child_pid,
        "release_root": str(source_root),
        "release": _verify_release_source(source_root, release_commit),
        "issued_dependencies": dependencies,
        "round0012_diagnostic_dependency": _round0012_diagnostic(),
        "external_interpreter": str(Path(sys.executable).resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "progress_path": str(progress_path),
    }
    receipt = seal_record(body)
    atomic_write_json(destination, receipt, replace=False)
    return receipt


def _transform_a(block: np.ndarray) -> np.ndarray:
    global _TRANSFORM_A_CALLS
    _TRANSFORM_A_CALLS += 1
    return np.column_stack(
        (block[:, 0] + block[:, 1], block[:, 2] - block[:, 3])
    ).astype("<f4")


def _transform_b(block: np.ndarray) -> np.ndarray:
    global _TRANSFORM_B_CALLS
    _TRANSFORM_B_CALLS += 1
    return np.column_stack(
        (block[:, 0] + block[:, 1] + 100, block[:, 2] - block[:, 3])
    ).astype("<f4")


def create_transform_execution_spec(
    release_commit: str,
    release_root: os.PathLike[str] | str,
    root: os.PathLike[str] | str = ROUND_ROOT,
) -> dict[str, Any]:
    require_cuda_hidden()
    root_path = _round_root(root)
    source_root = _release_root(release_root)
    destination = root_path / "receipts" / "transform-execution-spec.json"
    if destination.exists():
        receipt = read_json(destination)
        verify_sealed_record(receipt)
        if receipt.get("schema") != TRANSFORM_SPEC_RECEIPT_SCHEMA:
            raise PackError("existing transform spec receipt has the wrong schema")
        return receipt
    release = _verify_release_source(source_root, release_commit)
    spec = build_transform_execution_spec(
        _transform_a,
        release_root=source_root,
        release_commit=release_commit,
        transform_config=_TRANSFORM_CONFIG,
    )
    artifact = spec["defining_artifact"]
    committed_bytes = bytes(
        _git(
            source_root,
            "show",
            f"{release_commit}:{artifact['relative_path']}",
            binary=True,
        )
    )
    if (
        hashlib.sha256(committed_bytes).hexdigest() != artifact["sha256_full_file"]
        or len(committed_bytes) != artifact["size_bytes"]
    ):
        raise PackError("observed transform artifact bytes differ from the exact release blob")
    body = {
        "schema": TRANSFORM_SPEC_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "release": release,
        "transform_execution_spec": spec,
        "transform_execution_spec_sha256": spec[TRANSFORM_EXECUTION_SPEC_SEAL],
        "caller_declared_transform_id": _CALLER_TRANSFORM_ID,
        "caller_declared_implementation_sha256": _CALLER_IMPLEMENTATION_LABEL,
        "canonical_config": _TRANSFORM_CONFIG,
        "release_blob_sha256_reproduced": True,
    }
    receipt = seal_record(body)
    atomic_write_json(destination, receipt, replace=False)
    return receipt


class _DestinationSentinel:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fspath_calls = 0

    def __fspath__(self) -> str:
        self.fspath_calls += 1
        return os.fspath(self.path)


def _fixture_source(
    root: Path, arrays: Sequence[np.ndarray], *, prefix: str
) -> tuple[RawSourceMap, list[RawMapMember], np.ndarray]:
    members: list[RawMapMember] = []
    cursor = 0
    for index, values in enumerate(arrays):
        member = _write_raw_fixture(
            root / f"{prefix}-{index:05d}-of-{len(arrays):05d}.npy", values
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


def _payload_files(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            item = _stable_file(path)
            item["relative_path"] = str(path.relative_to(root))
            records.append(item)
    return records


def _tree_identity(root: Path) -> dict[str, tuple[str, int]]:
    return {
        item["relative_path"]: (item["sha256"], item["size_bytes"])
        for item in _payload_files(root)
    }


def _stream_kwargs(
    *, transform_spec: Mapping[str, Any], release_root: Path
) -> dict[str, Any]:
    return {
        "transform_id": _CALLER_TRANSFORM_ID,
        "transform_implementation_sha256": _CALLER_IMPLEMENTATION_LABEL,
        "transform_config": _TRANSFORM_CONFIG,
        "transform_execution_spec": dict(transform_spec),
        "release_root": release_root,
        "output_dim": 2,
        "output_dtype": np.dtype("<f4"),
        "rows_per_chunk": 4,
        "read_block_rows": 2,
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
    raise PackError(f"Round 0013 adversarial fixture {case!r} unexpectedly passed")


def run_round0013_fixtures(
    release_commit: str,
    release_root: os.PathLike[str] | str,
    root: os.PathLike[str] | str = ROUND_ROOT,
) -> dict[str, Any]:
    """Run A->A, exact A->B, source, declared-plan, and stale-receipt probes."""

    global _TRANSFORM_A_CALLS, _TRANSFORM_B_CALLS
    require_cuda_hidden()
    root_path = _round_root(root)
    source_root = _release_root(release_root)
    receipt_path = root_path / "receipts" / "fixture-receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        verify_sealed_record(receipt)
        if receipt.get("schema") != FIXTURE_SCHEMA:
            raise PackError("existing Round 0013 fixture receipt has the wrong schema")
        return receipt
    spec_receipt = _load_required_receipt(
        root_path / "receipts" / "transform-execution-spec.json",
        TRANSFORM_SPEC_RECEIPT_SCHEMA,
    )
    if spec_receipt["release"]["head"] != release_commit:
        raise PackError("transform specification release differs from fixture release")
    transform_spec = spec_receipt["transform_execution_spec"]
    kwargs = _stream_kwargs(transform_spec=transform_spec, release_root=source_root)
    fixture_root = root_path / "fixtures"
    if fixture_root.exists() and any(fixture_root.iterdir()):
        raise PackError(f"refusing unreceipted Round 0013 fixture root {fixture_root}")
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
    try:
        stream_transform_to_npy_chunks(
            source,
            fixture_root / "unchanged-resume",
            _transform_a,
            interrupt_after_chunks=2,
            **kwargs,
        )
    except PlannedInterruption:
        pass
    else:
        raise PackError("A->A fixture did not interrupt after two chunks")
    resumed = stream_transform_to_npy_chunks(
        source, fixture_root / "unchanged-resume", _transform_a, **kwargs
    )
    clean = stream_transform_to_npy_chunks(
        source, fixture_root / "unchanged-clean", _transform_a, **kwargs
    )
    resume_files = _payload_files(fixture_root / "unchanged-resume")
    clean_files = _payload_files(fixture_root / "unchanged-clean")
    resume_identity = {
        value["relative_path"]: (value["sha256"], value["size_bytes"])
        for value in resume_files
    }
    clean_identity = {
        value["relative_path"]: (value["sha256"], value["size_bytes"])
        for value in clean_files
    }
    observed = np.concatenate(
        [
            np.load(path / "coordinates.npy", allow_pickle=False)
            for path in sorted((fixture_root / "unchanged-resume").glob("chunk-*"))
        ]
    )
    if (
        resumed["capability_sha256"] != clean["capability_sha256"]
        or resume_identity != clean_identity
        or not np.array_equal(observed, _transform_a(expected))
    ):
        raise PackError("A->A interrupted resume is not byte-identical to clean")

    adversarial_arrays = [
        np.arange(100, 116, dtype=np.float32).reshape(4, 4),
        np.arange(116, 132, dtype=np.float32).reshape(4, 4),
    ]
    adversarial_source, _, _ = _fixture_source(
        fixture_root / "a-b-source", adversarial_arrays, prefix="a-b"
    )
    adversarial_output = fixture_root / "a-b-resume"
    try:
        stream_transform_to_npy_chunks(
            adversarial_source,
            adversarial_output,
            _transform_a,
            interrupt_after_chunks=1,
            **kwargs,
        )
    except PlannedInterruption:
        pass
    else:
        raise PackError("A->B fixture did not commit exactly one A chunk")
    if not (adversarial_output / "chunk-00000").is_dir() or (
        adversarial_output / "chunk-00001"
    ).exists():
        raise PackError("A->B fixture did not stop at exactly one committed chunk")
    adversarial_before = _payload_files(adversarial_output)
    adversarial_before_identity = _tree_identity(adversarial_output)
    destination_sentinel = _DestinationSentinel(adversarial_output)
    _TRANSFORM_B_CALLS = 0
    adversarial_error = None
    try:
        stream_transform_to_npy_chunks(
            adversarial_source,
            destination_sentinel,
            _transform_b,
            **kwargs,
        )
    except PackError as error:
        adversarial_error = str(error)
    if (
        adversarial_error is None
        or "transform execution authentication failed" not in adversarial_error
        or destination_sentinel.fspath_calls != 0
        or _TRANSFORM_B_CALLS != 0
        or _tree_identity(adversarial_output) != adversarial_before_identity
        or (adversarial_output / "chunk-00001").exists()
    ):
        raise PackError("exact A->B pre-destination rejection contract failed")
    adversarial_after = _payload_files(adversarial_output)

    changed_arrays = [
        np.arange(200, 216, dtype=np.float32).reshape(4, 4),
        np.arange(216, 232, dtype=np.float32).reshape(4, 4),
    ]
    changed_source, changed_members, _ = _fixture_source(
        fixture_root / "changed-source", changed_arrays, prefix="changed"
    )
    changed_output = fixture_root / "changed-resume"
    try:
        stream_transform_to_npy_chunks(
            changed_source,
            changed_output,
            _transform_a,
            interrupt_after_chunks=1,
            **kwargs,
        )
    except PlannedInterruption:
        pass
    else:
        raise PackError("changed-source fixture did not commit one chunk")
    changed_before = _tree_identity(changed_output)
    changed_path = changed_members[0].path
    replacement_values = np.ascontiguousarray(changed_arrays[0] + 0.5, dtype=RAW_DTYPE)
    replacement_path = changed_path.with_name(f".{changed_path.name}.replacement")
    with replacement_path.open("xb") as handle:
        handle.write(replacement_values.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(replacement_path, changed_path)
    digest, _, _ = sha256_file(changed_path)
    replacement_member = dataclasses.replace(
        changed_members[0], identity=file_identity(changed_path), sha256=digest
    )
    changed_source_v2 = RawSourceMap(
        [replacement_member, *changed_members[1:]], total_rows=8, dimension=4
    )
    _TRANSFORM_A_CALLS = 0
    changed_error = None
    try:
        stream_transform_to_npy_chunks(
            changed_source_v2, changed_output, _transform_a, **kwargs
        )
    except PackError as error:
        changed_error = str(error)
    if (
        changed_error is None
        or "stream plan mismatch" not in changed_error
        or _TRANSFORM_A_CALLS != 0
        or _tree_identity(changed_output) != changed_before
        or (changed_output / "chunk-00001").exists()
    ):
        raise PackError("changed-source rejection regression failed")
    changed_transform_calls = _TRANSFORM_A_CALLS
    changed_tree_byte_identical = _tree_identity(changed_output) == changed_before

    declared_plan_results: list[dict[str, Any]] = []
    declared_output = fixture_root / "unchanged-resume"
    declared_before = _tree_identity(declared_output)
    for case, updates in (
        ("implementation-label", {"transform_implementation_sha256": "f" * 64}),
        ("canonical-config", {"transform_config": {"expressions": ["different"]}}),
        ("output-shape", {"output_dim": 3}),
        ("chunk-geometry", {"rows_per_chunk": 2}),
    ):
        mutated = dict(kwargs)
        mutated.update(updates)
        _TRANSFORM_A_CALLS = 0
        destination: Any = declared_output
        sentinel = None
        if case == "canonical-config":
            sentinel = _DestinationSentinel(declared_output)
            destination = sentinel
        result = _expect_pack_error(
            case,
            lambda destination=destination, mutated=mutated: (
                stream_transform_to_npy_chunks(
                    source, destination, _transform_a, **mutated
                )
            ),
        )
        result["transform_calls"] = _TRANSFORM_A_CALLS
        result["destination_fspath_calls"] = (
            None if sentinel is None else sentinel.fspath_calls
        )
        result["tree_byte_identical"] = _tree_identity(declared_output) == declared_before
        if _TRANSFORM_A_CALLS != 0 or not result["tree_byte_identical"]:
            raise PackError(f"declared-plan regression changed output: {case}")
        declared_plan_results.append(result)

    stale_output = fixture_root / "stale-receipt"
    stream_transform_to_npy_chunks(source, stale_output, _transform_a, **kwargs)
    stale_path = stale_output / "chunk-00000" / "receipt.json"
    stale = read_json(stale_path)
    stale.pop("receipt_sha256")
    stale["stream_plan_sha256"] = "e" * 64
    atomic_write_json(stale_path, seal_record(stale), replace=True)
    stale_before = _tree_identity(stale_output)
    _TRANSFORM_A_CALLS = 0
    stale_result = _expect_pack_error(
        "stale-receipt",
        lambda: stream_transform_to_npy_chunks(
            source, stale_output, _transform_a, **kwargs
        ),
    )
    stale_result.update(
        transform_calls=_TRANSFORM_A_CALLS,
        tree_byte_identical=_tree_identity(stale_output) == stale_before,
    )
    if _TRANSFORM_A_CALLS != 0 or not stale_result["tree_byte_identical"]:
        raise PackError("stale receipt regression changed output or called transform")

    mutation_root = fixture_root / "source-mutations"
    mutation_root.mkdir()
    mutation_matrix = [
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
        members[0],
        path=truncated_path.resolve(),
        identity=file_identity(truncated_path),
        sha256=truncated_digest,
    )
    with truncated_path.open("r+b") as handle:
        handle.truncate(truncated_path.stat().st_size - RAW_DTYPE.itemsize)
        handle.flush()
        os.fsync(handle.fileno())
    mutation_matrix.append(
        _expect_pack_error(
            "truncated",
            lambda: verify_raw_map_members(
                [truncated],
                total_rows=truncated.global_stop,
                dimension=source.dimension,
                full_hash=False,
            ),
        )
    )
    mismatch_path = mutation_root / "hash-mismatch.npy"
    shutil.copyfile(members[0].path, mismatch_path)
    mismatched = dataclasses.replace(
        members[0], path=mismatch_path.resolve(), identity=file_identity(mismatch_path)
    )
    with mismatch_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
        handle.flush()
        os.fsync(handle.fileno())
    mutation_matrix.append(
        _expect_pack_error(
            "hash-mismatch",
            lambda: verify_raw_map_members(
                [mismatched],
                total_rows=mismatched.global_stop,
                dimension=source.dimension,
                full_hash=True,
                check_identity=False,
            ),
        )
    )

    observed_execution = resumed["capability_payload"]["plan"]["transform"]
    body = {
        "schema": FIXTURE_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "implementation_release_commit": release_commit,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "transform_spec_receipt_sha256": spec_receipt["receipt_sha256"],
        "observed_transform_execution": observed_execution,
        "canonical_stream_plan": {
            "schema": STREAM_PLAN_SCHEMA,
            "stream_plan_sha256": resumed["stream_plan_sha256"],
            "stream_plan_receipt_sha256": resumed["stream_plan_receipt_sha256"],
            "complete_source_identity_sha256": resumed["capability_payload"][
                "complete_source_identity_sha256"
            ],
        },
        "a_to_a_resume": {
            "byte_identical_to_clean": resume_identity == clean_identity,
            "capability_sha256": resumed["capability_sha256"],
            "clean_capability_sha256": clean["capability_sha256"],
            "created_chunks_after_resume": resumed["created_chunks_this_invocation"],
            "reused_chunks_after_resume": resumed["resumed_chunks_this_invocation"],
            "payload_files": resume_files,
        },
        "a_to_b_same_declared_metadata": {
            "caller_declared_transform_id": _CALLER_TRANSFORM_ID,
            "caller_declared_implementation_sha256": _CALLER_IMPLEMENTATION_LABEL,
            "canonical_config": _TRANSFORM_CONFIG,
            "a_chunks_committed": 1,
            "rejected_before_destination_resolution": (
                destination_sentinel.fspath_calls == 0
            ),
            "destination_fspath_calls": destination_sentinel.fspath_calls,
            "b_transform_calls": _TRANSFORM_B_CALLS,
            "rejected_before_b_call": _TRANSFORM_B_CALLS == 0,
            "rejected_before_chunk_reuse_or_output": (
                _tree_identity(adversarial_output) == adversarial_before_identity
            ),
            "capability_returned": False,
            "second_chunk_absent": not (adversarial_output / "chunk-00001").exists(),
            "whole_tree_byte_identical": (
                _tree_identity(adversarial_output) == adversarial_before_identity
            ),
            "error": adversarial_error,
            "files_before": adversarial_before,
            "files_after": adversarial_after,
        },
        "changed_source_resume": {
            "replacement_sha256": digest,
            "transform_calls": changed_transform_calls,
            "tree_byte_identical": changed_tree_byte_identical,
            "error": changed_error,
        },
        "declared_plan_mutations": declared_plan_results,
        "stale_receipt": stale_result,
        "source_mutation_matrix": mutation_matrix,
        "phase_wall_seconds": time.monotonic() - started,
    }
    if body["torch_imported"] or body["cuda_visible_devices"] != "":
        raise PackError("Round 0013 fixtures did not remain CUDA-hidden and Torch-free")
    receipt = seal_record(body)
    atomic_write_json(receipt_path, receipt, replace=False)
    return receipt


def record_round0013_graph_provenance(
    release_commit: str,
    root: os.PathLike[str] | str = ROUND_ROOT,
    upstream_root: os.PathLike[str] | str = UPSTREAM_ROOT,
) -> dict[str, Any]:
    require_cuda_hidden()
    root_path = _round_root(root)
    diagnostic = _round0012_diagnostic()
    receipt = _record_graph_provenance(
        root_path,
        upstream_root,
        _expected_root=ROUND_ROOT,
        _round_id=ROUND_ID,
        _schema=GRAPH_PROVENANCE_SCHEMA,
        _log_label="round0013",
        _implementation_release_commit=release_commit,
        _diagnostic_dependency=diagnostic,
    )
    if receipt.get("implementation_release_commit") != release_commit:
        raise PackError("graph provenance receipt release binding mismatch")
    return receipt


def full_round0013_read_only_reopen(
    release_commit: str,
    root: os.PathLike[str] | str = ROUND_ROOT,
    upstream_root: os.PathLike[str] | str = UPSTREAM_ROOT,
    *,
    progress_path: os.PathLike[str] | str = ROUND_ROOT / "management" / "progress.json",
    log_path: os.PathLike[str] | str = ROUND_ROOT / "logs" / "full-read-only-reopen.log",
    block_rows: int = DEFAULT_BLOCK_ROWS,
    heavy_io_authorized: bool = False,
) -> dict[str, Any]:
    require_cuda_hidden()
    root_path = _round_root(root)
    diagnostic = _round0012_diagnostic()
    receipt = _full_read_only_reopen(
        root_path,
        upstream_root,
        progress_path=progress_path,
        log_path=log_path,
        block_rows=block_rows,
        heavy_io_authorized=heavy_io_authorized,
        _expected_root=ROUND_ROOT,
        _round_id=ROUND_ID,
        _full_reopen_schema=FULL_REOPEN_SCHEMA,
        _graph_provenance_schema=GRAPH_PROVENANCE_SCHEMA,
        _log_label="round0013",
        _round_label="Round 0013",
        _implementation_release_commit=release_commit,
        _diagnostic_dependency=diagnostic,
    )
    if receipt.get("implementation_release_commit") != release_commit:
        raise PackError("full reopen receipt release binding mismatch")
    return receipt


def _round0013_capability_payload(
    *,
    upstream_manifest: Mapping[str, Any],
    transform_spec: Mapping[str, Any],
    fixture: Mapping[str, Any],
    full_reopen: Mapping[str, Any],
    provenance: Mapping[str, Any],
    release_commit: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(upstream_manifest["capability_payload"])
    payload["round_id"] = ROUND_ID
    payload["implementation_release_commit"] = release_commit
    payload["raw_source"]["receipt_origin"] = (
        "round0010-read-only-upstream; freshly reopened by Round 0013"
    )
    payload["materialized_fp16"]["receipt_origin"] = (
        "round0010-read-only-upstream; freshly reopened by Round 0013"
    )
    payload["graph"]["receipt_origin"] = (
        "round0010-read-only-upstream; freshly reopened by Round 0013"
    )
    payload["graph"]["provenance"] = {
        "receipt_sha256": provenance["receipt_sha256"],
        "status": provenance["status"],
        "diagnostic_only": provenance["diagnostic_only"],
        "claim": provenance["claim"],
        "registered_alignment_sha256": provenance["registered_alignment_sha256"],
    }
    observed = fixture["observed_transform_execution"]
    payload["loader_contract"]["streamed_output"] = {
        "schema": fixture["canonical_stream_plan"]["schema"],
        "persistence": "direct open_memmap chunks",
        "atomic_unit": "fsynced data+receipt directory rename",
        "authentication_boundary": (
            "mechanically authenticate the actual callable's qualified identity, "
            "full defining artifact, normalized code bytes, exact release commit, "
            "and canonical config before resolving any destination path"
        ),
        "observed_transform_execution": observed,
        "observed_transform_execution_sha256": observed[
            "observed_execution_sha256"
        ],
        "transform_execution_spec_receipt_sha256": transform_spec["receipt_sha256"],
        "stream_plan_sha256": fixture["canonical_stream_plan"][
            "stream_plan_sha256"
        ],
        "stream_plan_receipt_sha256": fixture["canonical_stream_plan"][
            "stream_plan_receipt_sha256"
        ],
        "a_to_a_byte_identical_resume": fixture["a_to_a_resume"][
            "byte_identical_to_clean"
        ],
        "a_to_b_rejected_before_destination": fixture[
            "a_to_b_same_declared_metadata"
        ]["rejected_before_destination_resolution"],
        "a_to_b_rejected_before_callback": fixture[
            "a_to_b_same_declared_metadata"
        ]["rejected_before_b_call"],
        "a_to_b_whole_tree_byte_identical": fixture[
            "a_to_b_same_declared_metadata"
        ]["whole_tree_byte_identical"],
    }
    payload["fixture_receipt_sha256"] = fixture["receipt_sha256"]
    payload["qualification"] = {
        "schema": "round0013-30m-qualification-v1",
        "full_reopen_receipt_sha256": full_reopen["receipt_sha256"],
        "graph_provenance_receipt_sha256": provenance["receipt_sha256"],
        "transform_spec_receipt_sha256": transform_spec["receipt_sha256"],
        "rows_compared_source_fp32_to_persisted_fp16": full_reopen[
            "source_to_fp16_conversion"
        ]["rows"],
        "conversion_payload_sha256": full_reopen["source_to_fp16_conversion"][
            "persisted_fp16_payload_sha256"
        ],
        "complete_reopen": full_reopen["complete_reopen"],
        "implementation_release_commit": release_commit,
    }
    payload["lineage"] = {
        "supersedes_round": "0012",
        "round0012_diagnostic_manifest": _round0012_diagnostic(),
        "upstream_bytes_reused_read_only": True,
        "upstream_artifacts_overwritten": False,
        "prior_receipts_or_booleans_copied_as_proof": False,
    }
    return payload


def assemble_round0013_capability(
    release_commit: str,
    root: os.PathLike[str] | str = ROUND_ROOT,
    upstream_root: os.PathLike[str] | str = UPSTREAM_ROOT,
) -> dict[str, Any]:
    require_cuda_hidden()
    root_path = _round_root(root)
    upstream = Path(upstream_root).resolve()
    if upstream != UPSTREAM_ROOT:
        raise PackError(f"Round 0013 may reopen only {UPSTREAM_ROOT}")
    destination = root_path / f"{PACK_SCHEMA}.json"
    if destination.exists():
        manifest = read_json(destination)
        verify_sealed_record(manifest, field="manifest_receipt_sha256")
        return manifest
    transform_spec = _load_required_receipt(
        root_path / "receipts" / "transform-execution-spec.json",
        TRANSFORM_SPEC_RECEIPT_SCHEMA,
    )
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
        transform_spec["release"]["head"] != release_commit
        or fixture.get("implementation_release_commit") != release_commit
        or full_reopen.get("implementation_release_commit") != release_commit
        or provenance.get("implementation_release_commit") != release_commit
        or full_reopen.get("complete_reopen") is not True
        or full_reopen.get("source_to_fp16_conversion", {}).get("rows") != 30_000_000
        or full_reopen.get("source_to_fp16_conversion", {}).get("byte_exact") is not True
        or fixture["a_to_b_same_declared_metadata"][
            "rejected_before_destination_resolution"
        ]
        is not True
        or fixture["a_to_b_same_declared_metadata"]["b_transform_calls"] != 0
        or fixture["a_to_b_same_declared_metadata"][
            "whole_tree_byte_identical"
        ]
        is not True
    ):
        raise PackError("Round 0013 receipts do not qualify capability assembly")
    closure = _load_upstream_closure(upstream)
    payload = _round0013_capability_payload(
        upstream_manifest=closure["manifest"],
        transform_spec=transform_spec,
        fixture=fixture,
        full_reopen=full_reopen,
        provenance=provenance,
        release_commit=release_commit,
    )
    capability_hash = canonical_sha256(payload)
    body = {
        "schema": PACK_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "capability_name": PACK_SCHEMA,
        "implementation_release_commit": release_commit,
        "capability_payload": payload,
        "capability_hash_sha256": capability_hash,
        "own_receipt_hashes": {
            "transform_execution_spec": transform_spec["receipt_sha256"],
            "fixtures": fixture["receipt_sha256"],
            "full_read_only_reopen": full_reopen["receipt_sha256"],
            "graph_provenance": provenance["receipt_sha256"],
        },
        "upstream_read_only_receipt_hashes": closure["manifest"]["receipt_hashes"],
        "upstream_manifest": closure["manifest_file"],
        "round0012_diagnostic_dependency": _round0012_diagnostic(),
        "claims": {
            "cpu_only_input_pack": True,
            "observed_transform_authenticated_resume": True,
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
    return manifest


def reopen_round0013_capability(
    release_commit: str,
    release_root: os.PathLike[str] | str,
    root: os.PathLike[str] | str = ROUND_ROOT,
    upstream_root: os.PathLike[str] | str = UPSTREAM_ROOT,
) -> dict[str, Any]:
    """Reproduce the complete capability hash in a separate read-only process."""

    require_cuda_hidden()
    root_path = _round_root(root)
    source_root = _release_root(release_root)
    upstream = Path(upstream_root).resolve()
    if upstream != UPSTREAM_ROOT:
        raise PackError(f"Round 0013 may reopen only {UPSTREAM_ROOT}")
    destination = root_path / "receipts" / "capability-reopen.json"
    if destination.exists():
        receipt = read_json(destination)
        verify_sealed_record(receipt)
        if receipt.get("schema") != CAPABILITY_REOPEN_SCHEMA:
            raise PackError("existing capability reopen receipt has the wrong schema")
        return receipt
    release = _verify_release_source(source_root, release_commit)
    manifest_path = root_path / f"{PACK_SCHEMA}.json"
    manifest = read_json(manifest_path)
    verify_sealed_record(manifest, field="manifest_receipt_sha256")
    payload = manifest.get("capability_payload")
    if (
        manifest.get("round_id") != ROUND_ID
        or manifest.get("implementation_release_commit") != release_commit
        or not isinstance(payload, dict)
        or canonical_sha256(payload) != manifest.get("capability_hash_sha256")
    ):
        raise PackError("Round 0013 capability hash/release does not reproduce")
    transform_spec = _load_required_receipt(
        root_path / "receipts" / "transform-execution-spec.json",
        TRANSFORM_SPEC_RECEIPT_SCHEMA,
    )
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
        "transform_execution_spec": transform_spec["receipt_sha256"],
        "fixtures": fixture["receipt_sha256"],
        "full_read_only_reopen": full_reopen["receipt_sha256"],
        "graph_provenance": provenance["receipt_sha256"],
    }
    if manifest.get("own_receipt_hashes") != expected_own:
        raise PackError("Round 0013 own-receipt closure does not reproduce")
    artifact_identities = []
    for artifact in full_reopen["artifact_closure"]:
        path = Path(artifact["path"])
        observed = file_identity(path)
        if observed != artifact["identity_after"]:
            raise PackError(f"qualified artifact identity changed after full reopen: {path}")
        artifact_identities.append(
            {"role": artifact["role"], "path": str(path), "identity": observed}
        )
    upstream_files = []
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
        or manifest.get("round0012_diagnostic_dependency")
        != _round0012_diagnostic()
    ):
        raise PackError("final capability upstream/diagnostic closure changed")
    manifest_file = _stable_file(manifest_path)
    body = {
        "schema": CAPABILITY_REOPEN_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "implementation_release_commit": release_commit,
        "release": release,
        "manifest_file": manifest_file,
        "manifest_receipt_sha256": manifest["manifest_receipt_sha256"],
        "capability_hash_sha256": manifest["capability_hash_sha256"],
        "own_receipt_hashes": expected_own,
        "upstream_receipt_hashes": closure["manifest"]["receipt_hashes"],
        "artifact_identities_reopened": artifact_identities,
        "upstream_receipt_files_reopened": upstream_files,
        "round0012_diagnostic_reopened": _round0012_diagnostic(),
        "complete_content_scan_receipt_reopened": True,
        "capability_hash_reproduced": True,
        "torch_imported": "torch" in sys.modules,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if body["torch_imported"] or body["cuda_visible_devices"] != "":
        raise PackError("capability reopen was not CUDA-hidden and Torch-free")
    receipt = seal_record(body)
    atomic_write_json(destination, receipt, replace=False)
    return receipt
