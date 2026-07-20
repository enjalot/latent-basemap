"""Round 0011: CPU-only, content-bound Jina document inventory.

This module deliberately has no torch dependency.  It discovers the exact
prompt-free Jina source manifests registered by Round 0009, reopens and hashes
their text Parquets directly, profiles the pinned tokenizer from the local HF
snapshot, builds four deterministic one-million-row unit manifests made from
25,000-row chunks, exercises fail-closed resume fixtures, and assembles the
``jina-document-inventory-v1`` capability.

All writes are new, atomic files below the Round 0011 runtime root.  Source
documents and prompt-free embeddings are only opened read-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import resource
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROUND_ID = "0011"
CAPABILITY_NAME = "jina-document-inventory-v1"
INVENTORY_SCHEMA = "round0011.source_inventory.v1"
PROFILE_SCHEMA = "round0011.tokenizer_profile.v1"
UNITS_SCHEMA = "round0011.unit_manifests.v1"
FIXTURE_SCHEMA = "round0011.fixture_receipt.v1"
CAPABILITY_SCHEMA = "jina-document-inventory-v1"
REOPEN_SCHEMA = "round0011.capability_reopen.v1"

EXPECTED_TOTAL_ROWS = 49_126_376
EXPECTED_ENGLISH_ROWS = 9_126_376
EXPECTED_FINEWEB2_ROWS = 2_000_000
EXPECTED_DISCOVERY_RECEIPT = (
    "30e616115ef1e9c799610306b9b6bdd2429d785c85be4b759dfd536055fdd8df"
)

MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"
MODEL_REVISION = "ac5d898c8d382b17167c33e5c8af644a3519b47d"
PROMPT_TEXT = "Document: "
PROMPT_HEX = "446f63756d656e743a20"
MAX_SEQUENCE_LENGTH = 512
EMBED_DIM = 768
OUTPUT_DTYPE = "float16"
OUTPUT_BYTES_PER_ROW = EMBED_DIM * 2
ATOMIC_CHUNK_ROWS = 25_000
UNIT_ROWS = 1_000_000
PROFILE_HASH_SAMPLE_ROWS = 128

DEFAULT_EMBEDDING_ROOT = Path("/data/embeddings")
DEFAULT_OUTPUT_ROOT = Path("/data/latent-basemap/runs/round-0011")
DEFAULT_HF_ROOT = Path("/data/hf/hub")

ENGLISH_SPECS = (
    (
        "fineweb-edu",
        "eng_Latn",
        "fineweb-edu-sample-10BT-chunked-500-jina-v5-nano",
    ),
    (
        "redpajama-v2",
        "eng_Latn",
        "RedPajama-Data-V2-sample-10B-chunked-500-jina-v5-nano",
    ),
    (
        "pile-uncopyrighted",
        "eng_Latn",
        "pile-uncopyrighted-chunked-500-jina-v5-nano",
    ),
)

FINEWEB2_LANGUAGES = (
    "arb_Arab",
    "ces_Latn",
    "cmn_Hani",
    "deu_Latn",
    "ell_Grek",
    "fra_Latn",
    "hin_Deva",
    "ind_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "kor_Hang",
    "nld_Latn",
    "pol_Latn",
    "por_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "swe_Latn",
    "tha_Thai",
    "tur_Latn",
    "vie_Latn",
)


class ContractError(RuntimeError):
    """A fail-closed Round 0011 contract violation with a stable reason code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclasses.dataclass(frozen=True)
class CorpusSpec:
    corpus_id: str
    language: str
    family: str
    manifest_dir: str


def corpus_specs() -> tuple[CorpusSpec, ...]:
    english = tuple(
        CorpusSpec(corpus_id, language, "english", manifest_dir)
        for corpus_id, language, manifest_dir in ENGLISH_SPECS
    )
    multilingual = tuple(
        CorpusSpec(
            f"fineweb2-{language}",
            language,
            "fineweb2",
            f"fineweb2-{language}-chunked-500-jina-v5-nano",
        )
        for language in FINEWEB2_LANGUAGES
    )
    return english + multilingual


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def deterministic_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)


def file_sha256(path: Path, *, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=block_bytes) as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_new_bytes(path: Path, payload: bytes) -> str:
    """Publish a complete file without overwriting an existing artifact.

    An identical existing file makes a resumed command idempotent.  Different
    bytes at the same path are an output collision and fail closed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ContractError("output_collision", f"non-regular output {path}")
        if path.read_bytes() == payload:
            return "existing-identical"
        raise ContractError("output_collision", f"different bytes already exist at {path}")

    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(tmp, path)
        _fsync_directory(path.parent)
        return "created"
    except FileExistsError:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return "existing-identical"
        raise ContractError("output_collision", f"concurrent output at {path}")
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_new_json(path: Path, value: Any) -> str:
    return atomic_write_new_bytes(path, canonical_json_bytes(value))


class ProgressLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        *,
        phase: str,
        event: str,
        current: int | None = None,
        total: int | None = None,
        detail: str | None = None,
        artifact: str | None = None,
    ) -> None:
        record = {
            "utc": utc_now(),
            "round_id": ROUND_ID,
            "pid": os.getpid(),
            "phase": phase,
            "event": event,
            "current": current,
            "total": total,
            "detail": detail,
            "artifact": artifact,
        }
        line = canonical_json_bytes(record)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)
        print(json.dumps(record, sort_keys=True), flush=True)


def assert_cpu_only_process() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ContractError(
            "cuda_not_hidden", "CUDA_VISIBLE_DEVICES must be present and empty"
        )
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise ContractError("torch_loaded", "torch is loaded in the CPU-only process")
    maps = Path("/proc/self/maps")
    if maps.exists():
        mapped = maps.read_text(errors="replace").lower()
        forbidden = ("libcuda.so", "libcudart", "libtorch_cuda", "libnvidia-ml")
        hit = next((name for name in forbidden if name in mapped), None)
        if hit:
            raise ContractError("cuda_library_loaded", f"mapped forbidden library {hit}")


def _stat_identity(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ContractError("unsupported_source_kind", f"source is not a regular file: {path}")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "nlink": int(info.st_nlink),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def stable_file_identity(path: Path) -> dict[str, Any]:
    before = _stat_identity(path)
    digest = file_sha256(path)
    after = _stat_identity(path)
    if before != after:
        raise ContractError("source_mutated", f"filesystem identity changed while hashing {path}")
    return {"path": str(path), "sha256": digest, "filesystem": after}


def _runtime_start() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io = _proc_io()
    return {
        "wall": time.perf_counter(),
        "cpu": time.process_time(),
        "max_rss_kib": int(usage.ru_maxrss),
        "io": io,
    }


def _proc_io() -> dict[str, int]:
    path = Path("/proc/self/io")
    if not path.exists():
        return {}
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, value = line.split(":", 1)
        values[key.strip()] = int(value.strip())
    return values


def _runtime_finish(start: Mapping[str, Any]) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io = _proc_io()
    before_io = start.get("io", {})
    return {
        "wall_seconds": time.perf_counter() - float(start["wall"]),
        "cpu_seconds": time.process_time() - float(start["cpu"]),
        "max_rss_kib": int(usage.ru_maxrss),
        "max_rss_growth_kib": max(0, int(usage.ru_maxrss) - int(start["max_rss_kib"])),
        "io_delta": {
            key: int(value) - int(before_io.get(key, 0)) for key, value in io.items()
        },
    }


def _eligible_discovery_manifests(embedding_root: Path) -> list[Path]:
    paths = [embedding_root / directory / "manifest.json" for _, _, directory in ENGLISH_SPECS]
    paths.extend(
        embedding_root
        / f"fineweb2-{language}-chunked-500-jina-v5-nano"
        / "manifest.json"
        for language in FINEWEB2_LANGUAGES
    )
    return sorted(paths, key=lambda path: str(path))


def _verify_discovery_universe(embedding_root: Path) -> list[Path]:
    expected = _eligible_discovery_manifests(embedding_root)
    discovered = []
    for path in embedding_root.glob("fineweb2-*-chunked-500-jina-v5-nano/manifest.json"):
        discovered.append(path)
    for _, _, directory in ENGLISH_SPECS:
        path = embedding_root / directory / "manifest.json"
        if path.exists():
            discovered.append(path)
    discovered = sorted(set(discovered), key=lambda path: str(path))
    if discovered != expected:
        missing = sorted(str(path) for path in set(expected) - set(discovered))
        extra = sorted(str(path) for path in set(discovered) - set(expected))
        raise ContractError("source_universe_mismatch", f"missing={missing}; extra={extra}")
    if any(not path.is_file() or path.is_symlink() for path in expected):
        raise ContractError("source_manifest_missing", "one or more discovery manifests unavailable")
    return expected


def _discovery_receipt(manifest_identities: Sequence[Mapping[str, Any]]) -> str:
    # The registered receipt was produced by path-sorting the NUL-delimited
    # manifest list, then feeding that order to sha256sum.  Sorting completed
    # lines would sort by digest and produce a different (unregistered) value.
    lines = [
        f"{identity['sha256']}  {identity['path']}\n"
        for identity in sorted(manifest_identities, key=lambda value: value["path"])
    ]
    return sha256_bytes("".join(lines).encode("utf-8"))


def _schema_signature(parquet_file: Any) -> list[dict[str, str]]:
    schema = parquet_file.schema_arrow
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def _validate_document_schema(signature: Sequence[Mapping[str, str]], path: Path) -> None:
    fields = {field["name"]: field["type"] for field in signature}
    required = {
        "chunk_text": {"string", "large_string"},
        "chunk_token_count": {"int64"},
        "chunk_index": {"int64"},
    }
    for name, allowed in required.items():
        if fields.get(name) not in allowed:
            raise ContractError(
                "unsupported_document_schema",
                f"{path}: {name}={fields.get(name)!r}, expected one of {sorted(allowed)}",
            )


def document_id(corpus_id: str, corpus_source_sha256: str, source_row: int) -> str:
    if source_row < 0:
        raise ContractError("document_row_out_of_bounds", f"negative row {source_row}")
    payload = (
        b"jina-document-v1\0"
        + corpus_id.encode("utf-8")
        + b"\0"
        + corpus_source_sha256.encode("ascii")
        + b"\0"
        + str(source_row).encode("ascii")
    )
    return sha256_bytes(payload)


def validate_range_cover(ranges: Sequence[Mapping[str, Any]], total_rows: int) -> None:
    cursor = 0
    member_names: set[str] = set()
    identities: set[tuple[int, int]] = set()
    for index, member in enumerate(ranges):
        start = int(member["corpus_row_start"])
        end = int(member["corpus_row_end"])
        name = str(member["relative_path"])
        if name in member_names:
            raise ContractError("duplicate_member", f"duplicate member {name}")
        member_names.add(name)
        fs = member["filesystem"]
        inode_identity = (int(fs["device"]), int(fs["inode"]))
        if inode_identity in identities:
            raise ContractError("duplicate_member_inode", f"duplicate inode for {name}")
        identities.add(inode_identity)
        if start != cursor:
            code = "range_overlap" if start < cursor else "range_gap"
            raise ContractError(code, f"member {index} starts {start}, expected {cursor}")
        if end <= start or end - start != int(member["row_count"]):
            raise ContractError("invalid_range", f"invalid range [{start},{end}) for {name}")
        cursor = end
    if cursor != total_rows:
        raise ContractError("range_gap", f"covered {cursor} rows, expected {total_rows}")


def _corpus_identity_payload(corpus: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "corpus_id": corpus["corpus_id"],
        "language": corpus["language"],
        "family": corpus["family"],
        "members": [
            {
                "relative_path": member["relative_path"],
                "sha256": member["sha256"],
                "size_bytes": member["filesystem"]["size_bytes"],
                "row_count": member["row_count"],
                "row_group_counts": member["row_group_counts"],
                "schema": member["schema"],
            }
            for member in corpus["members"]
        ],
    }


def _inventory_identity_payload(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": INVENTORY_SCHEMA,
        "round_id": ROUND_ID,
        "discovery_manifest_receipt_sha256": inventory["discovery"][
            "manifest_receipt_sha256"
        ],
        "discovery_manifests": inventory["discovery"]["manifests"],
        "expected": inventory["expected"],
        "observed": inventory["observed"],
        "corpus_order": inventory["corpus_order"],
        "corpora": inventory["corpora"],
        "document_identity": inventory["document_identity"],
    }


def build_inventory(
    *, embedding_root: Path, progress: ProgressLog | None = None
) -> dict[str, Any]:
    """Build the complete deterministic direct-source inventory in memory."""

    assert_cpu_only_process()
    import pyarrow.parquet as pq

    started = _runtime_start()
    manifest_paths = _verify_discovery_universe(embedding_root)
    manifest_identities: list[dict[str, Any]] = []
    for path in manifest_paths:
        manifest_identities.append(stable_file_identity(path))
    receipt = _discovery_receipt(manifest_identities)
    if receipt != EXPECTED_DISCOVERY_RECEIPT:
        raise ContractError(
            "discovery_receipt_mismatch",
            f"observed {receipt}, expected {EXPECTED_DISCOVERY_RECEIPT}",
        )

    identities_by_path = {entry["path"]: entry for entry in manifest_identities}
    specs_by_directory = {spec.manifest_dir: spec for spec in corpus_specs()}
    corpus_records: list[dict[str, Any]] = []
    program_cursor = 0
    source_member_count = 0

    if progress:
        progress.emit(
            phase="source-inventory",
            event="discovery-verified",
            current=0,
            total=23,
            detail=f"23 manifest receipt {receipt}",
        )

    for corpus_index, path in enumerate(manifest_paths):
        directory = path.parent.name
        spec = specs_by_directory.get(directory)
        if spec is None:
            raise ContractError("unknown_corpus", f"unregistered manifest directory {directory}")
        manifest = read_json(path)
        required = {
            "model",
            "slug",
            "max_seq_length",
            "source",
            "n_shards",
            "shards_completed",
            "total_chunks_so_far",
            "shards",
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise ContractError("manifest_schema", f"{path}: missing {missing}")
        if manifest["model"] != MODEL_ID or manifest["slug"] != "jina-v5-nano":
            raise ContractError("manifest_model_mismatch", str(path))
        if int(manifest["max_seq_length"]) != MAX_SEQUENCE_LENGTH:
            raise ContractError("manifest_tokenizer_mismatch", str(path))
        shards = list(manifest["shards"])
        if (
            int(manifest["n_shards"]) != len(shards)
            or int(manifest["shards_completed"]) != len(shards)
            or not shards
        ):
            raise ContractError("manifest_incomplete", str(path))
        declared_order = [str(shard["shard"]) for shard in shards]
        if declared_order != sorted(declared_order) or len(set(declared_order)) != len(
            declared_order
        ):
            raise ContractError("source_reordered", f"noncanonical shard order in {path}")

        source_root = Path(str(manifest["source"]))
        if not source_root.is_absolute() or source_root.is_symlink() or not source_root.is_dir():
            raise ContractError("source_root_invalid", str(source_root))
        root_resolved = source_root.resolve()
        root_stat = source_root.stat()
        members: list[dict[str, Any]] = []
        corpus_cursor = 0
        for shard_index, shard in enumerate(shards):
            name = str(shard["shard"])
            if Path(name).name != name:
                raise ContractError("source_path_escape", name)
            source_path = source_root / name
            try:
                source_path.resolve().relative_to(root_resolved)
            except ValueError as error:
                raise ContractError("source_path_escape", str(source_path)) from error

            file_identity = stable_file_identity(source_path)
            parquet = pq.ParquetFile(source_path)
            schema = _schema_signature(parquet)
            _validate_document_schema(schema, source_path)
            row_count = int(parquet.metadata.num_rows)
            row_group_counts = [
                int(parquet.metadata.row_group(i).num_rows)
                for i in range(parquet.metadata.num_row_groups)
            ]
            if sum(row_group_counts) != row_count or row_count != int(shard["n"]):
                raise ContractError(
                    "source_row_count_drift",
                    f"{source_path}: parquet={row_count}, manifest={shard['n']}",
                )
            members.append(
                {
                    "member_index": shard_index,
                    "relative_path": name,
                    "absolute_path": str(source_path),
                    "sha256": file_identity["sha256"],
                    "filesystem": file_identity["filesystem"],
                    "row_count": row_count,
                    "row_group_counts": row_group_counts,
                    "schema": schema,
                    "corpus_row_start": corpus_cursor,
                    "corpus_row_end": corpus_cursor + row_count,
                }
            )
            corpus_cursor += row_count
            source_member_count += 1
            if progress:
                progress.emit(
                    phase="source-inventory",
                    event="source-hashed",
                    current=source_member_count,
                    total=59,
                    detail=f"{spec.corpus_id}:{name} {row_count} rows",
                )

        if corpus_cursor != int(manifest["total_chunks_so_far"]):
            raise ContractError(
                "corpus_total_mismatch",
                f"{spec.corpus_id}: {corpus_cursor} != {manifest['total_chunks_so_far']}",
            )
        corpus: dict[str, Any] = {
            "corpus_index": corpus_index,
            "corpus_id": spec.corpus_id,
            "language": spec.language,
            "family": spec.family,
            "source_root": str(source_root),
            "source_root_filesystem": {
                "device": int(root_stat.st_dev),
                "inode": int(root_stat.st_ino),
                "mode": int(root_stat.st_mode),
                "mtime_ns": int(root_stat.st_mtime_ns),
                "ctime_ns": int(root_stat.st_ctime_ns),
            },
            "discovery_manifest": identities_by_path[str(path)],
            "row_count": corpus_cursor,
            "program_global_row_start": program_cursor,
            "program_global_row_end": program_cursor + corpus_cursor,
            "members": members,
        }
        validate_range_cover(members, corpus_cursor)
        corpus_sha = deterministic_hash(_corpus_identity_payload(corpus))
        corpus["corpus_source_sha256"] = corpus_sha
        corpus["document_identity_witnesses"] = {
            "first": document_id(spec.corpus_id, corpus_sha, 0),
            "last": document_id(spec.corpus_id, corpus_sha, corpus_cursor - 1),
        }
        corpus_records.append(corpus)
        program_cursor += corpus_cursor
        if progress:
            progress.emit(
                phase="source-inventory",
                event="corpus-complete",
                current=corpus_index + 1,
                total=23,
                detail=f"{spec.corpus_id}: {corpus_cursor} rows, source={corpus_sha}",
            )

    if source_member_count != 59:
        raise ContractError(
            "source_member_count_mismatch", f"observed {source_member_count}, expected 59"
        )
    corpus_ids = [record["corpus_id"] for record in corpus_records]
    if len(corpus_ids) != 23 or len(set(corpus_ids)) != 23:
        raise ContractError("duplicate_corpus", f"corpus ids={corpus_ids}")
    english_rows = sum(
        record["row_count"] for record in corpus_records if record["family"] == "english"
    )
    fineweb2 = [record for record in corpus_records if record["family"] == "fineweb2"]
    if english_rows != EXPECTED_ENGLISH_ROWS:
        raise ContractError(
            "english_row_mismatch", f"observed {english_rows}, expected {EXPECTED_ENGLISH_ROWS}"
        )
    if len(fineweb2) != 20 or any(
        record["row_count"] != EXPECTED_FINEWEB2_ROWS for record in fineweb2
    ):
        raise ContractError("fineweb2_row_mismatch", "expected 20 corpora of 2,000,000 rows")
    if program_cursor != EXPECTED_TOTAL_ROWS:
        raise ContractError(
            "total_row_mismatch", f"observed {program_cursor}, expected {EXPECTED_TOTAL_ROWS}"
        )
    witnesses = [
        witness
        for corpus in corpus_records
        for witness in corpus["document_identity_witnesses"].values()
    ]
    validate_unique_document_ids(witnesses)

    inventory: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "round_id": ROUND_ID,
        "capability_component": "direct-source-inventory",
        "discovery": {
            "authority": "registered prompt-free manifests are discovery inputs only",
            "manifest_receipt_algorithm": (
                "sha256(sorted absolute-path sha256sum lines, two spaces before path)"
            ),
            "manifest_receipt_sha256": receipt,
            "manifests": manifest_identities,
        },
        "expected": {
            "corpora": 23,
            "english_corpora": 3,
            "fineweb2_corpora": 20,
            "english_rows": EXPECTED_ENGLISH_ROWS,
            "fineweb2_rows_per_corpus": EXPECTED_FINEWEB2_ROWS,
            "total_rows": EXPECTED_TOTAL_ROWS,
            "source_members": 59,
        },
        "observed": {
            "corpora": len(corpus_records),
            "english_corpora": 3,
            "fineweb2_corpora": len(fineweb2),
            "english_rows": english_rows,
            "total_rows": program_cursor,
            "source_members": source_member_count,
            "source_bytes": sum(
                member["filesystem"]["size_bytes"]
                for corpus in corpus_records
                for member in corpus["members"]
            ),
        },
        "corpus_order": corpus_ids,
        "corpora": corpus_records,
        "document_identity": {
            "schema": "jina-document-v1",
            "formula": (
                "sha256(b'jina-document-v1\\0' + utf8(corpus_id) + b'\\0' + "
                "ascii(corpus_source_sha256) + b'\\0' + ascii(source_row))"
            ),
            "uniqueness_argument": (
                "23 unique corpus IDs, content-bound corpus identities, and complete "
                "nonoverlapping integer source-row ranges"
            ),
            "range_coverage_verified": True,
        },
    }
    inventory["source_inventory_sha256"] = deterministic_hash(
        _inventory_identity_payload(inventory)
    )
    inventory["runtime_observation"] = _runtime_finish(started)
    assert_cpu_only_process()
    return inventory


def validate_unique_document_ids(ids: Iterable[str]) -> None:
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            raise ContractError("duplicate_document_id", value)
        seen.add(value)


def validate_inventory_artifact(inventory: Mapping[str, Any]) -> None:
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ContractError("inventory_schema", str(inventory.get("schema")))
    observed = deterministic_hash(_inventory_identity_payload(inventory))
    if observed != inventory.get("source_inventory_sha256"):
        raise ContractError("inventory_hash_mismatch", f"observed {observed}")
    if inventory["observed"]["total_rows"] != EXPECTED_TOTAL_ROWS:
        raise ContractError("total_row_mismatch", str(inventory["observed"]["total_rows"]))
    if len(inventory["corpora"]) != 23:
        raise ContractError("source_universe_mismatch", "inventory does not contain 23 corpora")
    for corpus in inventory["corpora"]:
        validate_range_cover(corpus["members"], int(corpus["row_count"]))
        observed_corpus = deterministic_hash(_corpus_identity_payload(corpus))
        if observed_corpus != corpus["corpus_source_sha256"]:
            raise ContractError("corpus_hash_mismatch", corpus["corpus_id"])


def _nearest_rank(values: Any, quantile: float) -> int:
    import numpy as np

    array = np.asarray(values)
    if array.size == 0:
        raise ContractError("empty_corpus", "cannot select a quantile from zero rows")
    rank = min(array.size - 1, max(0, math.ceil(quantile * array.size) - 1))
    threshold = np.partition(array, rank)[rank]
    return int(np.flatnonzero(array == threshold)[0])


def _quantile_value(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ContractError("empty_profile", "no successful token counts")
    ordered = sorted(int(value) for value in values)
    rank = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[rank]


def _load_source_token_counts(corpus: Mapping[str, Any]) -> Any:
    import numpy as np
    import pyarrow.parquet as pq

    counts = np.empty(int(corpus["row_count"]), dtype=np.int64)
    cursor = 0
    for member in corpus["members"]:
        parquet = pq.ParquetFile(member["absolute_path"])
        for batch in parquet.iter_batches(columns=["chunk_token_count"], batch_size=131_072):
            values = batch.column(0).to_numpy(zero_copy_only=False)
            counts[cursor : cursor + len(values)] = values
            cursor += len(values)
    if cursor != len(counts):
        raise ContractError(
            "profile_row_count_drift", f"{corpus['corpus_id']}: read {cursor}/{len(counts)}"
        )
    if bool((counts < 0).any()):
        raise ContractError("invalid_source_token_count", corpus["corpus_id"])
    return counts


def profile_selection(corpus: Mapping[str, Any], source_counts: Any) -> list[dict[str, Any]]:
    import numpy as np

    seed_material = sha256_bytes(
        (
            "round0011-profile-v1\0"
            + corpus["corpus_id"]
            + "\0"
            + corpus["corpus_source_sha256"]
        ).encode("utf-8")
    )
    seed = int(seed_material[:16], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    random_rows = sorted(
        int(value)
        for value in rng.choice(
            len(source_counts),
            size=min(PROFILE_HASH_SAMPLE_ROWS, len(source_counts)),
            replace=False,
        )
    )
    reasons: dict[int, set[str]] = {row: {"content-bound-hash-sample"} for row in random_rows}
    strata = (("median", 0.50), ("p95", 0.95), ("p99", 0.99))
    for name, quantile in strata:
        row = _nearest_rank(source_counts, quantile)
        reasons.setdefault(row, set()).add(name)
    max_value = int(source_counts.max())
    max_row = int(np.flatnonzero(source_counts == max_value)[0])
    reasons.setdefault(max_row, set()).add("maximum")
    return [
        {
            "source_row": row,
            "selection_reasons": sorted(values),
            "source_chunk_token_count": int(source_counts[row]),
        }
        for row, values in sorted(reasons.items())
    ]


def _member_for_row(corpus: Mapping[str, Any], source_row: int) -> tuple[Mapping[str, Any], int]:
    if source_row < 0 or source_row >= int(corpus["row_count"]):
        raise ContractError("document_row_out_of_bounds", f"{corpus['corpus_id']}:{source_row}")
    for member in corpus["members"]:
        if int(member["corpus_row_start"]) <= source_row < int(member["corpus_row_end"]):
            return member, source_row - int(member["corpus_row_start"])
    raise ContractError("range_gap", f"no source member for {corpus['corpus_id']}:{source_row}")


def _fetch_selected_texts(
    corpus: Mapping[str, Any], selections: Sequence[Mapping[str, Any]]
) -> dict[int, str]:
    import pyarrow.parquet as pq

    by_member: dict[str, list[tuple[int, int]]] = {}
    for selection in selections:
        source_row = int(selection["source_row"])
        member, local_row = _member_for_row(corpus, source_row)
        by_member.setdefault(member["absolute_path"], []).append((local_row, source_row))

    texts: dict[int, str] = {}
    for path, requested in by_member.items():
        requested.sort()
        positions = {local: source for local, source in requested}
        parquet = pq.ParquetFile(path)
        cursor = 0
        for batch in parquet.iter_batches(columns=["chunk_text"], batch_size=65_536):
            end = cursor + batch.num_rows
            local_rows = [local for local in positions if cursor <= local < end]
            if local_rows:
                column = batch.column(0)
                for local in sorted(local_rows):
                    value = column[local - cursor].as_py()
                    if not isinstance(value, str):
                        raise ContractError(
                            "invalid_document_text", f"{path}:{local} is {type(value).__name__}"
                        )
                    texts[positions[local]] = value
            cursor = end
        expected_rows = {source for _, source in requested}
        if not expected_rows.issubset(texts):
            missing = sorted(expected_rows - set(texts))
            raise ContractError("profile_row_missing", f"{path}: {missing[:8]}")
    return texts


def _tokenizer_snapshot(hf_root: Path) -> Path:
    model_root = hf_root / "models--jinaai--jina-embeddings-v5-text-nano-retrieval"
    ref_path = model_root / "refs" / "main"
    if not ref_path.is_file() or ref_path.is_symlink():
        raise ContractError("tokenizer_assets_missing", str(ref_path))
    revision = ref_path.read_text().strip()
    if revision != MODEL_REVISION:
        raise ContractError(
            "tokenizer_revision_drift", f"local ref {revision}, expected {MODEL_REVISION}"
        )
    snapshot = model_root / "snapshots" / MODEL_REVISION
    if not snapshot.is_dir():
        raise ContractError("tokenizer_assets_missing", str(snapshot))
    return snapshot


def tokenizer_identity(hf_root: Path) -> dict[str, Any]:
    snapshot = _tokenizer_snapshot(hf_root)
    asset_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "config_sentence_transformers.json",
        "config.json",
        "modules.json",
        "1_Pooling/config.json",
        "configuration_eurobert.py",
        "modeling_eurobert.py",
    )
    assets: list[dict[str, Any]] = []
    for name in asset_names:
        path = snapshot / name
        if not path.exists():
            raise ContractError("tokenizer_assets_missing", str(path))
        resolved = path.resolve(strict=True)
        identity = stable_file_identity(resolved)
        identity["snapshot_relative_path"] = name
        identity["snapshot_path"] = str(path)
        assets.append(identity)
    sentence_config = read_json(snapshot / "config_sentence_transformers.json")
    if sentence_config.get("prompts", {}).get("document") != PROMPT_TEXT:
        raise ContractError("prompt_drift", "snapshot native document prompt changed")
    pooling = read_json(snapshot / "1_Pooling" / "config.json")
    if not pooling.get("pooling_mode_lasttoken") or not pooling.get("include_prompt"):
        raise ContractError("pooling_drift", "expected last-token pooling with include_prompt")
    identity: dict[str, Any] = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot": str(snapshot),
        "tokenizer_backend": "tokenizers.Tokenizer.from_file",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "prompt_text": PROMPT_TEXT,
        "prompt_utf8_hex": PROMPT_HEX,
        "pooling": "lasttoken",
        "include_prompt": True,
        "normalization": True,
        "compute_dtype_for_later_gpu_round": "float32",
        "persisted_dtype_for_later_gpu_round": OUTPUT_DTYPE,
        "assets": assets,
    }
    identity["tokenizer_identity_sha256"] = deterministic_hash(
        {key: value for key, value in identity.items() if key != "snapshot"}
    )
    return identity


def _profile_identity_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PROFILE_SCHEMA,
        "round_id": ROUND_ID,
        "source_inventory_sha256": profile["source_inventory_sha256"],
        "tokenizer": profile["tokenizer"],
        "selection_rule": profile["selection_rule"],
        "selection_sha256": profile["selection_sha256"],
        "corpora": profile["corpora"],
        "global_profile": profile["global_profile"],
        "disk_output_forecast": profile["disk_output_forecast"],
    }


def build_tokenizer_profile(
    *, inventory: Mapping[str, Any], hf_root: Path, progress: ProgressLog | None = None
) -> dict[str, Any]:
    assert_cpu_only_process()
    validate_inventory_artifact(inventory)
    from tokenizers import Tokenizer, __version__ as tokenizers_version

    started = _runtime_start()
    tok_identity = tokenizer_identity(hf_root)
    tokenizer_path = Path(tok_identity["snapshot"]) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Record untruncated counts, then derive the exact 512-token effective load.
    tokenizer.no_truncation()
    corpus_profiles: list[dict[str, Any]] = []
    all_counts: list[int] = []
    all_effective_counts: list[int] = []
    all_selections_for_hash: list[dict[str, Any]] = []
    total_empty = 0
    total_errors = 0
    total_truncated = 0

    if progress:
        progress.emit(
            phase="tokenizer-profile",
            event="tokenizer-bound",
            current=0,
            total=23,
            detail=f"revision={MODEL_REVISION} identity={tok_identity['tokenizer_identity_sha256']}",
        )

    for index, corpus in enumerate(inventory["corpora"]):
        counts = _load_source_token_counts(corpus)
        selections = profile_selection(corpus, counts)
        texts = _fetch_selected_texts(corpus, selections)
        records: list[dict[str, Any]] = []
        corpus_errors = 0
        corpus_empty = 0
        corpus_truncated = 0
        for selection in selections:
            source_row = int(selection["source_row"])
            text = texts[source_row]
            text_bytes = text.encode("utf-8")
            prompted = PROMPT_TEXT + text
            try:
                encoded = tokenizer.encode(prompted, add_special_tokens=True)
                token_count = len(encoded.ids)
                effective_count = min(token_count, MAX_SEQUENCE_LENGTH)
                error = None
            except Exception as exc:  # pragma: no cover - real tokenizer failures are data-dependent
                token_count = None
                effective_count = None
                error = type(exc).__name__
                corpus_errors += 1
            empty = len(text_bytes) == 0
            truncated = bool(token_count is not None and token_count > MAX_SEQUENCE_LENGTH)
            corpus_empty += int(empty)
            corpus_truncated += int(truncated)
            member, local_row = _member_for_row(corpus, source_row)
            record = {
                **selection,
                "member_relative_path": member["relative_path"],
                "member_local_row": local_row,
                "document_id": document_id(
                    corpus["corpus_id"], corpus["corpus_source_sha256"], source_row
                ),
                "raw_text_sha256": sha256_bytes(text_bytes),
                "raw_text_utf8_bytes": len(text_bytes),
                "prompted_text_sha256": sha256_bytes(prompted.encode("utf-8")),
                "token_count_untruncated": token_count,
                "token_count_effective_512": effective_count,
                "truncated_at_512": truncated,
                "empty_raw_text": empty,
                "error": error,
            }
            records.append(record)
            all_selections_for_hash.append(
                {
                    "corpus_id": corpus["corpus_id"],
                    "source_row": source_row,
                    "selection_reasons": selection["selection_reasons"],
                }
            )
        successful = [
            int(record["token_count_untruncated"])
            for record in records
            if record["token_count_untruncated"] is not None
        ]
        effective = [
            int(record["token_count_effective_512"])
            for record in records
            if record["token_count_effective_512"] is not None
        ]
        if not successful:
            raise ContractError("tokenizer_profile_failed", corpus["corpus_id"])
        all_counts.extend(successful)
        all_effective_counts.extend(effective)
        total_empty += corpus_empty
        total_errors += corpus_errors
        total_truncated += corpus_truncated
        corpus_profiles.append(
            {
                "corpus_id": corpus["corpus_id"],
                "language": corpus["language"],
                "family": corpus["family"],
                "corpus_source_sha256": corpus["corpus_source_sha256"],
                "row_count": corpus["row_count"],
                "sample_count": len(records),
                "selection_records": records,
                "token_count_untruncated": {
                    "median": _quantile_value(successful, 0.50),
                    "p95": _quantile_value(successful, 0.95),
                    "p99": _quantile_value(successful, 0.99),
                    "maximum": max(successful),
                    "mean": sum(successful) / len(successful),
                },
                "effective_token_count_512": {
                    "median": _quantile_value(effective, 0.50),
                    "p95": _quantile_value(effective, 0.95),
                    "p99": _quantile_value(effective, 0.99),
                    "maximum": max(effective),
                    "mean": sum(effective) / len(effective),
                },
                "truncated_count": corpus_truncated,
                "truncation_rate": corpus_truncated / len(records),
                "empty_count": corpus_empty,
                "empty_rate": corpus_empty / len(records),
                "error_count": corpus_errors,
                "error_rate": corpus_errors / len(records),
            }
        )
        if progress:
            progress.emit(
                phase="tokenizer-profile",
                event="corpus-profiled",
                current=index + 1,
                total=23,
                detail=(
                    f"{corpus['corpus_id']}: n={len(records)} "
                    f"p99={corpus_profiles[-1]['token_count_untruncated']['p99']} "
                    f"max={corpus_profiles[-1]['token_count_untruncated']['maximum']}"
                ),
            )

    sample_count = len(all_counts)
    if total_errors:
        raise ContractError("tokenizer_errors", f"{total_errors}/{sample_count} samples failed")
    selection_sha = deterministic_hash(all_selections_for_hash)
    weighted_mean = sum(
        float(corpus["token_count_untruncated"]["mean"]) * int(corpus["row_count"])
        for corpus in corpus_profiles
    ) / EXPECTED_TOTAL_ROWS
    profile: dict[str, Any] = {
        "schema": PROFILE_SCHEMA,
        "round_id": ROUND_ID,
        "source_inventory_sha256": inventory["source_inventory_sha256"],
        "tokenizer": {
            **{key: value for key, value in tok_identity.items() if key != "snapshot"},
            "tokenizers_package_version": tokenizers_version,
            "truncation_profile": (
                "untruncated tokenization with effective load min(count,512); no model load"
            ),
        },
        "selection_rule": {
            "version": "round0011-profile-v1",
            "per_corpus_hash_sample_rows": PROFILE_HASH_SAMPLE_ROWS,
            "hash_sample_seed": (
                "first 64 bits of sha256('round0011-profile-v1\\0' + corpus_id + "
                "'\\0' + corpus_source_sha256), PCG64 without replacement"
            ),
            "workload_strata": (
                "first source row at exact nearest-rank median/p95/p99/maximum "
                "of source chunk_token_count"
            ),
            "text_convention": "literal UTF-8 'Document: ' bytes followed by raw chunk_text",
        },
        "selection_sha256": selection_sha,
        "corpora": corpus_profiles,
        "global_profile": {
            "sample_count": sample_count,
            "token_count_untruncated": {
                "median": _quantile_value(all_counts, 0.50),
                "p95": _quantile_value(all_counts, 0.95),
                "p99": _quantile_value(all_counts, 0.99),
                "maximum": max(all_counts),
                "sample_mean": sum(all_counts) / sample_count,
                "corpus_row_weighted_mean_estimate": weighted_mean,
            },
            "effective_token_count_512": {
                "median": _quantile_value(all_effective_counts, 0.50),
                "p95": _quantile_value(all_effective_counts, 0.95),
                "p99": _quantile_value(all_effective_counts, 0.99),
                "maximum": max(all_effective_counts),
                "sample_mean": sum(all_effective_counts) / sample_count,
            },
            "truncated_count": total_truncated,
            "truncation_rate": total_truncated / sample_count,
            "empty_count": total_empty,
            "empty_rate": total_empty / sample_count,
            "error_count": total_errors,
            "error_rate": total_errors / sample_count,
        },
        "disk_output_forecast": {
            "classification": "CPU profile forecast; not an embedding benchmark",
            "embedding_dimension": EMBED_DIM,
            "persisted_dtype": OUTPUT_DTYPE,
            "bytes_per_row": OUTPUT_BYTES_PER_ROW,
            "full_universe_vector_bytes": EXPECTED_TOTAL_ROWS * OUTPUT_BYTES_PER_ROW,
            "four_unit_vector_bytes": 4 * UNIT_ROWS * OUTPUT_BYTES_PER_ROW,
            "full_universe_gib": EXPECTED_TOTAL_ROWS * OUTPUT_BYTES_PER_ROW / (1 << 30),
            "four_unit_gib": 4 * UNIT_ROWS * OUTPUT_BYTES_PER_ROW / (1 << 30),
            "recommended_free_space_gib_before_later_gpu_tranche": 100,
            "estimated_full_universe_untruncated_tokens": int(
                round(weighted_mean * EXPECTED_TOTAL_ROWS)
            ),
        },
    }
    profile["tokenizer_profile_sha256"] = deterministic_hash(
        _profile_identity_payload(profile)
    )
    profile["runtime_observation"] = _runtime_finish(started)
    assert_cpu_only_process()
    return profile


def validate_profile_artifact(profile: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
    validate_inventory_artifact(inventory)
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ContractError("profile_schema", str(profile.get("schema")))
    if profile.get("source_inventory_sha256") != inventory["source_inventory_sha256"]:
        raise ContractError("profile_source_drift", "profile does not bind this inventory")
    if profile["tokenizer"]["prompt_utf8_hex"] != PROMPT_HEX:
        raise ContractError("prompt_drift", str(profile["tokenizer"]["prompt_utf8_hex"]))
    if profile["tokenizer"]["revision"] != MODEL_REVISION:
        raise ContractError("tokenizer_revision_drift", str(profile["tokenizer"]["revision"]))
    observed = deterministic_hash(_profile_identity_payload(profile))
    if observed != profile.get("tokenizer_profile_sha256"):
        raise ContractError("profile_hash_mismatch", observed)
    if profile["global_profile"]["error_count"] != 0:
        raise ContractError("tokenizer_errors", "profile contains tokenizer errors")


def multilingual_selection_key(corpus: Mapping[str, Any]) -> str:
    return sha256_bytes(
        (
            "round0011-multilingual-unit-v1\0"
            + corpus["corpus_id"]
            + "\0"
            + corpus["corpus_source_sha256"]
        ).encode("utf-8")
    )


def _source_segments(
    corpus: Mapping[str, Any], start: int, end: int
) -> list[dict[str, Any]]:
    if start < 0 or end <= start or end > int(corpus["row_count"]):
        raise ContractError("document_row_out_of_bounds", f"[{start},{end})")
    segments: list[dict[str, Any]] = []
    cursor = start
    for member in corpus["members"]:
        member_start = int(member["corpus_row_start"])
        member_end = int(member["corpus_row_end"])
        overlap_start = max(start, member_start)
        overlap_end = min(end, member_end)
        if overlap_start < overlap_end:
            segments.append(
                {
                    "member_index": member["member_index"],
                    "relative_path": member["relative_path"],
                    "source_sha256": member["sha256"],
                    "member_local_row_start": overlap_start - member_start,
                    "member_local_row_end": overlap_end - member_start,
                    "corpus_row_start": overlap_start,
                    "corpus_row_end": overlap_end,
                    "row_count": overlap_end - overlap_start,
                }
            )
            if overlap_start != cursor:
                raise ContractError("range_gap", f"segment begins {overlap_start}, expected {cursor}")
            cursor = overlap_end
    if cursor != end:
        raise ContractError("range_gap", f"segments cover through {cursor}, expected {end}")
    return segments


def _unit_manifest(
    *,
    unit_index: int,
    unit_id: str,
    corpus: Mapping[str, Any],
    profile_corpus: Mapping[str, Any],
    tokenizer_identity_sha256: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    start = 0
    end = UNIT_ROWS
    if int(corpus["row_count"]) < UNIT_ROWS:
        raise ContractError("unit_too_large", corpus["corpus_id"])
    chunks: list[dict[str, Any]] = []
    for chunk_index, chunk_start in enumerate(range(start, end, ATOMIC_CHUNK_ROWS)):
        chunk_end = chunk_start + ATOMIC_CHUNK_ROWS
        segments = _source_segments(corpus, chunk_start, chunk_end)
        commitment_payload = {
            "corpus_id": corpus["corpus_id"],
            "corpus_source_sha256": corpus["corpus_source_sha256"],
            "source_row_start": chunk_start,
            "source_row_end": chunk_end,
            "segments": segments,
        }
        chunks.append(
            {
                "chunk_index": chunk_index,
                "chunk_id": f"{unit_id}-chunk-{chunk_index:04d}",
                "source_row_start": chunk_start,
                "source_row_end": chunk_end,
                "row_count": ATOMIC_CHUNK_ROWS,
                "first_document_id": document_id(
                    corpus["corpus_id"], corpus["corpus_source_sha256"], chunk_start
                ),
                "last_document_id": document_id(
                    corpus["corpus_id"], corpus["corpus_source_sha256"], chunk_end - 1
                ),
                "source_segments": segments,
                "source_commitment_sha256": deterministic_hash(commitment_payload),
                "future_output_relative_path": f"{unit_id}/chunk-{chunk_index:04d}.npy",
                "future_receipt_relative_path": f"{unit_id}/chunk-{chunk_index:04d}.receipt.json",
            }
        )
    expected_mean = float(profile_corpus["token_count_untruncated"]["mean"])
    unit: dict[str, Any] = {
        "unit_index": unit_index,
        "unit_id": unit_id,
        "corpus_id": corpus["corpus_id"],
        "language": corpus["language"],
        "family": corpus["family"],
        "corpus_source_sha256": corpus["corpus_source_sha256"],
        "source_row_start": start,
        "source_row_end": end,
        "row_count": end - start,
        "selection": dict(selection),
        "tokenizer_identity_sha256": tokenizer_identity_sha256,
        "prompt_utf8_hex": PROMPT_HEX,
        "atomic_chunk_rows": ATOMIC_CHUNK_ROWS,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "workload_forecast": {
            "classification": "CPU sample estimate; not GPU throughput evidence",
            "estimated_untruncated_tokens": int(round(expected_mean * UNIT_ROWS)),
            "sample_mean_tokens_per_document": expected_mean,
            "sample_p95_tokens_per_document": profile_corpus["token_count_untruncated"]["p95"],
            "sample_p99_tokens_per_document": profile_corpus["token_count_untruncated"]["p99"],
            "vector_output_bytes": UNIT_ROWS * OUTPUT_BYTES_PER_ROW,
            "historical_prompt_free_rows_per_second": None,
        },
    }
    unit["unit_manifest_sha256"] = deterministic_hash(
        {key: value for key, value in unit.items() if key != "unit_manifest_sha256"}
    )
    return unit


def _unit_set_identity_payload(units: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": UNITS_SCHEMA,
        "round_id": ROUND_ID,
        "source_inventory_sha256": units["source_inventory_sha256"],
        "tokenizer_profile_sha256": units["tokenizer_profile_sha256"],
        "selection_rule": units["selection_rule"],
        "multilingual_candidates": units["multilingual_candidates"],
        "units": units["units"],
        "yield_boundary_rule": units["yield_boundary_rule"],
    }


def build_units(
    *, inventory: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    validate_profile_artifact(profile, inventory)
    corpora = {corpus["corpus_id"]: corpus for corpus in inventory["corpora"]}
    profiles = {corpus["corpus_id"]: corpus for corpus in profile["corpora"]}
    multilingual = [corpus for corpus in inventory["corpora"] if corpus["family"] == "fineweb2"]
    candidates = sorted(
        (
            {
                "corpus_id": corpus["corpus_id"],
                "language": corpus["language"],
                "corpus_source_sha256": corpus["corpus_source_sha256"],
                "selection_key_sha256": multilingual_selection_key(corpus),
            }
            for corpus in multilingual
        ),
        key=lambda item: (item["selection_key_sha256"], item["corpus_id"]),
    )
    selected_multilingual = candidates[0]
    selected_ids = (
        "fineweb-edu",
        "redpajama-v2",
        "pile-uncopyrighted",
        selected_multilingual["corpus_id"],
    )
    unit_records: list[dict[str, Any]] = []
    for index, corpus_id in enumerate(selected_ids):
        corpus = corpora[corpus_id]
        if index < 3:
            selection = {
                "rule": "first 1,000,000 rows of each registered English corpus",
                "corpus_rank": index,
            }
        else:
            selection = {
                "rule": (
                    "lexicographically smallest sha256('round0011-multilingual-unit-v1\\0' "
                    "+ corpus_id + '\\0' + corpus_source_sha256), corpus_id tie-break"
                ),
                "selection_key_sha256": selected_multilingual["selection_key_sha256"],
                "candidate_rank": 0,
            }
        unit_records.append(
            _unit_manifest(
                unit_index=index,
                unit_id=f"unit-{index:02d}-{corpus_id}",
                corpus=corpus,
                profile_corpus=profiles[corpus_id],
                tokenizer_identity_sha256=profile["tokenizer"]["tokenizer_identity_sha256"],
                selection=selection,
            )
        )
    units: dict[str, Any] = {
        "schema": UNITS_SCHEMA,
        "round_id": ROUND_ID,
        "source_inventory_sha256": inventory["source_inventory_sha256"],
        "tokenizer_profile_sha256": profile["tokenizer_profile_sha256"],
        "selection_rule": {
            "english": "first 1,000,000 registered rows from each English corpus",
            "multilingual": (
                "minimum content-bound selection_key_sha256 across all twenty FineWeb2 corpora"
            ),
            "unit_rows": UNIT_ROWS,
            "atomic_chunk_rows": ATOMIC_CHUNK_ROWS,
        },
        "multilingual_candidates": candidates,
        "units": unit_records,
        "yield_boundary_rule": {
            "classification": "requirement for a later owner-gated GPU round",
            "atomic_boundary_rows": ATOMIC_CHUNK_ROWS,
            "rule": (
                "measure exact-revision load/read/embed/write/fsync/reopen wall for representative "
                "25k chunks; choose the smallest measured atomic yield boundary whose p90 plus "
                "15% shutdown margin is <=30 minutes; if the 25k boundary fails, stop for a "
                "scope/estimate reset rather than claiming preemptibility"
            ),
            "gpu_measurement_present": False,
        },
    }
    validate_unit_set(units, inventory, profile)
    units["unit_manifest_set_sha256"] = deterministic_hash(_unit_set_identity_payload(units))
    return units


def validate_unit_set(
    units: Mapping[str, Any], inventory: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    if units.get("schema") != UNITS_SCHEMA:
        raise ContractError("unit_schema", str(units.get("schema")))
    if units.get("source_inventory_sha256") != inventory["source_inventory_sha256"]:
        raise ContractError("unit_source_drift", "unit set binds a different inventory")
    if units.get("tokenizer_profile_sha256") != profile["tokenizer_profile_sha256"]:
        raise ContractError("unit_tokenizer_drift", "unit set binds a different profile")
    expected_tokenizer = profile["tokenizer"]["tokenizer_identity_sha256"]
    unit_ids: set[str] = set()
    document_witnesses: list[str] = []
    for unit in units["units"]:
        if unit["unit_id"] in unit_ids:
            raise ContractError("duplicate_unit", unit["unit_id"])
        unit_ids.add(unit["unit_id"])
        if unit["tokenizer_identity_sha256"] != expected_tokenizer:
            raise ContractError("tokenizer_drift", unit["unit_id"])
        if unit["prompt_utf8_hex"] != PROMPT_HEX:
            raise ContractError("prompt_drift", unit["unit_id"])
        if int(unit["row_count"]) != UNIT_ROWS or int(unit["chunk_count"]) != 40:
            raise ContractError("unit_shape", unit["unit_id"])
        cursor = int(unit["source_row_start"])
        for index, chunk in enumerate(unit["chunks"]):
            if int(chunk["chunk_index"]) != index:
                raise ContractError("source_reordered", unit["unit_id"])
            start = int(chunk["source_row_start"])
            end = int(chunk["source_row_end"])
            if start != cursor:
                code = "range_overlap" if start < cursor else "range_gap"
                raise ContractError(code, chunk["chunk_id"])
            if end - start != ATOMIC_CHUNK_ROWS:
                raise ContractError("invalid_range", chunk["chunk_id"])
            cursor = end
            document_witnesses.extend(
                [chunk["first_document_id"], chunk["last_document_id"]]
            )
        if cursor != int(unit["source_row_end"]):
            raise ContractError("range_gap", unit["unit_id"])
        observed_unit = deterministic_hash(
            {key: value for key, value in unit.items() if key != "unit_manifest_sha256"}
        )
        if observed_unit != unit["unit_manifest_sha256"]:
            raise ContractError("unit_hash_mismatch", unit["unit_id"])
    if len(units["units"]) != 4:
        raise ContractError("unit_count", str(len(units["units"])))
    validate_unique_document_ids(document_witnesses)


def validate_unit_artifact(
    units: Mapping[str, Any], inventory: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    validate_unit_set(units, inventory, profile)
    observed = deterministic_hash(_unit_set_identity_payload(units))
    if observed != units.get("unit_manifest_set_sha256"):
        raise ContractError("unit_set_hash_mismatch", observed)


def _validate_receipt(
    *, unit: Mapping[str, Any], chunk: Mapping[str, Any], receipt: Mapping[str, Any], output: Path
) -> None:
    required = {
        "status": "complete",
        "unit_manifest_sha256": unit["unit_manifest_sha256"],
        "chunk_id": chunk["chunk_id"],
        "source_commitment_sha256": chunk["source_commitment_sha256"],
        "tokenizer_identity_sha256": unit["tokenizer_identity_sha256"],
        "prompt_utf8_hex": unit["prompt_utf8_hex"],
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            code = "prompt_drift" if key == "prompt_utf8_hex" else "receipt_binding_drift"
            raise ContractError(code, f"{chunk['chunk_id']}:{key}")
    if int(receipt.get("output_bytes", -1)) != output.stat().st_size:
        raise ContractError("output_size_drift", chunk["chunk_id"])
    if receipt.get("output_sha256") != file_sha256(output):
        raise ContractError("output_hash_drift", chunk["chunk_id"])


def resume_index(unit: Mapping[str, Any], output_root: Path) -> int:
    """Return the first incomplete chunk, rejecting every ambiguous state."""

    first_missing: int | None = None
    for index, chunk in enumerate(unit["chunks"]):
        output = output_root / f"chunk-{index:04d}.bin"
        receipt_path = output_root / f"chunk-{index:04d}.receipt.json"
        output_exists = output.exists()
        receipt_exists = receipt_path.exists()
        if output_exists and (output.is_symlink() or not output.is_file()):
            raise ContractError("output_collision", str(output))
        if receipt_exists and (receipt_path.is_symlink() or not receipt_path.is_file()):
            raise ContractError("partial_receipt", str(receipt_path))
        if output_exists != receipt_exists:
            code = "output_collision" if output_exists else "partial_receipt"
            raise ContractError(code, chunk["chunk_id"])
        if output_exists:
            if first_missing is not None:
                raise ContractError("noncontiguous_resume", chunk["chunk_id"])
            _validate_receipt(
                unit=unit,
                chunk=chunk,
                receipt=read_json(receipt_path),
                output=output,
            )
        elif first_missing is None:
            first_missing = index
    return len(unit["chunks"]) if first_missing is None else first_missing


def _fixture_unit() -> dict[str, Any]:
    chunks = []
    for index in range(4):
        chunks.append(
            {
                "chunk_index": index,
                "chunk_id": f"fixture-chunk-{index:04d}",
                "source_commitment_sha256": sha256_bytes(f"source-{index}".encode()),
            }
        )
    unit: dict[str, Any] = {
        "unit_id": "fixture-unit",
        "tokenizer_identity_sha256": sha256_bytes(b"fixture-tokenizer"),
        "prompt_utf8_hex": PROMPT_HEX,
        "chunks": chunks,
    }
    unit["unit_manifest_sha256"] = deterministic_hash(
        {key: value for key, value in unit.items() if key != "unit_manifest_sha256"}
    )
    return unit


def _write_fixture_chunk(root: Path, unit: Mapping[str, Any], index: int) -> None:
    chunk = unit["chunks"][index]
    output = root / f"chunk-{index:04d}.bin"
    payload = f"fixture-output-{index}\n".encode()
    atomic_write_new_bytes(output, payload)
    receipt = {
        "status": "complete",
        "unit_manifest_sha256": unit["unit_manifest_sha256"],
        "chunk_id": chunk["chunk_id"],
        "source_commitment_sha256": chunk["source_commitment_sha256"],
        "tokenizer_identity_sha256": unit["tokenizer_identity_sha256"],
        "prompt_utf8_hex": unit["prompt_utf8_hex"],
        "output_bytes": len(payload),
        "output_sha256": sha256_bytes(payload),
    }
    atomic_write_new_json(root / f"chunk-{index:04d}.receipt.json", receipt)


def _expect_rejection(name: str, expected_code: str, fn: Any) -> dict[str, Any]:
    try:
        fn()
    except ContractError as error:
        if error.code != expected_code:
            raise ContractError(
                "fixture_wrong_rejection",
                f"{name}: observed {error.code}, expected {expected_code}",
            ) from error
        return {
            "name": name,
            "expected": "reject",
            "observed": "reject",
            "reason_code": error.code,
            "passed": True,
        }
    raise ContractError("fixture_failed_open", name)


def _fixture_identity_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": FIXTURE_SCHEMA,
        "round_id": ROUND_ID,
        "cases": receipt["cases"],
    }


def run_fixture_matrix(*, scratch_parent: Path) -> dict[str, Any]:
    started = _runtime_start()
    cases: list[dict[str, Any]] = []
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="round0011-fixtures-", dir=scratch_parent) as tmp:
        root = Path(tmp)
        unit = _fixture_unit()
        _write_fixture_chunk(root, unit, 0)
        _write_fixture_chunk(root, unit, 1)
        index = resume_index(unit, root)
        if index != 2:
            raise ContractError("fixture_resume_index", f"observed {index}, expected 2")
        cases.append(
            {
                "name": "resume-after-interrupted-chunk",
                "expected": "resume-index-2",
                "observed": "resume-index-2",
                "passed": True,
            }
        )

        partial = root / "partial"
        partial.mkdir()
        atomic_write_new_json(partial / "chunk-0000.receipt.json", {"status": "complete"})
        cases.append(
            _expect_rejection(
                "partial-receipt", "partial_receipt", lambda: resume_index(unit, partial)
            )
        )

        collision = root / "collision"
        collision.mkdir()
        atomic_write_new_bytes(collision / "chunk-0000.bin", b"unreceipted")
        cases.append(
            _expect_rejection(
                "output-collision", "output_collision", lambda: resume_index(unit, collision)
            )
        )

        prompt = root / "prompt"
        prompt.mkdir()
        _write_fixture_chunk(prompt, unit, 0)
        prompt_receipt_path = prompt / "chunk-0000.receipt.json"
        prompt_receipt = read_json(prompt_receipt_path)
        prompt_receipt["prompt_utf8_hex"] = "00"
        prompt_receipt_path.unlink()
        atomic_write_new_json(prompt_receipt_path, prompt_receipt)
        cases.append(
            _expect_rejection("prompt-drift", "prompt_drift", lambda: resume_index(unit, prompt))
        )

        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        _write_fixture_chunk(tokenizer, unit, 0)
        tokenizer_receipt_path = tokenizer / "chunk-0000.receipt.json"
        tokenizer_receipt = read_json(tokenizer_receipt_path)
        tokenizer_receipt["tokenizer_identity_sha256"] = "0" * 64
        tokenizer_receipt_path.unlink()
        atomic_write_new_json(tokenizer_receipt_path, tokenizer_receipt)
        cases.append(
            _expect_rejection(
                "tokenizer-drift",
                "receipt_binding_drift",
                lambda: resume_index(unit, tokenizer),
            )
        )

        source = root / "source.bin"
        atomic_write_new_bytes(source, b"source-v1")
        expected_sha = file_sha256(source)
        source.unlink()
        atomic_write_new_bytes(source, b"source-v2")
        cases.append(
            _expect_rejection(
                "source-drift",
                "source_hash_drift",
                lambda: _require_hash(source, expected_sha),
            )
        )

        base_ranges = [
            {
                "relative_path": "a.parquet",
                "corpus_row_start": 0,
                "corpus_row_end": 2,
                "row_count": 2,
                "filesystem": {"device": 1, "inode": 1},
            },
            {
                "relative_path": "b.parquet",
                "corpus_row_start": 2,
                "corpus_row_end": 4,
                "row_count": 2,
                "filesystem": {"device": 1, "inode": 2},
            },
        ]
        reordered = [base_ranges[1], base_ranges[0]]
        cases.append(
            _expect_rejection(
                "source-row-reorder",
                "range_gap",
                lambda: validate_range_cover(reordered, 4),
            )
        )
        overlap = json.loads(json.dumps(base_ranges))
        overlap[1]["corpus_row_start"] = 1
        overlap[1]["row_count"] = 3
        cases.append(
            _expect_rejection(
                "range-overlap", "range_overlap", lambda: validate_range_cover(overlap, 4)
            )
        )
        gap = json.loads(json.dumps(base_ranges))
        gap[1]["corpus_row_start"] = 3
        gap[1]["corpus_row_end"] = 5
        cases.append(
            _expect_rejection(
                "range-gap", "range_gap", lambda: validate_range_cover(gap, 5)
            )
        )
        cases.append(
            _expect_rejection(
                "duplicate-document-id",
                "duplicate_document_id",
                lambda: validate_unique_document_ids(["doc-a", "doc-a"]),
            )
        )

    receipt: dict[str, Any] = {
        "schema": FIXTURE_SCHEMA,
        "round_id": ROUND_ID,
        "cases": cases,
        "all_passed": all(case["passed"] for case in cases),
    }
    receipt["fixture_matrix_sha256"] = deterministic_hash(_fixture_identity_payload(receipt))
    receipt["runtime_observation"] = _runtime_finish(started)
    return receipt


def _require_hash(path: Path, expected: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise ContractError("source_hash_drift", f"{observed} != {expected}")


def validate_fixture_artifact(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema") != FIXTURE_SCHEMA or not receipt.get("all_passed"):
        raise ContractError("fixture_receipt", "fixture matrix is not completely passing")
    observed = deterministic_hash(_fixture_identity_payload(receipt))
    if observed != receipt.get("fixture_matrix_sha256"):
        raise ContractError("fixture_hash_mismatch", observed)


def capability_payload(
    *,
    inventory: Mapping[str, Any],
    profile: Mapping[str, Any],
    units: Mapping[str, Any],
    fixtures: Mapping[str, Any],
) -> dict[str, Any]:
    validate_unit_artifact(units, inventory, profile)
    validate_fixture_artifact(fixtures)
    return {
        "schema": CAPABILITY_SCHEMA,
        "capability": CAPABILITY_NAME,
        "round_id": ROUND_ID,
        "source_inventory_sha256": inventory["source_inventory_sha256"],
        "tokenizer_profile_sha256": profile["tokenizer_profile_sha256"],
        "tokenizer_identity_sha256": profile["tokenizer"]["tokenizer_identity_sha256"],
        "unit_manifest_set_sha256": units["unit_manifest_set_sha256"],
        "fixture_matrix_sha256": fixtures["fixture_matrix_sha256"],
        "observed_corpora": inventory["observed"]["corpora"],
        "observed_rows": inventory["observed"]["total_rows"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_utf8_hex": PROMPT_HEX,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "unit_ids": [unit["unit_id"] for unit in units["units"]],
        "hash_scope": (
            "all deterministic direct-source, tokenizer, profile, unit, and fixture evidence; "
            "created timestamps and observed CPU/RSS/I/O timings are intentionally outside the "
            "reproducible capability hash and remain bound by component artifact file hashes"
        ),
    }


def assemble_capability(
    *,
    root: Path,
    inventory: Mapping[str, Any],
    profile: Mapping[str, Any],
    units: Mapping[str, Any],
    fixtures: Mapping[str, Any],
) -> dict[str, Any]:
    payload = capability_payload(
        inventory=inventory, profile=profile, units=units, fixtures=fixtures
    )
    component_paths = (
        root / "inventory" / "23-corpus-source-inventory.json",
        root / "tokenizer" / "tokenizer-profile.json",
        root / "units" / "four-unit-manifests.json",
        root / "fixtures" / "fixture-receipt.json",
    )
    components = []
    for path in component_paths:
        components.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema": CAPABILITY_SCHEMA,
        "created_utc": utc_now(),
        "capability_payload": payload,
        "capability_sha256": deterministic_hash(payload),
        "component_artifacts": components,
        "gpu_work_performed": False,
        "torch_or_cuda_initialized": False,
        "result_status": "implementation-handoff; formal result is planner-managed",
    }


def validate_capability_artifact(capability: Mapping[str, Any]) -> None:
    if capability.get("schema") != CAPABILITY_SCHEMA:
        raise ContractError("capability_schema", str(capability.get("schema")))
    observed = deterministic_hash(capability["capability_payload"])
    if observed != capability.get("capability_sha256"):
        raise ContractError("capability_hash_mismatch", observed)
    for component in capability["component_artifacts"]:
        path = Path(component["path"])
        if file_sha256(path) != component["sha256"]:
            raise ContractError("component_artifact_drift", str(path))


def reopen_capability(
    *, embedding_root: Path, hf_root: Path, root: Path, progress: ProgressLog
) -> dict[str, Any]:
    started = _runtime_start()
    capability_path = root / f"{CAPABILITY_NAME}.json"
    capability = read_json(capability_path)
    validate_capability_artifact(capability)
    progress.emit(
        phase="capability-reopen",
        event="direct-rehash-started",
        current=0,
        total=4,
        detail="rehashing all 59 direct source Parquets",
        artifact=str(capability_path),
    )
    inventory = build_inventory(embedding_root=embedding_root, progress=progress)
    original_inventory = read_json(root / "inventory" / "23-corpus-source-inventory.json")
    if inventory["source_inventory_sha256"] != original_inventory["source_inventory_sha256"]:
        raise ContractError("source_hash_drift", "direct inventory no longer reproduces")
    progress.emit(
        phase="capability-reopen",
        event="inventory-reproduced",
        current=1,
        total=4,
        detail=inventory["source_inventory_sha256"],
    )
    profile = build_tokenizer_profile(inventory=inventory, hf_root=hf_root, progress=progress)
    original_profile = read_json(root / "tokenizer" / "tokenizer-profile.json")
    if profile["tokenizer_profile_sha256"] != original_profile["tokenizer_profile_sha256"]:
        raise ContractError("tokenizer_profile_drift", "profile no longer reproduces")
    progress.emit(
        phase="capability-reopen",
        event="profile-reproduced",
        current=2,
        total=4,
        detail=profile["tokenizer_profile_sha256"],
    )
    units = build_units(inventory=inventory, profile=profile)
    original_units = read_json(root / "units" / "four-unit-manifests.json")
    if units["unit_manifest_set_sha256"] != original_units["unit_manifest_set_sha256"]:
        raise ContractError("unit_manifest_drift", "unit manifests no longer reproduce")
    fixtures = run_fixture_matrix(scratch_parent=root / "fixtures")
    original_fixtures = read_json(root / "fixtures" / "fixture-receipt.json")
    if fixtures["fixture_matrix_sha256"] != original_fixtures["fixture_matrix_sha256"]:
        raise ContractError("fixture_hash_mismatch", "fixture matrix no longer reproduces")
    payload = capability_payload(
        inventory=inventory, profile=profile, units=units, fixtures=fixtures
    )
    reproduced = deterministic_hash(payload)
    if reproduced != capability["capability_sha256"]:
        raise ContractError(
            "capability_hash_mismatch",
            f"reproduced {reproduced}, expected {capability['capability_sha256']}",
        )
    progress.emit(
        phase="capability-reopen",
        event="capability-reproduced",
        current=4,
        total=4,
        detail=reproduced,
        artifact=str(capability_path),
    )
    receipt = {
        "schema": REOPEN_SCHEMA,
        "round_id": ROUND_ID,
        "created_utc": utc_now(),
        "capability_path": str(capability_path),
        "expected_capability_sha256": capability["capability_sha256"],
        "reproduced_capability_sha256": reproduced,
        "direct_source_rehash": True,
        "tokenizer_profile_recomputed": True,
        "unit_manifests_recomputed": True,
        "fixture_matrix_recomputed": True,
        "passed": True,
        "runtime_observation": _runtime_finish(started),
    }
    assert_cpu_only_process()
    return receipt


def _paths(root: Path) -> dict[str, Path]:
    return {
        "inventory": root / "inventory" / "23-corpus-source-inventory.json",
        "profile": root / "tokenizer" / "tokenizer-profile.json",
        "units": root / "units" / "four-unit-manifests.json",
        "fixtures": root / "fixtures" / "fixture-receipt.json",
        "capability": root / f"{CAPABILITY_NAME}.json",
        "reopen": root / "receipts" / "capability-reopen-receipt.json",
        "progress": root / "logs" / "runner-progress.jsonl",
    }


def _require_root(root: Path) -> None:
    expected = DEFAULT_OUTPUT_ROOT.resolve()
    actual = root.resolve()
    if actual != expected:
        raise ContractError("wrong_output_root", f"{actual} != {expected}")
    if root.is_symlink() or not root.is_dir():
        raise ContractError("output_root_missing", str(root))


def _write_component(path: Path, value: Mapping[str, Any], progress: ProgressLog, phase: str) -> None:
    disposition = atomic_write_new_json(path, value)
    progress.emit(
        phase=phase,
        event="artifact-sealed",
        current=1,
        total=1,
        detail=f"{disposition}; sha256={file_sha256(path)}",
        artifact=str(path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("inventory", "profile", "units", "fixtures", "assemble", "reopen"),
    )
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    assert_cpu_only_process()
    _require_root(args.output_root)
    paths = _paths(args.output_root)
    progress = ProgressLog(paths["progress"])
    progress.emit(
        phase=args.command,
        event="child-started",
        current=0,
        total=1,
        detail=f"pid={os.getpid()} cuda_hidden=true",
        artifact=str(paths.get(args.command, paths["progress"])),
    )

    if args.command == "inventory":
        inventory = build_inventory(embedding_root=args.embedding_root, progress=progress)
        _write_component(paths["inventory"], inventory, progress, "source-inventory")
    elif args.command == "profile":
        inventory = read_json(paths["inventory"])
        profile = build_tokenizer_profile(
            inventory=inventory, hf_root=args.hf_root, progress=progress
        )
        _write_component(paths["profile"], profile, progress, "tokenizer-profile")
    elif args.command == "units":
        inventory = read_json(paths["inventory"])
        profile = read_json(paths["profile"])
        units = build_units(inventory=inventory, profile=profile)
        _write_component(paths["units"], units, progress, "unit-manifests")
    elif args.command == "fixtures":
        fixtures = run_fixture_matrix(scratch_parent=args.output_root / "fixtures")
        _write_component(paths["fixtures"], fixtures, progress, "fixtures")
    elif args.command == "assemble":
        inventory = read_json(paths["inventory"])
        profile = read_json(paths["profile"])
        units = read_json(paths["units"])
        fixtures = read_json(paths["fixtures"])
        capability = assemble_capability(
            root=args.output_root,
            inventory=inventory,
            profile=profile,
            units=units,
            fixtures=fixtures,
        )
        _write_component(paths["capability"], capability, progress, "capability-assembly")
    else:
        receipt = reopen_capability(
            embedding_root=args.embedding_root,
            hf_root=args.hf_root,
            root=args.output_root,
            progress=progress,
        )
        _write_component(paths["reopen"], receipt, progress, "capability-reopen")

    assert_cpu_only_process()
    progress.emit(
        phase=args.command,
        event="child-complete",
        current=1,
        total=1,
        detail="stage completed without Torch/CUDA initialization",
        artifact=str(paths.get(args.command, paths["progress"])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
