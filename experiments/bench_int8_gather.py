"""Round 0025 int8 shard builder and gather→H2D→dequant benchmark.

The MiniLM source files in /data/embeddings are raw headerless float32 buffers
despite their .npy suffix.  R0025 defines the row universe as complete
384-float rows in sorted source files.  A trailing non-row byte fragment is
not consumed as data, but the full source-file SHA-256 still binds it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    refuse_existing,
)


ROUND_ID = "0025"
ROUND_LABEL = "Round 0025"
DIMENSION = 384
ROW_BYTES = DIMENSION * 4
FIRST_ROWS_PER_CORPUS = 50_000_000
CANARY_ROWS = 1_000_000
FULL_SAMPLE_SIZE = 10_000
SANITY_SAMPLE_SEED = 20260719
DEFAULT_OUTPUT_ROOT = "/data/latent-basemap/runs/round-0025/queue/artifacts"

SOURCE_ROOTS = {
    "fineweb": "/data/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "redpajama": "/data/embeddings/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "pile": "/data/embeddings/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
}


class Round0025Error(RuntimeError):
    """Fail-closed R0025 runtime error."""


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _json(path: str | os.PathLike, body: dict[str, Any]) -> dict[str, Any]:
    value = _seal(body)
    atomic_write_new_json(path, value, immutable=True, indent=1)
    return value


def _sha256_file(path: str, chunk_size: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: str, *, hash_file: bool) -> dict[str, Any]:
    canonical = os.path.realpath(path)
    if canonical != os.path.abspath(path) or not os.path.isfile(canonical):
        raise Round0025Error(f"{ROUND_LABEL} source path is not canonical file: {path}")
    size = os.path.getsize(canonical)
    return {
        "canonical_path": canonical,
        "kind": "file",
        "bytes": int(size),
        "sha256": _sha256_file(canonical) if hash_file else None,
    }


def build_source_plan(
    source_roots: dict[str, str] | None = None,
    *,
    hash_files: bool = True,
) -> dict[str, Any]:
    roots = source_roots or SOURCE_ROOTS
    sources: list[dict[str, Any]] = []
    corpus_summaries: dict[str, Any] = {}
    all_global_cursor = 0
    first_global_cursor = 0
    trailing_fragments: list[dict[str, Any]] = []

    for corpus in ("fineweb", "redpajama", "pile"):
        root = os.path.realpath(roots[corpus])
        paths = sorted(str(path) for path in Path(root).glob("data-*.npy"))
        if not paths:
            raise Round0025Error(f"{ROUND_LABEL} no source shards for {corpus}: {root}")
        corpus_rows = 0
        first_rows = 0
        corpus_start = all_global_cursor
        first_start = first_global_cursor
        for shard_index, path in enumerate(paths):
            signature = _file_signature(path, hash_file=hash_files)
            size = int(signature["bytes"])
            rows = size // ROW_BYTES
            trailing = size % ROW_BYTES
            if rows <= 0:
                raise Round0025Error(f"{ROUND_LABEL} empty source shard: {path}")
            local_first_start = 0
            local_first_stop = min(rows, max(0, FIRST_ROWS_PER_CORPUS - first_rows))
            contributes_first = local_first_stop > 0
            if trailing:
                trailing_fragments.append({
                    "corpus": corpus,
                    "path": signature["canonical_path"],
                    "bytes": trailing,
                    "full_rows": rows,
                    "policy": "ignored-non-row-trailing-fragment-full-file-hash-bound",
                })
            item = {
                "corpus": corpus,
                "shard_index": shard_index,
                "signature": signature,
                "rows": int(rows),
                "trailing_bytes": int(trailing),
                "all_global_start": int(all_global_cursor),
                "all_global_stop": int(all_global_cursor + rows),
                "first50_local_start": int(local_first_start),
                "first50_local_stop": int(local_first_stop),
                "first50_global_start": (
                    int(first_global_cursor) if contributes_first else None
                ),
                "first50_global_stop": (
                    int(first_global_cursor + local_first_stop)
                    if contributes_first else None
                ),
            }
            sources.append(item)
            all_global_cursor += rows
            corpus_rows += rows
            if contributes_first:
                first_global_cursor += local_first_stop
                first_rows += local_first_stop
        if first_rows != FIRST_ROWS_PER_CORPUS:
            raise Round0025Error(
                f"{ROUND_LABEL} {corpus} has {first_rows} first rows, expected "
                f"{FIRST_ROWS_PER_CORPUS}")
        corpus_summaries[corpus] = {
            "root": root,
            "file_count": len(paths),
            "all_rows": int(corpus_rows),
            "all_global_start": int(corpus_start),
            "all_global_stop": int(corpus_start + corpus_rows),
            "first50_rows": int(first_rows),
            "first50_global_start": int(first_start),
            "first50_global_stop": int(first_start + first_rows),
        }

    body = {
        "schema": "round0025-source-plan-v1",
        "dimension": DIMENSION,
        "row_bytes": ROW_BYTES,
        "source_format": "raw-headerless-float32-little-endian-with-.npy-suffix",
        "row_policy": "complete-384-float32-rows-only",
        "trailing_fragment_policy": (
            "ignore-incomplete-non-row-trailing-bytes-but-bind-full-source-file-sha256"
        ),
        "corpora": corpus_summaries,
        "universes": {
            "minilm-int8-150m": {
                "rows": int(3 * FIRST_ROWS_PER_CORPUS),
                "composition": "first-50m-rows-per-corpus-in-fineweb-redpajama-pile-order",
            },
            "minilm-int8-405m": {
                "rows": int(all_global_cursor),
                "composition": "all-complete-local-rows-in-fineweb-redpajama-pile-order",
            },
        },
        "trailing_fragments": trailing_fragments,
        "sources": sources,
    }
    return _seal(body)


def _source_expected_sha(source: dict[str, Any]) -> str:
    value = source["signature"].get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise Round0025Error(f"{ROUND_LABEL} source plan lacks full SHA-256")
    return value


def _read_source_blocks(source: dict[str, Any], *, block_rows: int) -> Iterable[tuple[int, np.ndarray, str]]:
    path = source["signature"]["canonical_path"]
    rows = int(source["rows"])
    trailing = int(source["trailing_bytes"])
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as handle:
        local = 0
        while local < rows:
            take = min(block_rows, rows - local)
            raw = handle.read(take * ROW_BYTES)
            if len(raw) != take * ROW_BYTES:
                raise Round0025Error(f"{ROUND_LABEL} short read: {path}")
            digest.update(raw)
            array = np.frombuffer(raw, dtype="<f4").reshape(take, DIMENSION)
            yield local, array, ""
            local += take
        if trailing:
            fragment = handle.read(trailing)
            if len(fragment) != trailing:
                raise Round0025Error(f"{ROUND_LABEL} short trailing read: {path}")
            digest.update(fragment)
        extra = handle.read(1)
        if extra:
            raise Round0025Error(f"{ROUND_LABEL} source grew during read: {path}")
    observed = digest.hexdigest()
    expected = _source_expected_sha(source)
    if observed != expected:
        raise Round0025Error(
            f"{ROUND_LABEL} source hash mismatch for {path}: {observed} != {expected}")


def _quantize_block(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if block.dtype != np.dtype("<f4") or block.ndim != 2 or block.shape[1] != DIMENSION:
        raise Round0025Error(f"{ROUND_LABEL} source block shape/dtype changed")
    if not np.isfinite(block).all():
        raise Round0025Error(f"{ROUND_LABEL} source block contains non-finite values")
    max_abs = np.max(np.abs(block), axis=1).astype(np.float32)
    safe = np.where(max_abs > 0.0, max_abs / 127.0, 1.0 / 127.0).astype(np.float32)
    quantized = np.rint(block / safe[:, None]).clip(-127, 127).astype(np.int8)
    return quantized, safe.astype("<f2")


class AtomicRawArray:
    def __init__(self, destination: str, *, dtype: str, shape: tuple[int, ...]) -> None:
        self.destination = refuse_existing(destination, label=f"{ROUND_LABEL} raw array")
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(v) for v in shape)
        self.bytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        self.temp = os.path.join(
            os.path.dirname(self.destination),
            f".{os.path.basename(self.destination)}.partial-{os.getpid()}")
        if os.path.exists(self.temp):
            raise FileExistsError(self.temp)
        self.array = np.memmap(self.temp, dtype=self.dtype, mode="w+", shape=self.shape)
        self.digest = hashlib.sha256()
        self.cursor = 0

    def write(self, values: np.ndarray) -> None:
        n = len(values)
        if self.cursor + n > self.shape[0]:
            raise Round0025Error(f"{ROUND_LABEL} output overflow: {self.destination}")
        cast = np.ascontiguousarray(values, dtype=self.dtype)
        self.array[self.cursor:self.cursor + n] = cast
        self.digest.update(cast.tobytes(order="C"))
        self.cursor += n

    def publish(self) -> dict[str, Any]:
        if self.cursor != self.shape[0]:
            raise Round0025Error(
                f"{ROUND_LABEL} output row count mismatch for {self.destination}: "
                f"{self.cursor} != {self.shape[0]}")
        self.array.flush()
        del self.array
        with open(self.temp, "rb") as handle:
            os.fsync(handle.fileno())
        if os.path.getsize(self.temp) != self.bytes:
            raise Round0025Error(f"{ROUND_LABEL} output size mismatch: {self.temp}")
        os.chmod(self.temp, 0o444)
        os.link(self.temp, self.destination, follow_symlinks=False)
        os.unlink(self.temp)
        return {
            "canonical_path": self.destination,
            "kind": "file",
            "bytes": self.bytes,
            "sha256": self.digest.hexdigest(),
        }


def _cosine_from_quantized(original: np.ndarray, quantized: np.ndarray, scale: np.ndarray) -> np.ndarray:
    dequantized = quantized.astype(np.float32) * scale.astype(np.float32)[:, None]
    numerator = np.sum(original * dequantized, axis=1)
    denom = np.linalg.norm(original, axis=1) * np.linalg.norm(dequantized, axis=1)
    return numerator / np.maximum(denom, 1e-12)


def _sample_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        raise Round0025Error(f"{ROUND_LABEL} missing quantization sanity samples")
    return {
        "count": int(len(array)),
        "minimum": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
    }


def quantize_full(source_plan: dict[str, Any], output_root: str, *, block_rows: int = 65536) -> dict[str, Any]:
    preflight_free_gib = _disk_free_gib(os.path.dirname(os.path.abspath(output_root)))
    if preflight_free_gib < 400.0:
        raise Round0025Error(
            f"{ROUND_LABEL} disk free {preflight_free_gib:.2f} GiB < 400 GiB")
    root = create_fresh_directory(output_root, label=f"{ROUND_LABEL} int8 output")
    universes = source_plan["universes"]
    rows150 = int(universes["minilm-int8-150m"]["rows"])
    rows405 = int(universes["minilm-int8-405m"]["rows"])
    for universe in ("minilm-int8-150m", "minilm-int8-405m"):
        os.makedirs(os.path.join(root, universe), exist_ok=True)
    outputs = {
        "minilm-int8-150m": {
            "int8": AtomicRawArray(
                os.path.join(root, "minilm-int8-150m", "embeddings.i8"),
                dtype="i1", shape=(rows150, DIMENSION)),
            "scales": AtomicRawArray(
                os.path.join(root, "minilm-int8-150m", "scales.f16"),
                dtype="<f2", shape=(rows150,)),
        },
        "minilm-int8-405m": {
            "int8": AtomicRawArray(
                os.path.join(root, "minilm-int8-405m", "embeddings.i8"),
                dtype="i1", shape=(rows405, DIMENSION)),
            "scales": AtomicRawArray(
                os.path.join(root, "minilm-int8-405m", "scales.f16"),
                dtype="<f2", shape=(rows405,)),
        },
    }

    rng = np.random.RandomState(SANITY_SAMPLE_SEED)
    sample_ids = np.sort(rng.choice(rows405, FULL_SAMPLE_SIZE, replace=False)).astype(np.int64)
    sample_cursor = 0
    sample_cosines: list[float] = []
    started = time.monotonic()
    source_receipts: list[dict[str, Any]] = []

    for source in source_plan["sources"]:
        all_start = int(source["all_global_start"])
        first_start = source.get("first50_global_start")
        first_stop = source.get("first50_global_stop")
        for local_start, block, _ in _read_source_blocks(source, block_rows=block_rows):
            q, scales = _quantize_block(block)
            n = len(block)
            all_block_start = all_start + local_start
            all_block_stop = all_block_start + n
            outputs["minilm-int8-405m"]["int8"].write(q)
            outputs["minilm-int8-405m"]["scales"].write(scales)

            while sample_cursor < len(sample_ids) and sample_ids[sample_cursor] < all_block_stop:
                selected: list[int] = []
                while sample_cursor < len(sample_ids) and sample_ids[sample_cursor] < all_block_stop:
                    if sample_ids[sample_cursor] >= all_block_start:
                        selected.append(int(sample_ids[sample_cursor] - all_block_start))
                    sample_cursor += 1
                if selected:
                    idx = np.asarray(selected, dtype=np.int64)
                    sample_cosines.extend(
                        float(value)
                        for value in _cosine_from_quantized(block[idx], q[idx], scales[idx])
                    )

            if first_start is not None and first_stop is not None:
                local_first_lo = int(source["first50_local_start"])
                local_first_hi = int(source["first50_local_stop"])
                block_lo = local_start
                block_hi = local_start + n
                overlap_lo = max(block_lo, local_first_lo)
                overlap_hi = min(block_hi, local_first_hi)
                if overlap_hi > overlap_lo:
                    lo = overlap_lo - block_lo
                    hi = overlap_hi - block_lo
                    outputs["minilm-int8-150m"]["int8"].write(q[lo:hi])
                    outputs["minilm-int8-150m"]["scales"].write(scales[lo:hi])

        source_receipts.append({
            "path": source["signature"]["canonical_path"],
            "sha256": _source_expected_sha(source),
            "bytes": int(source["signature"]["bytes"]),
            "rows": int(source["rows"]),
            "trailing_bytes": int(source["trailing_bytes"]),
        })

    published = {}
    for universe, arrays in outputs.items():
        published[universe] = {
            "rows": int(universes[universe]["rows"]),
            "dimension": DIMENSION,
            "int8": arrays["int8"].publish(),
            "scales": arrays["scales"].publish(),
            "row_scale_dtype": "<f2",
            "embedding_dtype": "int8",
            "quantization": "per-row-symmetric-round-clip-minus127-plus127",
        }

    body = {
        "schema": "round0025-int8-shards-v1",
        "source_plan_identity_sha256": source_plan["identity_sha256"],
        "source_policy": {
            "source_format": source_plan["source_format"],
            "row_policy": source_plan["row_policy"],
            "trailing_fragment_policy": source_plan["trailing_fragment_policy"],
            "trailing_fragments": source_plan["trailing_fragments"],
        },
        "universes": published,
        "source_receipts": source_receipts,
        "quantization_sanity": {
            "universe": "minilm-int8-405m",
            "sample_seed": SANITY_SAMPLE_SEED,
            "sample_size": FULL_SAMPLE_SIZE,
            "sample_ids_sha256": hashlib.sha256(sample_ids.tobytes(order="C")).hexdigest(),
            "cosine": _sample_summary(sample_cosines),
            "diagnostic_threshold": 0.999,
            "threshold_passed": min(sample_cosines) >= 0.999 if sample_cosines else False,
        },
        "preflight_free_disk_gib": preflight_free_gib,
        "wall_seconds": round(time.monotonic() - started, 6),
    }
    manifest = _json(os.path.join(root, "int8-shards-v1.json"), body)
    return manifest


def _reserve_small_quantized_slice(
    source_plan: dict[str, Any],
    output_root: str,
    *,
    rows: int,
    block_rows: int = 65536,
) -> dict[str, Any]:
    root = create_fresh_directory(output_root, label=f"{ROUND_LABEL} canary output")
    int8 = AtomicRawArray(os.path.join(root, "canary-1m.i8"), dtype="i1", shape=(rows, DIMENSION))
    scales = AtomicRawArray(os.path.join(root, "canary-1m-scales.f16"), dtype="<f2", shape=(rows,))
    cosines: list[float] = []
    written = 0
    for source in source_plan["sources"]:
        for _local, block, _ in _read_source_blocks(source, block_rows=block_rows):
            if written >= rows:
                break
            take = min(rows - written, len(block))
            q, s = _quantize_block(block[:take])
            int8.write(q)
            scales.write(s)
            cosines.extend(float(v) for v in _cosine_from_quantized(block[:take], q, s)[:100])
            written += take
        if written >= rows:
            break
    if written != rows:
        raise Round0025Error(f"{ROUND_LABEL} canary could only build {written} rows")
    return {
        "rows": rows,
        "dimension": DIMENSION,
        "int8": int8.publish(),
        "scales": scales.publish(),
        "cosine": _sample_summary(cosines),
    }


def _posix_dontneed(path: str) -> dict[str, Any]:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return {"attempted": False, "reason": "posix_fadvise unavailable"}
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return {"attempted": True, "advice": "POSIX_FADV_DONTNEED"}


def _load_arrays(universe: dict[str, Any], *, residency: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = int(universe["rows"])
    int8_path = universe["int8"]["canonical_path"]
    scales_path = universe["scales"]["canonical_path"]
    if residency == "ram":
        started = time.monotonic()
        x = np.fromfile(int8_path, dtype=np.int8, count=rows * DIMENSION).reshape(rows, DIMENSION)
        scales = np.fromfile(scales_path, dtype="<f2", count=rows)
        return x, scales, {"residency": "ram", "load_wall_s": round(time.monotonic() - started, 6)}
    if residency == "nvme":
        advice = {"int8": _posix_dontneed(int8_path), "scales": _posix_dontneed(scales_path)}
        x = np.memmap(int8_path, dtype=np.int8, mode="r", shape=(rows, DIMENSION))
        scales = np.memmap(scales_path, dtype="<f2", mode="r", shape=(rows,))
        return x, scales, {"residency": "nvme", "fadvise": advice}
    raise ValueError(residency)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def run_bench_cell(
    *,
    name: str,
    universe: dict[str, Any],
    residency: str,
    iterations: int,
    warmup: int,
    batch_rows: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    rows = int(universe["rows"])
    device = torch.device("cuda")
    x, scales, residency_info = _load_arrays(universe, residency=residency)
    host_i8 = [
        torch.empty((batch_rows, DIMENSION), dtype=torch.int8, pin_memory=True)
        for _ in range(2)
    ]
    host_scales = [
        torch.empty((batch_rows,), dtype=torch.float16, pin_memory=True)
        for _ in range(2)
    ]
    host_i8_np = [buffer.numpy() for buffer in host_i8]
    host_scales_np = [buffer.numpy() for buffer in host_scales]
    rng = np.random.RandomState(seed)
    gather_ms: list[float] = []
    iter_ms: list[float] = []
    gpu_stage_ms: list[float] = []
    window_counts = [0, 0, 0, 0, 0]
    window_seconds = [0.0, 0.0, 0.0, 0.0, 0.0]
    checksum = 0.0

    def fill_buffer(slot: int) -> float:
        indices = rng.randint(0, rows, size=batch_rows, dtype=np.int64)
        started_gather = time.perf_counter()
        host_i8_np[slot][...] = x[indices]
        host_scales_np[slot][...] = scales[indices]
        return (time.perf_counter() - started_gather) * 1000.0

    fill_buffer(0)
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    total = warmup + iterations
    for step in range(total):
        current = step & 1
        next_slot = 1 - current
        t_iter = time.perf_counter()
        start_event.record()
        dev_i8 = host_i8[current].to(device=device, non_blocking=True)
        dev_scales = host_scales[current].to(device=device, non_blocking=True)
        values = dev_i8.to(torch.bfloat16) * dev_scales.to(torch.bfloat16).view(-1, 1)
        stop_event.record()
        next_gather_ms = fill_buffer(next_slot)
        torch.cuda.synchronize(device)
        t_done = time.perf_counter()
        gpu_elapsed_ms = float(start_event.elapsed_time(stop_event))
        if step >= warmup:
            elapsed = t_done - t_iter
            bucket = min(4, int((step - warmup) * 5 / iterations))
            window_counts[bucket] += 1
            window_seconds[bucket] += elapsed
            gather_ms.append(next_gather_ms)
            iter_ms.append(elapsed * 1000.0)
            gpu_stage_ms.append(gpu_elapsed_ms)
            checksum += float(host_i8_np[next_slot][0, 0]) * float(
                host_scales_np[next_slot][0])
    wall = time.perf_counter() - started
    rates = [
        (count / seconds if seconds > 0 else 0.0)
        for count, seconds in zip(window_counts, window_seconds)
    ]
    bytes_per_batch = batch_rows * (DIMENSION + 2)
    median_rate = float(statistics.median(rates))
    result = {
        "name": name,
        "universe_rows": rows,
        "residency": residency_info,
        "loop_shape": {
            "batch_rows": batch_rows,
            "dimension": DIMENSION,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "prefetch_buffers": 2,
            "operation": (
                "uniform-row-gather-int8-to-pinned-host-overlapped-with-current-"
                "h2d-dequant-bf16"
            ),
        },
        "batches_per_s_median_of_5_windows": median_rate,
        "batches_per_s_windows": rates,
        "gather_latency_ms": {
            "p50": _percentile(gather_ms, 0.50),
            "p99": _percentile(gather_ms, 0.99),
        },
        "iteration_latency_ms": {
            "p50": _percentile(iter_ms, 0.50),
            "p99": _percentile(iter_ms, 0.99),
        },
        "gpu_stage_ms": {
            "p50": _percentile(gpu_stage_ms, 0.50),
            "p99": _percentile(gpu_stage_ms, 0.99),
            "fraction_of_iteration_p50": (
                _percentile(gpu_stage_ms, 0.50) / _percentile(iter_ms, 0.50)
                if _percentile(iter_ms, 0.50) > 0 else None
            ),
        },
        "effective_transfer_gb_s_at_median_rate": median_rate * bytes_per_batch / 1e9,
        "host_rss_gb": _rss_gb(),
        "wall_seconds_including_warmup": wall,
        "checksum": checksum,
    }
    del x, scales, host_i8, host_scales
    torch.cuda.empty_cache()
    return result


def _rss_gb() -> float:
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2)
    except Exception:
        return float("nan")


def run_benchmark(
    shard_manifest: dict[str, Any],
    output_root: str,
    *,
    iterations: int = 20_000,
    warmup: int = 500,
    batch_rows: int = 8192,
) -> dict[str, Any]:
    root = create_fresh_directory(output_root, label=f"{ROUND_LABEL} bench output")
    universes = shard_manifest["universes"]
    started = time.monotonic()
    cells = []
    free_ram_gb = _available_ram_gb()
    if free_ram_gb < 70.0:
        cells.append({
            "name": "ram-150m",
            "skipped": True,
            "reason": f"free RAM {free_ram_gb:.2f} GiB < 70 GiB",
        })
    else:
        cells.append(run_bench_cell(
            name="ram-150m", universe=universes["minilm-int8-150m"],
            residency="ram", iterations=iterations, warmup=warmup,
            batch_rows=batch_rows, seed=250150))
    cells.append(run_bench_cell(
        name="nvme-150m-cold", universe=universes["minilm-int8-150m"],
        residency="nvme", iterations=iterations, warmup=warmup,
        batch_rows=batch_rows, seed=250151))
    cells.append(run_bench_cell(
        name="nvme-405m", universe=universes["minilm-int8-405m"],
        residency="nvme", iterations=iterations, warmup=warmup,
        batch_rows=batch_rows, seed=250405))
    by_name = {cell["name"]: cell for cell in cells}
    ram_rate = by_name.get("ram-150m", {}).get("batches_per_s_median_of_5_windows")
    nvme405_rate = by_name.get("nvme-405m", {}).get("batches_per_s_median_of_5_windows")
    body = {
        "schema": "round0025-int8-gather-throughput-v1",
        "shard_manifest": {
            "canonical_path": shard_manifest["canonical_path"],
            "sha256": shard_manifest["sha256"],
            "identity_sha256": shard_manifest["identity_sha256"],
        },
        "cells": cells,
        "decision": {
            "ram_150m_gpu_bound_threshold_batches_per_s": 115.0,
            "ram_150m_gpu_bound": (
                bool(ram_rate is not None and ram_rate >= 115.0)
                if isinstance(ram_rate, (int, float)) else None
            ),
            "nvme_405m_withdraw_threshold_batches_per_s": 50.0,
            "nvme_405m_below_50": (
                bool(nvme405_rate is not None and nvme405_rate < 50.0)
                if isinstance(nvme405_rate, (int, float)) else None
            ),
        },
        "quantization_sanity": shard_manifest["quantization_sanity"],
        "wall_seconds": round(time.monotonic() - started, 6),
    }
    return _json(os.path.join(root, "int8-gather-throughput-v1.json"), body)


def _available_ram_gb() -> float:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024 ** 2)
    return 0.0


def _disk_free_gib(path: str) -> float:
    return float(shutil.disk_usage(path).free) / (1024 ** 3)


def run_canary(source_plan: dict[str, Any], output_root: str) -> dict[str, Any]:
    slice_manifest = _reserve_small_quantized_slice(
        source_plan, output_root, rows=CANARY_ROWS)
    shard_manifest = {
        "canonical_path": os.path.join(output_root, "canary-shards.json"),
        "sha256": "canary-inline",
        "identity_sha256": "canary-inline",
        "universes": {
            "minilm-int8-150m": {
                "rows": slice_manifest["rows"],
                "int8": slice_manifest["int8"],
                "scales": slice_manifest["scales"],
            },
            "minilm-int8-405m": {
                "rows": slice_manifest["rows"],
                "int8": slice_manifest["int8"],
                "scales": slice_manifest["scales"],
            },
        },
        "quantization_sanity": {
            "universe": "canary-1m",
            "cosine": slice_manifest["cosine"],
            "diagnostic_threshold": 0.999,
            "threshold_passed": slice_manifest["cosine"]["minimum"] >= 0.999,
        },
    }
    bench = run_bench_cell(
        name="canary-1m", universe=shard_manifest["universes"]["minilm-int8-150m"],
        residency="nvme", iterations=100, warmup=10, batch_rows=8192, seed=250025)
    body = {
        "schema": "round0025-canary-v1",
        "source_plan_identity_sha256": source_plan["identity_sha256"],
        "source_policy": {
            "row_policy": source_plan["row_policy"],
            "trailing_fragment_policy": source_plan["trailing_fragment_policy"],
            "trailing_fragments": source_plan["trailing_fragments"],
        },
        "slice": slice_manifest,
        "bench": bench,
        "passed": bool(slice_manifest["cosine"]["minimum"] >= 0.999 and bench[
            "batches_per_s_median_of_5_windows"] > 0),
    }
    return _json(os.path.join(output_root, "verdict.json"), body)


def _load_manifest(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_source_plan_from_queue(queue_manifest: str) -> dict[str, Any]:
    manifest = _load_manifest(queue_manifest)
    plan = manifest.get("round0025", {}).get("source_plan")
    if not isinstance(plan, dict) or plan.get("schema") != "round0025-source-plan-v1":
        raise Round0025Error(f"{ROUND_LABEL} queue lacks source plan")
    return plan


def _attach_file_signature(path: str, receipt: dict[str, Any]) -> dict[str, Any]:
    canonical = os.path.realpath(path)
    with open(canonical, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {
        **receipt,
        "canonical_path": canonical,
        "bytes": os.path.getsize(canonical),
        "sha256": digest,
    }


def run_node_command(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_source_plan_from_queue(args.queue_manifest)
    if args.command == "canary":
        return run_canary(plan, args.output)
    if args.command == "quantize":
        result = quantize_full(plan, args.output, block_rows=args.block_rows)
        manifest_path = os.path.join(args.output, "int8-shards-v1.json")
        return _attach_file_signature(manifest_path, result)
    if args.command == "bench":
        shard_manifest = _load_manifest(args.shard_manifest)
        shard_manifest = _attach_file_signature(args.shard_manifest, shard_manifest)
        return run_benchmark(
            shard_manifest, args.output, iterations=args.iterations,
            warmup=args.warmup, batch_rows=args.batch_rows)
    raise AssertionError(args.command)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("source-plan")
    plan.add_argument("--no-hash", action="store_true")
    plan.add_argument("--out")
    for name in ("canary", "quantize"):
        command = sub.add_parser(name)
        command.add_argument("--queue-manifest", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--block-rows", type=int, default=65536)
    bench = sub.add_parser("bench")
    bench.add_argument("--queue-manifest", required=True)
    bench.add_argument("--shard-manifest", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--iterations", type=int, default=20_000)
    bench.add_argument("--warmup", type=int, default=500)
    bench.add_argument("--batch-rows", type=int, default=8192)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "source-plan":
        result = build_source_plan(hash_files=not args.no_hash)
        if args.out:
            _json(args.out, {k: v for k, v in result.items() if k != "identity_sha256"})
        else:
            print(json.dumps(result, sort_keys=True, indent=1))
        return 0
    result = run_node_command(args)
    print(json.dumps(result, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Round0025Error as error:
        print(f"ROUND0025_FAIL_CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
