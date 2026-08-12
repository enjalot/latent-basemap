"""R0262 — the host-int8 X ↔ weighted fuzzy edge sampler adapter.

review-0259-01 §D established the 100M rung's *only* remaining blocker, after
correcting R0259's headline:

    "no host-resident-X path is wired to the weighted fuzzy edge sampler"

Three pipelines exist in ``core._build_edge_list_loader`` and none of them can
run the 100M rung under the registered weighted-fuzzy recipe:

===================  ==========================  ============  ==============
pipeline             sampler                     X residency   weighted_ok
===================  ==========================  ============  ==============
``device``           ``DeviceEdgeSampler``       device fp16   True
``hybrid``           ``HostStreamEdgeSampler``   device fp16   True
``legacy``           ``EdgeListBalancedIterator``host memmap   **False**
===================  ==========================  ============  ==============

Both weighted-capable pipelines put X *on the card*.  At the 100M rung X is
``100_000_000 x 384``; even at int8 that is ``38_400_000_000`` B against a
``33_679_736_832`` B card, so neither can be constructed (review-0259-01 §C.1).
The one host-X pipeline, ``legacy``, is stamped ``weighted_ok=False`` and
``core.py:621`` raises rather than silently downgrade a weighted request to a
uniform sampler.  That raise is correct and stays.

This module supplies the missing fourth row:

===================  ==========================  ============  ==============
``host_int8_hybrid`` ``HostStreamEdgeSampler``   host int8     True
===================  ==========================  ============  ==============

**It builds no parallel array class.**  ``HostInt8MaterializedArray``
(``round0034_pipeline.py:725``) already holds host int8 rows plus exact per-row
fp16 scales behind two pinned double-buffered slot pairs with CUDA-event-guarded
reuse, and already exposes ``gather_pairs`` — a *fused* both-endpoint gather that
fills one slot on the host and issues one H2D per endpoint.  It has been
instantiated at ``row_count=150_000_000`` since 2026-07-20.  The wiring needed
was two edits, neither of them a new array type; see ``WIRING`` below.

What this module adds is the surrounding evidence: the exact int8 encoder used to
build the substrate artifact, full-substrate fidelity accounting, and the
backing/accounting guards the mandate requires.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "Round0262Error",
    "Round0262BackingError",
    "Round0262BudgetError",
    "Round0262WiringError",
    "WIRING",
    "INT8_DIVISOR",
    "SUBSTRATE_100M_PATH",
    "SUBSTRATE_100M_SHAPE",
    "INT8_100M_DIR",
    "INT8_100M_PATH",
    "SCALES_100M_PATH",
    "classify_host_backing",
    "host_memory_accounting",
    "assert_host_headroom",
    "quantise_block_int8",
    "dequantise_block",
    "quantise_substrate_to_int8",
    "build_host_int8_source",
    "assert_buffer_admits_batch",
]


class Round0262Error(RuntimeError):
    """Base class for R0262 refusals."""


class Round0262BackingError(Round0262Error):
    """A host array is not backed the way the caller declared."""


class Round0262BudgetError(Round0262Error):
    """A host allocation would not fit the declared headroom."""


class Round0262WiringError(Round0262Error):
    """The host-int8 pipeline was selected but cannot be served."""


#: The two edits that wire the pipeline, named so a reader can diff exactly
#: these and nothing else. Both are additive branches keyed on a marker that
#: already existed (``HostInt8MaterializedArray.round0034_host_int8``), so no
#: existing rung changes pipeline, sampler, residency or RNG stream.
WIRING = {
    "core": {
        "file": "basemap/pumap/parametric_umap/core.py",
        "symbol": "ParametricUMAP._build_edge_list_loader",
        "shape": (
            "one additive branch, taken only when X advertises "
            "`round0034_host_int8`, that stamps pipeline='host_int8_hybrid' "
            "with weighted_ok=True and hands the host-int8 X straight to "
            "HostStreamEdgeSampler"
        ),
        "existing_paths_touched": 0,
    },
    "sampler": {
        "file": "basemap/pumap/parametric_umap/datasets/edge_list_dataset.py",
        "symbol": "HostStreamEdgeSampler.__next__",
        "shape": (
            "one additive branch that calls dataset.gather_pairs(src, dst) once "
            "instead of dataset.index_select twice; the else-branch is the "
            "unchanged device path"
        ),
        "existing_paths_touched": 0,
    },
    "round0034_pipeline": {
        "file": "basemap/round0034_pipeline.py",
        "symbol": "HostInt8MaterializedArray",
        "shape": "UNCHANGED — gather_pairs and the round0034_host_int8 marker already existed",
        "existing_paths_touched": 0,
    },
}

#: int8 symmetric per-row encoding: ``q = round(x / scale)`` with
#: ``scale = max|x| / 127``.  127 (not 128) keeps the code symmetric so ``-128``
#: is never produced and ``|q| <= 127`` always holds.
INT8_DIVISOR = 127.0

SUBSTRATE_100M_PATH = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy"
)
SUBSTRATE_100M_SHAPE = (100_000_000, 384)

INT8_100M_DIR = (
    "/data/latent-basemap/runs/round-0262/artifacts/minilm-mixed-100m-int8-v1"
)
INT8_100M_PATH = os.path.join(INT8_100M_DIR, "substrate.i8")
SCALES_100M_PATH = os.path.join(INT8_100M_DIR, "substrate-scales.f16")


# --------------------------------------------------------------------------- #
# 1. backing and accounting — "check the backing, not the label"
# --------------------------------------------------------------------------- #

def classify_host_backing(array: Any) -> dict[str, Any]:
    """Classify a host array as anonymous or file-backed by its **base chain**.

    The mandate's third named defect: ``HostStreamEdgeSampler`` does

        self._src_h = np.ascontiguousarray(np.asarray(sources), dtype=np.int32)

    intending a host-resident copy.  When ``sources`` is already a C-contiguous
    int32 ``np.memmap`` that call **copies nothing** — it returns a plain
    ``ndarray`` view whose ``.base`` chain still ends at the ``mmap``.  So:

    * the bytes are page cache (file-backed, evictable, ``RssFile``), not
      anonymous RAM, and
    * ``isinstance(x, np.memmap)`` is **False**, because the returned view is a
      base ``ndarray`` — an isinstance reader rule calls it host-resident.

    This walks ``.base`` to the bottom and reports what is actually there, plus
    what the isinstance rule *would* have said, so the two can be compared.
    """
    chain: list[str] = []
    mmap_depth = -1
    node = array
    depth = 0
    while node is not None and depth < 64:
        chain.append(type(node).__name__)
        if isinstance(node, np.memmap) and mmap_depth < 0:
            mmap_depth = depth
        if type(node).__name__ == "mmap" and mmap_depth < 0:
            mmap_depth = depth
        node = getattr(node, "base", None)
        depth += 1
    file_backed = "mmap" in chain or mmap_depth >= 0
    return {
        "base_chain": chain,
        "file_backed": bool(file_backed),
        "backing": "file_backed_page_cache" if file_backed else "anonymous",
        "isinstance_np_memmap": bool(isinstance(array, np.memmap)),
        # The disagreement the mandate asked to be shown, not asserted.
        "isinstance_rule_agrees_with_base_chain": bool(
            isinstance(array, np.memmap) == file_backed
        ),
        "nbytes": int(getattr(array, "nbytes", 0)),
    }


def _proc_status() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/self/status", "r") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[-1] == "kB":
                out[key.strip()] = int(parts[0]) * 1024
    return out


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/meminfo", "r") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[-1] == "kB":
                out[key.strip()] = int(parts[0]) * 1024
    return out


def host_memory_accounting() -> dict[str, int]:
    """The three accountings a 38.4 GB host allocation can land in.

    ``RssAnon`` sees anonymous private pages (``np.empty``, ``np.fromfile``).
    ``RssFile`` sees mapped file pages (``np.load(mmap_mode='r')`` and every
    zero-copy view of one).  ``RssShmem`` sees ``MAP_SHARED`` pages, which is
    where ``load_array_polled``'s staging lands and why no anonymous guard sees
    it.  A guard that watches only ``RssAnon`` is blind to two of the three.
    """
    status = _proc_status()
    info = _meminfo()
    return {
        "RssAnon": status.get("RssAnon", 0),
        "RssFile": status.get("RssFile", 0),
        "RssShmem": status.get("RssShmem", 0),
        "VmRSS": status.get("VmRSS", 0),
        "VmHWM": status.get("VmHWM", 0),
        "MemAvailable": info.get("MemAvailable", 0),
        "MemFree": info.get("MemFree", 0),
        "SwapFree": info.get("SwapFree", 0),
        "SwapTotal": info.get("SwapTotal", 0),
        "Cached": info.get("Cached", 0),
    }


def assert_host_headroom(need_bytes: int, *, margin_bytes: int, where: str) -> dict[str, Any]:
    """Refuse a large host allocation that would drive this box toward swap.

    R0261 runs concurrently and also needs host memory, so this reads
    ``MemAvailable`` at the moment of the call rather than trusting a figure
    measured earlier in the node.
    """
    accounting = host_memory_accounting()
    available = accounting["MemAvailable"]
    admits = available >= (need_bytes + margin_bytes)
    receipt = {
        "where": where,
        "need_bytes": int(need_bytes),
        "margin_bytes": int(margin_bytes),
        "mem_available_bytes": int(available),
        "admits": bool(admits),
        "accounting_before": accounting,
    }
    if not admits:
        raise Round0262BudgetError(
            f"R0262 {where}: need {need_bytes} B + {margin_bytes} B margin but "
            f"MemAvailable is {available} B. Refusing to allocate — a 38 GB "
            f"allocation that pushes this 119 GB host into swap costs far more "
            f"than the round."
        )
    return receipt


# --------------------------------------------------------------------------- #
# 2. the int8 encoder
# --------------------------------------------------------------------------- #

def quantise_block_int8(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode fp32 rows as int8 + exact per-row fp16 scale.

    Symmetric per-row max-abs, the encoding ``HostInt8MaterializedArray``
    dequantises (``round0034_pipeline.py:903``: ``i8.float() * scale.float()``).
    The scale is **stored** as fp16 and the encoding divides by the fp16 value,
    not by the fp32 one, so the round-trip carries no scale-representation error
    beyond what the stored scale itself implies.
    """
    block = np.asarray(block, dtype=np.float32)
    absmax = np.abs(block).max(axis=1)
    # A genuinely all-zero row has no scale; give it the smallest positive fp16
    # normal so `scales > 0` (HostInt8MaterializedArray refuses scales <= 0) and
    # the row still dequantises to exactly zero.
    absmax = np.where(absmax > 0, absmax, np.float32(6.104e-5))
    scales = (absmax / np.float32(INT8_DIVISOR)).astype(np.float16)
    scales = np.where(scales > 0, scales, np.float16(6.0e-8)).astype(np.float16)
    encoded = np.rint(block / scales.astype(np.float32)[:, None])
    np.clip(encoded, -127, 127, out=encoded)
    return encoded.astype(np.int8), scales


def dequantise_block(encoded: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """The exact inverse the device performs, computed on the host in fp32."""
    return encoded.astype(np.float32) * np.asarray(scales, dtype=np.float32)[:, None]


class _FidelityAccumulator:
    """Exact streaming fidelity statistics over every row of the substrate.

    Nothing here is sampled: the reduction runs on the same block the encoder
    just produced, so the published error is over all ``N`` rows, not a subset.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.sq_err_sum = 0.0        # sum ||x - x_hat||^2
        self.sq_ref_sum = 0.0        # sum ||x||^2
        self.cos_sum = 0.0
        self.cos_min = 1.0
        self.rel_l2_max = 0.0
        self.rel_l2_sum = 0.0
        self.abs_err_max = 0.0
        self.clipped = 0
        self.zero_rows = 0

    def update(self, block: np.ndarray, encoded: np.ndarray, scales: np.ndarray) -> None:
        ref = np.asarray(block, dtype=np.float64)
        hat = dequantise_block(encoded, scales).astype(np.float64)
        diff = hat - ref
        err = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        ref_norm = np.sqrt(np.einsum("ij,ij->i", ref, ref))
        safe = np.where(ref_norm > 0, ref_norm, 1.0)
        rel = err / safe
        dot = np.einsum("ij,ij->i", ref, hat)
        hat_norm = np.sqrt(np.einsum("ij,ij->i", hat, hat))
        cos = dot / np.where(ref_norm * hat_norm > 0, ref_norm * hat_norm, 1.0)

        self.rows += int(len(ref))
        self.sq_err_sum += float((err ** 2).sum())
        self.sq_ref_sum += float((ref_norm ** 2).sum())
        self.cos_sum += float(cos.sum())
        self.cos_min = min(self.cos_min, float(cos.min()))
        self.rel_l2_max = max(self.rel_l2_max, float(rel.max()))
        self.rel_l2_sum += float(rel.sum())
        self.abs_err_max = max(self.abs_err_max, float(np.abs(diff).max()))
        self.clipped += int((np.abs(encoded) >= 127).sum())
        self.zero_rows += int((ref_norm == 0).sum())

    def report(self) -> dict[str, Any]:
        rows = max(self.rows, 1)
        return {
            "rows": int(self.rows),
            "relative_l2_error_mean": self.rel_l2_sum / rows,
            "relative_l2_error_max": self.rel_l2_max,
            "aggregate_relative_l2_error": (
                (self.sq_err_sum / self.sq_ref_sum) ** 0.5 if self.sq_ref_sum else 0.0
            ),
            "cosine_similarity_mean": self.cos_sum / rows,
            "cosine_similarity_min": self.cos_min,
            "absolute_component_error_max": self.abs_err_max,
            "components_at_the_clip": int(self.clipped),
            "all_zero_rows": int(self.zero_rows),
        }


def quantise_substrate_to_int8(
    *,
    source_path: str = SUBSTRATE_100M_PATH,
    expected_shape: tuple[int, int] = SUBSTRATE_100M_SHAPE,
    int8_path: str = INT8_100M_PATH,
    scales_path: str = SCALES_100M_PATH,
    rows_per_chunk: int = 1_000_000,
    poll: Callable[[str], Any] | None = None,
    row_limit: int | None = None,
) -> dict[str, Any]:
    """Stream the fp32 substrate to an int8 + fp16-scale artifact.

    Chunk-polled: ``poll`` is called between chunks so the cooperative abort
    flag is observed on this stage.  Fidelity is accumulated on the block that
    is already in registers, so full-substrate error accounting is free.
    """
    poll = poll or (lambda _where: None)
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if tuple(source.shape) != tuple(expected_shape) or source.dtype != np.dtype("float32"):
        raise Round0262WiringError(
            f"R0262: substrate at {source_path} is {source.shape}/{source.dtype}, "
            f"expected {expected_shape}/float32"
        )
    n_rows = int(expected_shape[0]) if row_limit is None else int(row_limit)
    dimension = int(expected_shape[1])
    os.makedirs(os.path.dirname(int8_path), exist_ok=True)

    accumulator = _FidelityAccumulator()
    chunks = 0
    started = time.monotonic()
    read_s = 0.0
    encode_s = 0.0
    write_s = 0.0
    with open(int8_path, "wb") as i8_handle, open(scales_path, "wb") as scale_handle:
        for start in range(0, n_rows, rows_per_chunk):
            poll(f"quantise_substrate chunk {chunks}")
            end = min(start + rows_per_chunk, n_rows)
            mark = time.monotonic()
            block = np.array(source[start:end], dtype=np.float32, copy=True)
            read_s += time.monotonic() - mark
            mark = time.monotonic()
            encoded, scales = quantise_block_int8(block)
            accumulator.update(block, encoded, scales)
            encode_s += time.monotonic() - mark
            mark = time.monotonic()
            i8_handle.write(encoded.tobytes())
            scale_handle.write(scales.astype("<f2").tobytes())
            write_s += time.monotonic() - mark
            chunks += 1
    wall = time.monotonic() - started

    receipt = {
        "source_path": source_path,
        "int8_path": int8_path,
        "scales_path": scales_path,
        "rows": n_rows,
        "dimension": dimension,
        "rows_per_chunk": int(rows_per_chunk),
        "chunks": chunks,
        "poll_sites": chunks,
        "int8_bytes": int(os.path.getsize(int8_path)),
        "scales_bytes": int(os.path.getsize(scales_path)),
        "expected_int8_bytes": n_rows * dimension,
        "expected_scales_bytes": n_rows * 2,
        "wall_s": wall,
        "read_s": read_s,
        "encode_s": encode_s,
        "write_s": write_s,
        "fidelity_over_every_row": accumulator.report(),
    }
    if receipt["int8_bytes"] != receipt["expected_int8_bytes"]:
        raise Round0262WiringError(
            f"R0262: int8 artifact is {receipt['int8_bytes']} B, expected "
            f"{receipt['expected_int8_bytes']} B"
        )
    if receipt["scales_bytes"] != receipt["expected_scales_bytes"]:
        raise Round0262WiringError(
            f"R0262: scales artifact is {receipt['scales_bytes']} B, expected "
            f"{receipt['expected_scales_bytes']} B"
        )
    return receipt


# --------------------------------------------------------------------------- #
# 3. constructing the host-int8 X
# --------------------------------------------------------------------------- #

def build_host_int8_source(
    *,
    int8_path: str,
    scales_path: str,
    row_count: int,
    dimension: int,
    device: str,
    buffer_rows: int,
    poll: Callable[[str], Any] | None = None,
    rows_per_chunk: int = 4_000_000,
):
    """Load the int8 artifact into **anonymous** host RAM and wrap it.

    Deliberately not ``HostInt8MaterializedArray.from_files``: that helper
    sha256s both members (``expected_input_signature``), which is ~58 GB of
    hashing at the 150M rung and ~38.6 GB here, and it reads with a single
    unpolled ``np.fromfile`` — no abort flag is observed for the whole read.
    This reads in polled chunks into one preallocated anonymous array and
    reports the backing it actually got.
    """
    from .round0034_pipeline import HostInt8MaterializedArray

    poll = poll or (lambda _where: None)
    need = row_count * dimension + row_count * 2
    headroom = assert_host_headroom(
        need, margin_bytes=12_000_000_000, where="build_host_int8_source"
    )
    before = host_memory_accounting()

    started = time.monotonic()
    encoded = np.empty((row_count, dimension), dtype=np.int8)
    with open(int8_path, "rb") as handle:
        for start in range(0, row_count, rows_per_chunk):
            poll(f"build_host_int8_source rows {start}")
            end = min(start + rows_per_chunk, row_count)
            want = (end - start) * dimension
            got = np.fromfile(handle, dtype=np.int8, count=want)
            if got.size != want:
                raise Round0262WiringError(
                    f"R0262: int8 artifact short read at row {start}: "
                    f"{got.size} of {want} bytes"
                )
            encoded[start:end] = got.reshape(end - start, dimension)
    scales = np.fromfile(scales_path, dtype="<f2", count=row_count)
    if scales.size != row_count:
        raise Round0262WiringError(
            f"R0262: scales artifact holds {scales.size} rows, expected {row_count}"
        )
    load_wall = time.monotonic() - started

    encoded_backing = classify_host_backing(encoded)
    scales_backing = classify_host_backing(scales)
    if encoded_backing["file_backed"] or scales_backing["file_backed"]:
        raise Round0262BackingError(
            "R0262: the host-int8 X must be anonymous RAM; the base chain says "
            f"encoded={encoded_backing['base_chain']} scales={scales_backing['base_chain']}"
        )

    source = HostInt8MaterializedArray(
        encoded, scales, device=device, buffer_rows=buffer_rows
    )
    after = host_memory_accounting()
    receipt = {
        "int8_path": int8_path,
        "scales_path": scales_path,
        "row_count": int(row_count),
        "dimension": int(dimension),
        "buffer_rows": int(buffer_rows),
        "planned_bytes": int(need),
        "load_wall_s": load_wall,
        "headroom": headroom,
        "encoded_backing": encoded_backing,
        "scales_backing": scales_backing,
        "accounting_before": before,
        "accounting_after": after,
        "rss_anon_delta_bytes": after["RssAnon"] - before["RssAnon"],
        "rss_file_delta_bytes": after["RssFile"] - before["RssFile"],
        "rss_shmem_delta_bytes": after["RssShmem"] - before["RssShmem"],
        # The mandate's question, answered as a field rather than as prose.
        "which_accounting_sees_the_allocation": "RssAnon",
    }
    return source, receipt


def assert_buffer_admits_batch(source: Any, batch_size: int) -> dict[str, Any]:
    """A host-int8 X must be able to stage one full endpoint batch.

    ``HostInt8MaterializedArray._validated_pair_rows`` refuses a gather wider
    than ``buffer_rows``.  ``HostStreamEdgeSampler`` gathers ``batch_size`` rows
    per endpoint, so a too-small buffer would raise on the **first** update of a
    10 h fit rather than at construction.  Fail here instead.
    """
    buffer_rows = int(getattr(source, "buffer_rows", 0))
    if buffer_rows < int(batch_size):
        raise Round0262WiringError(
            f"R0262: host-int8 X stages {buffer_rows} rows per slot but the "
            f"sampler gathers {batch_size} rows per endpoint. Construct the "
            f"array with buffer_rows >= batch_size."
        )
    return {"buffer_rows": buffer_rows, "batch_size": int(batch_size), "admits": True}
