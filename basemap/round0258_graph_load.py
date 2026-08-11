"""R0258 — the 100M graph load and edge preparation, measured and polled.

This is the last unmeasured interval between the program and a stoppable 100M
training node. R0253 §D3 named it and refused to carry its own `2M` figure up::

    Not met: the graph load and edge-list preparation at the 100M rung.
    R0238's 100,000,000-row substrate and R0240's k15 graph no longer exist on
    this box, and rebuilding them is ~7 GPU-h against this round's 1.0 cap.

**The premise is false and this module exists because it is false.** All three
artifacts are on the box, fully allocated, at their sealed R0240/R0243
signatures, and have been since they were built:

* `gsv:/data/latent-basemap/runs/round-0238/queue/artifacts/
  minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy`
  — `153,600,000,128` B, mtime `2026-08-09`;
* `gsv:/data/latent-basemap/runs/round-0240/queue/artifacts/
  minilm-mixed-100000k-cluster-spill-build-ladder-v1/builds/
  r0238-n100000000-c400-s8/graph-k15-{ids,cos}.*.npy` — `6,000,000,128` B each,
  mtime `2026-08-10`;
* R0243's fuzzy edge triple below — `10,044,413,144` B each, mtime `2026-08-10`.

So the measurement costs no rebuild and no `~7` GPU-h. It costs the read.

What is measured
----------------
The sequence a 100M training node performs between "the graph file exists" and
"the sampler is constructed", at the real rung, over the real bytes:

===  =========================================  ===================================
S1   `load` src/dst/wts                         `edge_list_dataset.load_edge_arrays`
S2   `bounds` min/max of both endpoint arrays   `graph_validation.validate_edge_bounds`
S3   `validity` isfinite / <=0 / >1 on weights  `round0107_training.DiverseWeightedJinaSampler.__init__`
S4   `contiguous` int32 host copies             `edge_list_dataset.HostStreamEdgeSampler.__init__`
S5   `cdf` float64 weight CDF + cumsum          `edge_list_dataset.HostStreamEdgeSampler.__init__`
===  =========================================  ===================================

Each stage exists in the release today, unpolled, and each is a single numpy or
`np.load` call over `2,511,103,254` elements. The unpolled arm runs them exactly
as shipped; the two polled arms run chunked reimplementations that are required
to be **bitwise identical**, not merely close.

A structural finding that falls out of S1 and is not a timing
-------------------------------------------------------------
`load_edge_arrays(path)` indexes an `.npz`. Every rung at or below 50M ships one
bulk `edges-k15-fuzzy.npz`; the 100M rung ships three streamed `.npy` members
plus a header, deliberately (`round0243_nodes.py:962`). **So the shipped trainer
cannot open the 100M graph at all** — `round0113_nodes.run_train` ->
`round0113_prompt_contrast.load_graph` -> `load_edge_arrays` has no 100M path.
What this module measures for S1 is therefore the loader this round adds, and
the `unpolled_control` arm is `np.load` of the same three members with no poll —
the shape the `.npz` path would have had.

Bitwise identity, and why it is the whole safety argument
---------------------------------------------------------
A poll can only be installed inside a numpy call by cutting the call into
chunks, and cutting a floating-point reduction into chunks changes its value.
Three of the five stages are exact under chunking for structural reasons
(a copy is a copy; `min` and `max` are associative and exact; `all`/`any` over
booleans are associative). Two are not, and are handled by construction:

* `np.cumsum` is a strictly sequential recurrence, so a chunked cumsum that adds
  the carry to the **first element of the chunk before** cumsum-ing it issues
  the identical sequence of floating-point additions in the identical order.
  Adding the carry *after* the chunk's cumsum does not, and is one of the
  planted controls.
* `np.sum` uses pairwise summation, which chunking does **not** preserve.
  It is therefore left as one call and reported as an irreducible unpollable
  residue rather than silently blocked.

Every polled stage's output is compared to the shipped stage's output with
`np.array_equal` over the raw bytes, and `bitwise_identity_controls()` plants
five numeric defects that this comparison must refuse and an `np.allclose`
comparison — the check a reasonable implementer would reach for — must accept.
That accept/refuse contrast is the R0254 standard: it proves each refusal is
doing work rather than passing vacuously.

Safety
------
Every bulk input is opened read-only. Nothing here signals any process, starts a
child, hands cuVS anything, or wraps a subprocess in a timeout. The node that
drives this module holds a live CUDA context on purpose — that is the case that
cannot be signalled and therefore the only case whose stop latency matters.
"""
from __future__ import annotations

import ast
import ctypes
import inspect
import mmap
import os
import textwrap
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from basemap.round0253_stop_hooks import registered_ceiling_s
from basemap.round0254_writeback import arm_schedule, interval_summary

ROUND_ID = "0258"


class Round0258GraphLoadError(RuntimeError):
    """R0258 fails closed."""


# --------------------------------------------------------------------------- #
# the artifact, at its sealed R0243 signature
# --------------------------------------------------------------------------- #

R0243_FUZZY_DIR = (
    "/data/latent-basemap/runs/round-0243/queue/artifacts/"
    "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1"
)

#: Directed entries in R0243's canonicalised fuzzy graph. Sealed in
#: `fuzzy-graph.json` as `directed_edges` and as `fuzzy.directed_edges`.
DIRECTED_EDGES = 2_511_103_254
ROWS = 100_000_000
K = 15

#: R0243's own `outputs` block, verbatim. Sizes are checked by the runner's
#: input preflight; the sha256 values are checked by this module's own reader
#: because the runner checks size only.
EDGE_ARRAYS: dict[str, dict[str, Any]] = {
    "sources": {
        "path": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-src.i32.npy"),
        "bytes": 10_044_413_144,
        "sha256": "dfb3f4c25f8024614e0738456fc02dd5ec8e5589020ea3eb2f9e977c7fca52be",
        "dtype": "int32",
    },
    "targets": {
        "path": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-dst.i32.npy"),
        "bytes": 10_044_413_144,
        "sha256": "cfd3011055e9f6f1a48f2c6897f3d4c1bfc4417fb59537fd31289563766d87ca",
        "dtype": "int32",
    },
    "weights": {
        "path": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-wts.f32.npy"),
        "bytes": 10_044_413_144,
        "sha256": "ea7cffb2c8db3bb27f1cd4890b6a9b8b0bb0facc58c2240c984119c38eb16be8",
        "dtype": "float32",
    },
}

EDGE_HEADER_PATH = os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-header.npz")

#: The read unit. 64 MiB is R0253's write unit, held fixed by R0254 after
#: review-0253-01 §D measured an 8 MiB unit and found it moved the median 11x,
#: the p99 ~700x and the maximum not at all. Reusing it here keeps the read and
#: write units comparable rather than introducing a second free variable.
READ_CHUNK_BYTES = 64 << 20

#: The scan unit for the numpy stages, in ELEMENTS. 16,777,216 int32 is 64 MiB;
#: the same count of float64 is 128 MiB, which is the largest single buffer any
#: polled stage touches between two abort reads.
SCAN_CHUNK_ELEMENTS = 16_777_216

ARM_UNPOLLED_CONTROL = "unpolled_control"
ARM_POLLED_BUFFERED = "polled_buffered"
ARM_POLLED_O_DIRECT = "polled_o_direct"

#: `o_direct` is review-0254-01 §B.1's corrected best write arm at `0.035147x`.
#: This path writes nothing, so the flush ladder does not apply to it; what
#: carries over is the read side of the same idea — bypass the page cache so the
#: 30 GB graph does not evict the 153.6 GB substrate the trainer is about to
#: memmap. It is shipped as an arm, not asserted.
SHIPPED_ARMS = (ARM_POLLED_BUFFERED, ARM_POLLED_O_DIRECT)
ARMS = (ARM_UNPOLLED_CONTROL,) + SHIPPED_ARMS

STAGE_LOAD = "load"
STAGE_BOUNDS = "bounds"
STAGE_VALIDITY = "validity"
STAGE_CONTIGUOUS = "contiguous"
STAGE_CDF = "cdf"
STAGES = (STAGE_LOAD, STAGE_BOUNDS, STAGE_VALIDITY, STAGE_CONTIGUOUS, STAGE_CDF)

ABORT_POLL_SITE_LOAD = "round0258_graph_load.load_array chunk"
ABORT_POLL_SITE_BOUNDS = "round0258_graph_load.endpoint_bounds chunk"
ABORT_POLL_SITE_VALIDITY = "round0258_graph_load.weight_validity chunk"
ABORT_POLL_SITE_CONTIGUOUS = "round0258_graph_load.contiguous_int32 chunk"
ABORT_POLL_SITE_CDF = "round0258_graph_load.weight_cdf chunk"
ABORT_POLL_SITES = (
    ABORT_POLL_SITE_LOAD,
    ABORT_POLL_SITE_BOUNDS,
    ABORT_POLL_SITE_VALIDITY,
    ABORT_POLL_SITE_CONTIGUOUS,
    ABORT_POLL_SITE_CDF,
)

WHAT_THE_UNPOLLED_ARM_IS = (
    "the unpolled arm is a pre-registered CONTROL and is built to fail. It runs "
    "the five stages exactly as the release ships them -- np.load, arr.min(), "
    "arr.max(), np.isfinite(w).all(), np.any(w <= 0), np.any(w > 1), "
    "np.ascontiguousarray, np.array(w, dtype=np.float64), w.sum(), "
    "np.cumsum(w, out=w), w /= total -- with no abort read anywhere inside any "
    "of them. Its key path carries the token `unpolled_control` on purpose: "
    "that is what rounds/report.py::_CONTROL_HINTS classifies on, so the census "
    "ranks it separately from the shipped arms."
)

THE_RESIDUE_THAT_CANNOT_BE_CHUNKED = (
    "np.sum uses pairwise summation. Blocking it and summing the block partials "
    "does not reproduce it bitwise, and `total` divides every entry of the "
    "sampling CDF, so a changed total is a changed science path. It is "
    "therefore left as ONE call in both polled arms, timed on its own, and "
    "published as an irreducible unpollable residue. This is the only stage "
    "interval in this module that polling cannot reduce."
)


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #


def _npy_payload(path: str) -> tuple[int, np.dtype, tuple[int, ...]]:
    """Return `(header_bytes, dtype, shape)` for a v1/v2 `.npy` file.

    The polled readers stream the payload themselves, so they need to know where
    it begins. Nothing here reads the payload.
    """
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise Round0258GraphLoadError(
                f"R0258 does not read .npy version {version} at {path}"
            )
        if fortran:
            raise Round0258GraphLoadError(
                f"R0258 refuses a Fortran-ordered edge array at {path}"
            )
        return handle.tell(), dtype, shape


def open_readonly_memmap(path: str, *, label: str) -> np.memmap:
    """R0230's reader rule: read-only `np.memmap`, asserted on the object.

    This is the R0232 safety precondition, applied even though no cuVS call
    exists on this path -- the rule earned two GPU wedges and is checked rather
    than assumed.
    """
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap) or array.flags.writeable:
        raise Round0258GraphLoadError(
            f"R0258 {label} at {path} is not a read-only np.memmap"
        )
    return array


def load_array_unpolled(path: str) -> np.ndarray:
    """The shipped shape: one `np.load`, whole array into anonymous memory.

    No abort read exists inside `np.load`. At `10,044,413,144` B this is a
    single uninterruptible interval and it is the control.
    """
    return np.load(path, allow_pickle=False)


def load_array_polled(
    path: str,
    *,
    poll: Any,
    chunk_bytes: int = READ_CHUNK_BYTES,
    o_direct: bool = False,
) -> np.ndarray:
    """Stream the same `.npy` payload in `chunk_bytes` reads, polling each read.

    Byte-for-byte the same array as `load_array_unpolled`; the only difference is
    that an abort read runs between chunks. With `o_direct` the page cache is
    bypassed, which needs a page-aligned destination -- so the payload lands in
    an aligned `mmap` and is viewed as the target dtype rather than copied.
    """
    if not callable(poll):
        raise Round0258GraphLoadError("R0258 polled loader requires a callable poll")
    if chunk_bytes <= 0 or chunk_bytes % mmap.PAGESIZE:
        raise Round0258GraphLoadError(
            f"R0258 read chunk {chunk_bytes} must be a positive multiple of "
            f"the {mmap.PAGESIZE} B page size"
        )
    offset, dtype, shape = _npy_payload(path)
    payload_bytes = int(np.prod(shape)) * int(dtype.itemsize)
    buffer = mmap.mmap(-1, payload_bytes)
    flags = os.O_RDONLY
    if o_direct:
        flags |= os.O_DIRECT
    fd = os.open(path, flags)
    try:
        # O_DIRECT requires the file offset to be page aligned too, so the
        # header is consumed by starting at the aligned page below the payload
        # and discarding the leading bytes of the first chunk.
        aligned = (offset // mmap.PAGESIZE) * mmap.PAGESIZE
        skip = offset - aligned
        os.lseek(fd, aligned, os.SEEK_SET)
        destination = np.frombuffer(buffer, dtype=np.uint8)
        scratch = mmap.mmap(-1, chunk_bytes + mmap.PAGESIZE)
        staging = np.frombuffer(scratch, dtype=np.uint8)
        written = 0
        while written < payload_bytes:
            want = min(chunk_bytes, payload_bytes - written + skip)
            want = ((want + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
            got = os.readv(fd, [staging[:want]])
            if got <= skip:
                raise Round0258GraphLoadError(
                    f"R0258 short read at offset {written} of {path}"
                )
            take = min(got - skip, payload_bytes - written)
            destination[written:written + take] = staging[skip:skip + take]
            written += take
            skip = 0
            poll(ABORT_POLL_SITE_LOAD)
        del staging, destination
    finally:
        os.close(fd)
    array = np.frombuffer(buffer, dtype=dtype, count=int(np.prod(shape)))
    return array.reshape(shape)


def evict_page_cache(path: str) -> float:
    """`posix_fadvise(DONTNEED)` the whole file; returns the seconds it cost.

    review-0254-01 §B.3 measured this syscall at `62` ms per resident GiB and
    named it as an unpollable interval in its own right. Each repetition starts
    from the same cold-cache state or the arms are not comparable, so the cost is
    paid deliberately and reported rather than hidden.
    """
    started = time.monotonic()
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return time.monotonic() - started


# --------------------------------------------------------------------------- #
# S2 -- endpoint bounds (graph_validation.validate_edge_bounds)
# --------------------------------------------------------------------------- #


def endpoint_bounds_unpolled(array: np.ndarray) -> tuple[int, int]:
    """`validate_edge_bounds`'s two calls, verbatim, with no abort read."""
    return int(array.min()), int(array.max())


def endpoint_bounds_polled(
    array: np.ndarray, *, poll: Any, chunk: int = SCAN_CHUNK_ELEMENTS
) -> tuple[int, int]:
    """Blocked min/max. Exact: min and max are associative over any partition."""
    if not callable(poll):
        raise Round0258GraphLoadError("R0258 polled bounds requires a callable poll")
    n = int(array.shape[0])
    low: int | None = None
    high: int | None = None
    for start in range(0, n, chunk):
        block = array[start:start + chunk]
        block_low = int(block.min())
        block_high = int(block.max())
        low = block_low if low is None else min(low, block_low)
        high = block_high if high is None else max(high, block_high)
        poll(ABORT_POLL_SITE_BOUNDS)
    if low is None or high is None:
        raise Round0258GraphLoadError("R0258 polled bounds saw an empty array")
    return low, high


# --------------------------------------------------------------------------- #
# S3 -- weight validity (round0107_training.DiverseWeightedJinaSampler)
# --------------------------------------------------------------------------- #


def weight_validity_unpolled(weights: np.ndarray) -> tuple[bool, bool, bool]:
    """The three shipped scans, verbatim.

    Each allocates a full boolean temporary -- `2,511,103,254` B apiece at this
    rung -- and none contains an abort read.
    """
    return (
        bool(np.isfinite(weights).all()),
        bool(np.any(weights <= 0)),
        bool(np.any(weights > 1)),
    )


def weight_validity_polled(
    weights: np.ndarray, *, poll: Any, chunk: int = SCAN_CHUNK_ELEMENTS
) -> tuple[bool, bool, bool]:
    """Blocked scans. Exact: `all` and `any` are associative over a partition.

    Also bounds the boolean temporary at `chunk` bytes instead of `n` bytes,
    which is a 150x reduction in transient anonymous memory at this rung -- a
    side effect, not the point.
    """
    if not callable(poll):
        raise Round0258GraphLoadError("R0258 polled validity requires a callable poll")
    n = int(weights.shape[0])
    finite = True
    non_positive = False
    above_one = False
    for start in range(0, n, chunk):
        block = weights[start:start + chunk]
        finite = finite and bool(np.isfinite(block).all())
        non_positive = non_positive or bool(np.any(block <= 0))
        above_one = above_one or bool(np.any(block > 1))
        poll(ABORT_POLL_SITE_VALIDITY)
    return finite, non_positive, above_one


# --------------------------------------------------------------------------- #
# S4 -- contiguous int32 host copies (HostStreamEdgeSampler.__init__)
# --------------------------------------------------------------------------- #


def contiguous_int32_unpolled(array: np.ndarray) -> np.ndarray:
    """`np.ascontiguousarray(np.asarray(x), dtype=np.int32)`, verbatim."""
    return np.ascontiguousarray(np.asarray(array), dtype=np.int32)


def contiguous_int32_polled(
    array: np.ndarray, *, poll: Any, chunk: int = SCAN_CHUNK_ELEMENTS
) -> np.ndarray:
    """Blocked copy into a preallocated int32 array. Exact: a copy is a copy."""
    if not callable(poll):
        raise Round0258GraphLoadError(
            "R0258 polled contiguous copy requires a callable poll"
        )
    n = int(array.shape[0])
    out = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        out[start:end] = array[start:end]
        poll(ABORT_POLL_SITE_CONTIGUOUS)
    return out


# --------------------------------------------------------------------------- #
# S5 -- the float64 sampling CDF (HostStreamEdgeSampler.__init__)
# --------------------------------------------------------------------------- #


def weight_cdf_unpolled(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """The shipped CDF construction, verbatim, with no abort read.

    `HostStreamEdgeSampler.__init__` lines 609-628: an f64 copy, `w.sum()`,
    `np.cumsum(w, out=w)`, `w /= total`. At this rung the f64 array is
    `20,088,826,032` B and `np.cumsum` over it is a single sequential recurrence
    of `2,511,103,254` additions with no interruption point of any kind.
    """
    w = np.array(weights, dtype=np.float64, copy=True)
    total = float(w.sum())
    if not (total > 0):
        raise Round0258GraphLoadError(
            f"R0258 non-positive weight total {total}: unsamplable"
        )
    np.cumsum(w, out=w)
    w /= total
    return w, total


def weight_cdf_polled(
    weights: np.ndarray,
    *,
    poll: Any,
    chunk: int = SCAN_CHUNK_ELEMENTS,
) -> tuple[np.ndarray, float, float]:
    """Bitwise-identical CDF with an abort read between chunks.

    Returns `(cdf, total, unpollable_sum_s)`.

    The conversion, the cumulative sum and the division are all chunked; the
    `np.sum` is not, for the reason in `THE_RESIDUE_THAT_CANNOT_BE_CHUNKED`, and
    its wall is returned so the residue is published rather than absorbed.

    The cumulative sum is exact because the carry is folded into the FIRST
    ELEMENT of the chunk before `np.cumsum` runs over it. `np.cumsum` computes
    `out[i] = out[i-1] + w[i]` strictly in order, so seeding `w[lo] += carry`
    and then cumsum-ing `w[lo:hi]` issues exactly the additions the whole-array
    call would have issued, in exactly that order. Adding the carry after the
    chunk's cumsum -- the obvious implementation -- does not, and is
    `carry_added_after_the_chunk` in `bitwise_identity_controls()`.
    """
    if not callable(poll):
        raise Round0258GraphLoadError("R0258 polled CDF requires a callable poll")
    n = int(weights.shape[0])
    w = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        w[start:end] = weights[start:end]
        poll(ABORT_POLL_SITE_CDF)

    residue_started = time.monotonic()
    total = float(w.sum())
    unpollable_sum_s = time.monotonic() - residue_started
    if not (total > 0):
        raise Round0258GraphLoadError(
            f"R0258 non-positive weight total {total}: unsamplable"
        )

    carry = 0.0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        w[start] += carry
        np.cumsum(w[start:end], out=w[start:end])
        carry = float(w[end - 1])
        poll(ABORT_POLL_SITE_CDF)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        w[start:end] /= total
        poll(ABORT_POLL_SITE_CDF)
    return w, total, unpollable_sum_s


# --------------------------------------------------------------------------- #
# the identity check, and the five planted numeric defects it must refuse
# --------------------------------------------------------------------------- #


def bitwise_identical(left: np.ndarray, right: np.ndarray) -> bool:
    """Exact equality of dtype, shape and every byte. Not `allclose`."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return bool(
        np.array_equal(
            left.view(np.uint8) if left.dtype.kind == "f" else left,
            right.view(np.uint8) if right.dtype.kind == "f" else right,
        )
    )


def _approximately_identical(left: np.ndarray, right: np.ndarray) -> bool:
    """The weaker predicate an implementer would reach for. Kept for contrast."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    return bool(np.allclose(left, right))


def _defect_carry_added_after_the_chunk(weights, *, chunk):
    w = np.array(weights, dtype=np.float64, copy=True)
    n = int(w.shape[0])
    total = float(w.sum())
    carry = 0.0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        np.cumsum(w[start:end], out=w[start:end])
        w[start:end] += carry
        carry = float(w[end - 1])
    w /= total
    return w


def _defect_no_carry(weights, *, chunk):
    w = np.array(weights, dtype=np.float64, copy=True)
    n = int(w.shape[0])
    total = float(w.sum())
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        np.cumsum(w[start:end], out=w[start:end])
    w /= total
    return w


def _defect_float32_accumulator(weights, *, chunk):
    w = np.array(weights, dtype=np.float32, copy=True)
    total = float(w.sum())
    np.cumsum(w, out=w)
    w /= total
    return w.astype(np.float64)


def _defect_blocked_total(weights, *, chunk):
    w = np.array(weights, dtype=np.float64, copy=True)
    n = int(w.shape[0])
    total = 0.0
    for start in range(0, n, chunk):
        total += float(w[start:min(start + chunk, n)].sum())
    carry = 0.0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        w[start] += carry
        np.cumsum(w[start:end], out=w[start:end])
        carry = float(w[end - 1])
    w /= total
    return w


def _defect_chunk_boundary_off_by_one(weights, *, chunk):
    w = np.array(weights, dtype=np.float64, copy=True)
    n = int(w.shape[0])
    total = float(w.sum())
    carry = 0.0
    for start in range(0, n, chunk):
        end = min(start + chunk + 1, n)
        w[start] += carry
        np.cumsum(w[start:end], out=w[start:end])
        carry = float(w[min(start + chunk, n) - 1])
    w /= total
    return w


#: name -> builder. Each plants a defect a real implementation could plausibly
#: ship, and each must be REFUSED by `bitwise_identical`.
NUMERIC_DEFECTS = {
    "carry_added_after_the_chunk": _defect_carry_added_after_the_chunk,
    "no_carry_between_chunks": _defect_no_carry,
    "float32_accumulator": _defect_float32_accumulator,
    "blocked_total_instead_of_one_np_sum": _defect_blocked_total,
    "chunk_boundary_off_by_one": _defect_chunk_boundary_off_by_one,
}


def bitwise_identity_controls(
    *, n: int = 4_000_017, chunk: int = 65_536, seed: int = 20260811
) -> dict[str, Any]:
    """Plant five numeric defects and require the shipped comparison to refuse.

    The contrast is the point, and it is the R0254 standard: each defect must be
    **refused** by `bitwise_identical` and **accepted** by `_approximately_identical`.
    A defect both predicates refuse proves nothing about the shipped one; a
    defect both accept proves the shipped one is not doing the work. `n` is
    deliberately not a multiple of `chunk` so the tail chunk is exercised.
    """
    rng = np.random.default_rng(seed)
    weights = rng.random(n, dtype=np.float64).astype(np.float32) + np.float32(1e-6)

    shipped, total = weight_cdf_unpolled(weights)
    polled, polled_total, residue_s = weight_cdf_polled(
        weights, poll=lambda _where: None, chunk=chunk
    )

    verdicts: dict[str, Any] = {}
    for name, builder in sorted(NUMERIC_DEFECTS.items()):
        planted = builder(weights, chunk=chunk)
        verdicts[name] = {
            "refused_by_the_shipped_comparison": not bitwise_identical(
                shipped, planted
            ),
            "accepted_by_an_allclose_comparison": _approximately_identical(
                shipped, planted
            ),
            "max_absolute_deviation": float(
                np.max(np.abs(np.asarray(planted, dtype=np.float64) - shipped))
            ),
        }
    honest = {
        "refused_by_the_shipped_comparison": not bitwise_identical(shipped, polled),
        "accepted_by_an_allclose_comparison": _approximately_identical(
            shipped, polled
        ),
    }
    return {
        "schema": "round0258-bitwise-identity-controls-v1",
        "elements": int(n),
        "chunk_elements": int(chunk),
        "tail_chunk_exercised": bool(n % chunk),
        "seed": int(seed),
        "shipped_total": float(total),
        "polled_total": float(polled_total),
        "totals_are_bitwise_equal": total == polled_total,
        "unpollable_np_sum_s": float(residue_s),
        "planted": verdicts,
        "the_polled_path_itself": honest,
        "defects_planted": len(NUMERIC_DEFECTS),
        "defects_refused_by_the_shipped_comparison": sum(
            1 for v in verdicts.values() if v["refused_by_the_shipped_comparison"]
        ),
        "defects_accepted_by_an_allclose_comparison": sum(
            1 for v in verdicts.values() if v["accepted_by_an_allclose_comparison"]
        ),
        "the_polled_path_is_bitwise_identical": not honest[
            "refused_by_the_shipped_comparison"
        ],
    }


def assert_bitwise_identity_controls(**kwargs: Any) -> dict[str, Any]:
    report = bitwise_identity_controls(**kwargs)
    if not report["the_polled_path_is_bitwise_identical"]:
        raise Round0258GraphLoadError(
            "R0258: the polled CDF is not bitwise identical to the shipped one"
        )
    if report["defects_refused_by_the_shipped_comparison"] != len(NUMERIC_DEFECTS):
        raise Round0258GraphLoadError(
            "R0258: a planted numeric defect was not refused -- the identity "
            "check is not doing its only job"
        )
    if report["defects_accepted_by_an_allclose_comparison"] < 1:
        raise Round0258GraphLoadError(
            "R0258: no planted defect was accepted by the weaker comparison, so "
            "the contrast proves nothing about the shipped one (R0254 standard)"
        )
    return report


# --------------------------------------------------------------------------- #
# the structural guard: does every polled stage poll inside its own loop?
# --------------------------------------------------------------------------- #

POLLED_FUNCTIONS = (
    load_array_polled,
    endpoint_bounds_polled,
    weight_validity_polled,
    contiguous_int32_polled,
    weight_cdf_polled,
)


def chunk_loop_polls(source: str | None = None, *, function: Any = None) -> dict[str, Any]:
    """Does the chunk loop contain an abort read, structurally?

    Modelled on `round0254_writeback.write_loop_polls` and, like it, `source`
    exists so a positive control plants the defect and runs it through **this**
    function rather than re-implementing the walk in the test body -- the
    review-0253-01 §I defect.

    A loop counts as a chunk loop when its body indexes with a slice or calls a
    read primitive; it counts as polled when a call to the name `poll` appears
    among the statements of the loop body, at any depth **inside a statement
    that always executes**. A `poll` in a dead branch, after the loop, or bound
    to something else does not count -- those are the planted defects.
    """
    if source is None:
        if function is None:
            raise Round0258GraphLoadError(
                "R0258 chunk-loop guard needs a function or a planted source"
            )
        source = inspect.getsource(function)
    text = textwrap.dedent(source)
    tree = ast.parse(text)
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        raise Round0258GraphLoadError("R0258 chunk-loop guard was given no function")
    target = functions[0]

    loops = [
        loop for loop in ast.walk(target)
        if isinstance(loop, (ast.While, ast.For))
        and any(
            isinstance(inner, ast.Subscript)
            or (isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr in ("readv", "readinto", "read"))
            for inner in ast.walk(loop)
        )
    ]

    def _live_statements(body: Sequence[ast.stmt]) -> list[ast.stmt]:
        live: list[ast.stmt] = []
        for statement in body:
            if isinstance(statement, ast.If):
                test = statement.test
                if isinstance(test, ast.Constant) and not test.value:
                    live.extend(_live_statements(statement.orelse))
                    continue
                # a conditional poll is not an unconditional one
                continue
            if isinstance(statement, (ast.Try, ast.With)):
                continue
            live.append(statement)
        return live

    polled: list[int] = []
    for loop in loops:
        for statement in _live_statements(loop.body):
            for node in ast.walk(statement):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "poll"):
                    polled.append(int(getattr(node, "lineno", -1)))

    shadowed = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "poll" for t in node.targets)
        for node in ast.walk(target)
    )
    return {
        "schema": "round0258-chunk-loop-poll-guard-v1",
        "function": target.name,
        "checked": "the shipped function" if function is not None else "planted source",
        "chunk_loops_found": len(loops),
        "poll_calls_inside_a_chunk_loop": len(polled),
        "poll_call_linenos": sorted(polled),
        "poll_is_locally_rebound": shadowed,
        "the_chunk_loop_polls": bool(loops) and bool(polled) and not shadowed,
    }


def assert_chunk_loops_poll() -> dict[str, Any]:
    """Every shipped polled stage must poll inside its own chunk loop."""
    audit = {
        function.__name__: chunk_loop_polls(function=function)
        for function in POLLED_FUNCTIONS
    }
    missing = sorted(
        name for name, report in audit.items() if not report["the_chunk_loop_polls"]
    )
    if missing:
        raise Round0258GraphLoadError(
            "R0258: these polled stages contain no abort read inside their chunk "
            f"loop: {missing}. This is the review-0252-01 §H shape -- an interval "
            "no census can see."
        )
    return {
        "schema": "round0258-chunk-loop-audit-v1",
        "functions_checked": len(POLLED_FUNCTIONS),
        "functions_that_poll_inside_their_chunk_loop": len(POLLED_FUNCTIONS),
        "by_function": audit,
        "declared_sites": list(ABORT_POLL_SITES),
    }


# --------------------------------------------------------------------------- #
# the five planted STRUCTURAL defects
# --------------------------------------------------------------------------- #

_PLANT_HONEST = '''
def stage(array, *, poll, chunk):
    out = np.empty(len(array))
    for start in range(0, len(array), chunk):
        out[start:start + chunk] = array[start:start + chunk]
        poll("site")
    return out
'''

_PLANT_POLL_AFTER_THE_LOOP = '''
def stage(array, *, poll, chunk):
    out = np.empty(len(array))
    for start in range(0, len(array), chunk):
        out[start:start + chunk] = array[start:start + chunk]
    poll("site")
    return out
'''

_PLANT_POLL_IN_A_DEAD_BRANCH = '''
def stage(array, *, poll, chunk):
    out = np.empty(len(array))
    for start in range(0, len(array), chunk):
        out[start:start + chunk] = array[start:start + chunk]
        if False:
            poll("site")
    return out
'''

_PLANT_POLL_ONLY_WHEN_A_CONDITION_HOLDS = '''
def stage(array, *, poll, chunk):
    out = np.empty(len(array))
    for start in range(0, len(array), chunk):
        out[start:start + chunk] = array[start:start + chunk]
        if start % 1000000000 == 0:
            poll("site")
    return out
'''

_PLANT_POLL_LOCALLY_REBOUND = '''
def stage(array, *, poll, chunk):
    poll = lambda where: None
    out = np.empty(len(array))
    for start in range(0, len(array), chunk):
        out[start:start + chunk] = array[start:start + chunk]
        poll("site")
    return out
'''

_PLANT_NO_CHUNK_LOOP_AT_ALL = '''
def stage(array, *, poll, chunk):
    poll("site")
    out = np.asarray(array).copy()
    poll("site")
    return out
'''

STRUCTURAL_DEFECTS = {
    "poll_after_the_loop": _PLANT_POLL_AFTER_THE_LOOP,
    "poll_in_a_dead_branch": _PLANT_POLL_IN_A_DEAD_BRANCH,
    "poll_only_when_a_condition_holds": _PLANT_POLL_ONLY_WHEN_A_CONDITION_HOLDS,
    "poll_locally_rebound": _PLANT_POLL_LOCALLY_REBOUND,
    "no_chunk_loop_at_all": _PLANT_NO_CHUNK_LOOP_AT_ALL,
}


def _substring_predicate(source: str) -> bool:
    """The predicate a round would have shipped before this one: does the text
    mention `poll(`? Every planted defect passes it. Kept so each refusal below
    is a demonstrated behaviour CHANGE rather than an unfalsifiable pass."""
    return "poll(" in source


def structural_defect_controls() -> dict[str, Any]:
    """Plant five install defects; the shipped guard must refuse all five."""
    verdicts: dict[str, Any] = {}
    for name, source in sorted(STRUCTURAL_DEFECTS.items()):
        report = chunk_loop_polls(source)
        verdicts[name] = {
            "refused_by_the_shipped_guard": not report["the_chunk_loop_polls"],
            "accepted_by_a_substring_predicate": _substring_predicate(source),
            "chunk_loops_found": report["chunk_loops_found"],
            "poll_calls_inside_a_chunk_loop": report["poll_calls_inside_a_chunk_loop"],
        }
    honest = chunk_loop_polls(_PLANT_HONEST)
    return {
        "schema": "round0258-structural-defect-controls-v1",
        "planted": verdicts,
        "defects_planted": len(STRUCTURAL_DEFECTS),
        "defects_refused_by_the_shipped_guard": sum(
            1 for v in verdicts.values() if v["refused_by_the_shipped_guard"]
        ),
        "defects_accepted_by_a_substring_predicate": sum(
            1 for v in verdicts.values() if v["accepted_by_a_substring_predicate"]
        ),
        "an_honest_install_passes_both": bool(
            honest["the_chunk_loop_polls"] and _substring_predicate(_PLANT_HONEST)
        ),
    }


def assert_structural_defect_controls() -> dict[str, Any]:
    report = structural_defect_controls()
    if report["defects_refused_by_the_shipped_guard"] != len(STRUCTURAL_DEFECTS):
        raise Round0258GraphLoadError(
            "R0258: a planted structural defect passed the chunk-loop guard"
        )
    if not report["an_honest_install_passes_both"]:
        raise Round0258GraphLoadError(
            "R0258: the chunk-loop guard refuses an honest install"
        )
    if report["defects_accepted_by_a_substring_predicate"] != len(STRUCTURAL_DEFECTS):
        raise Round0258GraphLoadError(
            "R0258: the contrast predicate did not accept every plant, so the "
            "refusals are not demonstrated behaviour changes (R0254 standard)"
        )
    return report


__all__ = [
    "ABORT_POLL_SITES",
    "ARMS",
    "ARM_POLLED_BUFFERED",
    "ARM_POLLED_O_DIRECT",
    "ARM_UNPOLLED_CONTROL",
    "DIRECTED_EDGES",
    "EDGE_ARRAYS",
    "EDGE_HEADER_PATH",
    "K",
    "NUMERIC_DEFECTS",
    "READ_CHUNK_BYTES",
    "ROUND_ID",
    "ROWS",
    "R0243_FUZZY_DIR",
    "Round0258GraphLoadError",
    "SCAN_CHUNK_ELEMENTS",
    "SHIPPED_ARMS",
    "STAGES",
    "STRUCTURAL_DEFECTS",
    "THE_RESIDUE_THAT_CANNOT_BE_CHUNKED",
    "WHAT_THE_UNPOLLED_ARM_IS",
    "arm_schedule",
    "assert_bitwise_identity_controls",
    "assert_chunk_loops_poll",
    "assert_structural_defect_controls",
    "bitwise_identical",
    "bitwise_identity_controls",
    "chunk_loop_polls",
    "contiguous_int32_polled",
    "contiguous_int32_unpolled",
    "endpoint_bounds_polled",
    "endpoint_bounds_unpolled",
    "evict_page_cache",
    "interval_summary",
    "load_array_polled",
    "load_array_unpolled",
    "open_readonly_memmap",
    "registered_ceiling_s",
    "structural_defect_controls",
    "weight_cdf_polled",
    "weight_cdf_unpolled",
    "weight_validity_polled",
    "weight_validity_unpolled",
]
