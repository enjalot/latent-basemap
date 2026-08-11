"""R0254 — what bounds the write stall, if anything.

R0253 measured one `os.write` of 64 MiB at `2.7842220919847023` s
`= 1.108830746147159x` the registered ceiling and concluded, in the one
prescriptive sentence its result contained:

    The next step is not more polling but a smaller write unit, and its cost has
    not been measured.

review-0253-01 §D measured it and falsified it.  Replicating the `49,152,000,000`
B rung seven times, an `8` MiB write unit moved the **median** `11x` and the
**p99** `~700x` and the **maximum not at all** — `1.576`/`1.793` s at 8 MiB
against `1.475`/`1.567`/`2.298` s at 64 MiB under matched load.  One 8 MiB
`os.write` stalled for `1.79` s.  The binding interval is not a property of the
write unit; it is a device writeback event that any single `os.write` can land
in, and no poll runs inside a syscall.

So this module does not vary the write unit.  It varies the **only three things
available to us without root** and measures the resulting per-write interval
*distribution* rather than one maximum:

* the `fsync` cadence, which is what bounds how much dirty data can be
  outstanding when a write arrives (`2` GiB is R0253's shipped value);
* `O_DIRECT`, which takes the page cache and the writeback threads out of the
  path entirely and makes every write wait on the device instead of on the
  kernel;
* `sync_file_range` write-behind, the standard remedy for exactly this shape:
  start the writeback of the extent just written and wait on an extent a fixed
  lag behind, so the queue drains steadily instead of in a `fsync`-shaped burst.

`vm.dirty_ratio` / `vm.dirty_bytes` tuning is the fourth candidate and is **not
attempted**: it is a global root-owned setting on a shared workstation, it would
change the device's behaviour for every other job on this box, and a measurement
taken under a modified sysctl would not describe the machine the ladder actually
runs on.  The values in force are read and published instead.

Two things this module measures that R0253's did not:

* **The final `fsync`.**  review-0253-01 §E: R0253's own generated census ranked
  `0.962761x` `2.417448` s second, at the site between the last in-loop poll and
  `"R0253 file written"` — the closing `os.fsync` + `posix_fadvise` + `close` on
  a 49 GB file.  It appears nowhere in R0253's result.  Here it is timed as its
  own named interval in every arm.
* **The raw per-write interval**, taken around `os.write` itself with
  `time.monotonic`, independently of the gate and of the poll recorder.  The gap
  series and the raw series are published side by side so a reader can see that
  the gap the census ranks is the syscall and not the instrumentation.
"""
from __future__ import annotations

import ast
import ctypes
import ctypes.util
import hashlib
import inspect
import mmap
import os
import textwrap
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0247_registry import registered_value


class Round0254WritebackError(RuntimeError):
    """The registered R0254 writeback-probe contract changed."""


ROUND_ID = "0254"

#: The rung that breached.  R0253 measured `1.108830746147159x` here and
#: review-0253-01 replicated the shape seven times at this exact size.  The
#: probe stays at one size deliberately: the question is no longer how the stall
#: scales, it is whether anything bounds it.
STALL_SIZE_BYTES = 49_152_000_000

#: R0253's write unit, held fixed.  The smaller-unit remedy is falsified
#: (review-0253-01 §D) and `plan-minilm-100m-v2.md` now records it; retrying it
#: would spend the round's disk budget re-deriving a known negative.
WRITE_BLOCK_BYTES = 64 << 20

#: `os.fsync` + `POSIX_FADV_DONTNEED` cadences, in bytes.  `2 GiB` is R0253's.
CADENCE_2GIB = 2 << 30
CADENCE_256MIB = 256 << 20
CADENCE_32MIB = 32 << 20

#: `sync_file_range` write-behind lag, in write blocks.
SFR_LAG_BLOCKS = 2

#: glibc's `sync_file_range` flags (`fcntl.h`).
SYNC_FILE_RANGE_WAIT_BEFORE = 1
SYNC_FILE_RANGE_WRITE = 2
SYNC_FILE_RANGE_WAIT_AFTER = 4

ARM_FSYNC_2GIB = "fsync_2gib"
ARM_FSYNC_256MIB = "fsync_256mib"
ARM_FSYNC_32MIB = "fsync_32mib"
ARM_FSYNC_FINAL_ONLY = "fsync_final_only"
ARM_O_DIRECT = "o_direct"
ARM_SYNC_FILE_RANGE = "sync_file_range"
#: The one arm that is built to fail, so `roundreport` ranks it separately: it
#: never polls, which is R0252's shipped shape and the reason a 93 s cooperative
#: stop appeared in no census.  The site string carries `unpolled` on purpose —
#: that is the token `rounds/report.py::_CONTROL_HINTS` classifies on.
ARM_UNPOLLED_CONTROL = "unpolled_control"

#: Every arm this round measures, in a fixed order.  The rotation in
#: `arm_schedule` is what balances device state across them.
SHIPPED_ARMS: tuple[str, ...] = (
    ARM_FSYNC_2GIB,
    ARM_FSYNC_256MIB,
    ARM_FSYNC_32MIB,
    ARM_FSYNC_FINAL_ONLY,
    ARM_O_DIRECT,
    ARM_SYNC_FILE_RANGE,
)

ABORT_POLL_SITE_BLOCK = "round0254_writeback.write_arm block"
ABORT_POLL_SITE_FLUSH = "round0254_writeback.write_arm flush"
ABORT_POLL_SITE_FINAL = "round0254_writeback.write_arm final fsync"

ABORT_POLL_SITES: tuple[str, ...] = (
    ABORT_POLL_SITE_BLOCK,
    ABORT_POLL_SITE_FLUSH,
    ABORT_POLL_SITE_FINAL,
)


# --------------------------------------------------------------------------- #
# the arms
# --------------------------------------------------------------------------- #


class _ArmSpec:
    __slots__ = ("name", "flush_every_bytes", "o_direct", "sync_file_range", "polls")

    def __init__(self, name: str, *, flush_every_bytes: int | None,
                 o_direct: bool, sync_file_range: bool, polls: bool) -> None:
        self.name = name
        self.flush_every_bytes = flush_every_bytes
        self.o_direct = o_direct
        self.sync_file_range = sync_file_range
        self.polls = polls

    def receipt(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "flush_every_bytes": self.flush_every_bytes,
            "o_direct": self.o_direct,
            "sync_file_range_write_behind": self.sync_file_range,
            "sync_file_range_lag_blocks": SFR_LAG_BLOCKS if self.sync_file_range else None,
            "polls_between_blocks": self.polls,
            "write_block_bytes": WRITE_BLOCK_BYTES,
        }


ARMS: dict[str, _ArmSpec] = {
    ARM_FSYNC_2GIB: _ArmSpec(
        ARM_FSYNC_2GIB, flush_every_bytes=CADENCE_2GIB,
        o_direct=False, sync_file_range=False, polls=True),
    ARM_FSYNC_256MIB: _ArmSpec(
        ARM_FSYNC_256MIB, flush_every_bytes=CADENCE_256MIB,
        o_direct=False, sync_file_range=False, polls=True),
    ARM_FSYNC_32MIB: _ArmSpec(
        ARM_FSYNC_32MIB, flush_every_bytes=CADENCE_32MIB,
        o_direct=False, sync_file_range=False, polls=True),
    ARM_FSYNC_FINAL_ONLY: _ArmSpec(
        ARM_FSYNC_FINAL_ONLY, flush_every_bytes=None,
        o_direct=False, sync_file_range=False, polls=True),
    ARM_O_DIRECT: _ArmSpec(
        ARM_O_DIRECT, flush_every_bytes=None,
        o_direct=True, sync_file_range=False, polls=True),
    ARM_SYNC_FILE_RANGE: _ArmSpec(
        ARM_SYNC_FILE_RANGE, flush_every_bytes=None,
        o_direct=False, sync_file_range=True, polls=True),
    ARM_UNPOLLED_CONTROL: _ArmSpec(
        ARM_UNPOLLED_CONTROL, flush_every_bytes=CADENCE_2GIB,
        o_direct=False, sync_file_range=False, polls=False),
}


def arm_schedule(arms: Sequence[str], repetitions: int) -> list[tuple[int, str]]:
    """Round-robin with a rotating start, so no arm owns a device state.

    review-0253-01 §C.3 is the reason this exists rather than a simple nested
    loop: its runs 1--2 ran on a quiescent device and produced a `0.42` s
    maximum, and runs 3--7, after ~100 GB of writes and unlinks had accumulated,
    produced `1.5`--`2.3` s.  An experiment that runs all of arm A and then all
    of arm B measures the device's history, not the arm.
    """
    if repetitions < 1:
        raise Round0254WritebackError("R0254 needs at least one repetition per arm")
    if not arms:
        raise Round0254WritebackError("R0254 needs at least one arm")
    width = len(arms)
    schedule: list[tuple[int, str]] = []
    for repetition in range(repetitions):
        for index in range(width):
            schedule.append((repetition, arms[(index + repetition) % width]))
    return schedule


# --------------------------------------------------------------------------- #
# sync_file_range, through libc
# --------------------------------------------------------------------------- #


_LIBC: Any = None


def _libc() -> Any:
    global _LIBC
    if _LIBC is None:
        name = ctypes.util.find_library("c") or "libc.so.6"
        library = ctypes.CDLL(name, use_errno=True)
        function = getattr(library, "sync_file_range", None)
        if function is None:
            raise Round0254WritebackError(
                "R0254 sync_file_range arm needs glibc's sync_file_range; this "
                "libc does not export it"
            )
        function.argtypes = [
            ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_uint
        ]
        function.restype = ctypes.c_int
        _LIBC = library
    return _LIBC


def sync_file_range(fd: int, offset: int, nbytes: int, flags: int) -> None:
    result = _libc().sync_file_range(
        int(fd), ctypes.c_longlong(int(offset)), ctypes.c_longlong(int(nbytes)),
        ctypes.c_uint(int(flags)),
    )
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), f"sync_file_range({offset},{nbytes})")


# --------------------------------------------------------------------------- #
# the write
# --------------------------------------------------------------------------- #


def _aligned_block(seed: int, size: int) -> Any:
    """A page-aligned buffer, because `O_DIRECT` requires one.

    `mmap.mmap(-1, size)` is an anonymous mapping and is therefore page-aligned
    by construction, which `bytes` is not.  The same buffer is used for every
    arm so the arms differ only in how the write is issued.
    """
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
    buffer = mmap.mmap(-1, size)
    buffer.write(payload)
    buffer.seek(0)
    return buffer


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Round0254WritebackError("R0254 quantile of an empty series")
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def interval_summary(values: Sequence[float], *, ceiling_s: float) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "note": "no intervals of this kind were issued in this arm",
        }
    total = sum(ordered)
    return {
        "count": len(ordered),
        "total_s": total,
        "mean_s": total / len(ordered),
        "median_s": _quantile(ordered, 0.5),
        "p90_s": _quantile(ordered, 0.90),
        "p99_s": _quantile(ordered, 0.99),
        "p999_s": _quantile(ordered, 0.999),
        "max_s": ordered[-1],
        "min_s": ordered[0],
        "max_over_the_ceiling": ordered[-1] / ceiling_s,
        "p99_over_the_ceiling": _quantile(ordered, 0.99) / ceiling_s,
        "intervals_over_the_ceiling": sum(1 for value in ordered if value > ceiling_s),
    }


def write_arm(
    path: str,
    *,
    arm: str,
    total_bytes: int = STALL_SIZE_BYTES,
    seed: int,
    poll: Any = None,
    block: Any = None,
) -> dict[str, Any]:
    """Write `total_bytes` under one arm's flush discipline, timing every write.

    Byte semantics are R0253's in every arm: the same 64 MiB pseudorandom block
    reused for every write, no `fallocate`, no `ftruncate`, no seek past EOF, so
    the file cannot be sparse and `st_blocks` is read from the inode as
    review-0252-01 §B.2 requires.  The arms differ ONLY in when the kernel is
    asked to put the bytes on the device.

    The digest is of the source block plus the block count, not a streaming hash
    of the stream: a streaming `sha256` costs ~`17` s at this size on this box
    (`2.80` GB/s, plan-minilm-100m-v2.md) and would sit inside the very interval
    being measured.  Identical `block_sha256` + identical `blocks_written` +
    identical `bytes` is the same evidence at no cost inside the loop.
    """
    spec = ARMS.get(str(arm))
    if spec is None:
        raise Round0254WritebackError(f"R0254 has no write arm {arm!r}")
    if total_bytes <= 0:
        raise Round0254WritebackError(f"R0254 write needs a positive size, got {total_bytes}")
    if spec.polls and poll is not None and not callable(poll):
        raise Round0254WritebackError("R0254 write-arm poll must be callable or None")
    owned_block = block is None
    buffer = _aligned_block(seed, WRITE_BLOCK_BYTES) if owned_block else block
    view = memoryview(buffer)
    block_sha256 = hashlib.sha256(view).hexdigest()

    ceiling = registered_ceiling_s()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if spec.o_direct:
        flags |= os.O_DIRECT

    write_intervals: list[float] = []
    flush_intervals: list[float] = []
    sfr_intervals: list[float] = []
    written = 0
    since_flush = 0
    blocks = 0
    flushes = 0
    polls = 0
    started = time.monotonic()
    fd = os.open(path, flags, 0o644)
    try:
        while written < total_bytes:
            if spec.polls and poll is not None:
                poll(ABORT_POLL_SITE_BLOCK)
                polls += 1
            take = min(WRITE_BLOCK_BYTES, total_bytes - written)
            chunk = view if take == WRITE_BLOCK_BYTES else view[:take]
            offset = 0
            while offset < take:
                piece = chunk if offset == 0 else chunk[offset:]
                at = time.monotonic()
                sent = os.write(fd, piece)
                write_intervals.append(time.monotonic() - at)
                # Every derived memoryview is an EXPORT of the mmap, and an mmap
                # with a live export cannot be closed: the tail block's
                # `view[:take]` slice survived the loop and `buffer.close()`
                # raised `BufferError: cannot close exported pointers exist`
                # after a full 49 GB write. Release each derived view as soon as
                # it is done rather than relying on refcount timing.
                if piece is not chunk:
                    piece.release()
                offset += sent
            if chunk is not view:
                chunk.release()
            start_of_block = written
            written += take
            blocks += 1
            since_flush += take

            if spec.sync_file_range:
                at = time.monotonic()
                sync_file_range(fd, start_of_block, take, SYNC_FILE_RANGE_WRITE)
                lag = start_of_block - SFR_LAG_BLOCKS * WRITE_BLOCK_BYTES
                if lag >= 0:
                    sync_file_range(
                        fd, lag, WRITE_BLOCK_BYTES,
                        SYNC_FILE_RANGE_WAIT_BEFORE
                        | SYNC_FILE_RANGE_WRITE
                        | SYNC_FILE_RANGE_WAIT_AFTER,
                    )
                sfr_intervals.append(time.monotonic() - at)

            if spec.flush_every_bytes is not None and since_flush >= spec.flush_every_bytes:
                if spec.polls and poll is not None:
                    poll(ABORT_POLL_SITE_FLUSH)
                    polls += 1
                at = time.monotonic()
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                flush_intervals.append(time.monotonic() - at)
                since_flush = 0
                flushes += 1

        # The closing flush, timed as its OWN interval.  review-0253-01 §E: this
        # is the site R0253's census ranked second at `0.962761x` and its result
        # never mentioned.
        if spec.polls and poll is not None:
            poll(ABORT_POLL_SITE_FINAL)
            polls += 1
        final_started = time.monotonic()
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        final_flush_s = time.monotonic() - final_started
        flushes += 1
    finally:
        close_started = time.monotonic()
        os.close(fd)
        close_s = time.monotonic() - close_started
    wall = time.monotonic() - started
    if owned_block:
        view.release()
        buffer.close()

    observed = os.path.getsize(path)
    if observed != total_bytes:
        raise Round0254WritebackError(
            f"R0254 wrote {observed} B to {path}, asked for {total_bytes}"
        )
    stat = os.stat(path)
    return {
        "arm": spec.name,
        "arm_spec": spec.receipt(),
        "path": path,
        "bytes": observed,
        "write_wall_s": wall,
        "write_throughput_bytes_per_s": (observed / wall) if wall > 0 else None,
        "blocks_written": blocks,
        "flushes": flushes,
        "abort_reads_inside_the_write": polls,
        "block_sha256": block_sha256,
        "allocated_blocks_512b": int(stat.st_blocks),
        "fully_allocated": bool(stat.st_blocks * 512 >= observed),
        "how_allocation_is_evidenced": (
            "os.stat().st_blocks read from the inode, not a hard-coded literal, "
            "as review-0252-01 §B.2 required of R0252's `sparse: false` field."
        ),
        "registered_ceiling_s": ceiling,
        "per_write_interval": interval_summary(write_intervals, ceiling_s=ceiling),
        "per_flush_interval": interval_summary(flush_intervals, ceiling_s=ceiling),
        "per_sync_file_range_interval": interval_summary(sfr_intervals, ceiling_s=ceiling),
        "final_flush_s": final_flush_s,
        "final_flush_over_the_ceiling": final_flush_s / ceiling,
        "close_s": close_s,
        "close_over_the_ceiling": close_s / ceiling,
        "the_widest_single_syscall_s": max(
            [*write_intervals, *flush_intervals, *sfr_intervals, final_flush_s, close_s]
        ),
        "what_the_raw_series_is": (
            "time.monotonic taken immediately around each os.write / os.fsync / "
            "sync_file_range, independently of the gate and of PollRecorder. It "
            "is published beside the gap series so a reader can confirm the "
            "interval the census ranks is the syscall and not the instrument."
        ),
    }


# --------------------------------------------------------------------------- #
# what the arms say together
# --------------------------------------------------------------------------- #


def registered_ceiling_s() -> float:
    return float(registered_value("r0246_max_poll_spacing_s"))


def stall_verdict(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold the repetitions of every arm into one statement per arm.

    The question is not "which arm was fastest" — the medians already differ by
    construction, and review-0253-01 §D showed a remedy can move the median `11x`
    and the p99 `~700x` while leaving the maximum untouched.  The question is
    **what bounds the maximum**, so the maximum of maxima across repetitions is
    the statistic each arm is judged on, with the per-repetition maxima printed
    beside it so a single lucky draw cannot stand in for a bound.
    """
    ceiling = registered_ceiling_s()
    by_arm: dict[str, dict[str, Any]] = {}
    for run in runs:
        arm = str(run["arm"])
        entry = by_arm.setdefault(arm, {
            "arm": arm,
            "repetitions": 0,
            "per_repetition_max_write_s": [],
            "per_repetition_median_write_s": [],
            "per_repetition_p99_write_s": [],
            "per_repetition_final_flush_s": [],
            "per_repetition_wall_s": [],
            "writes_over_the_ceiling": 0,
            "writes_observed": 0,
        })
        summary = run["per_write_interval"]
        entry["repetitions"] += 1
        entry["per_repetition_max_write_s"].append(float(summary["max_s"]))
        entry["per_repetition_median_write_s"].append(float(summary["median_s"]))
        entry["per_repetition_p99_write_s"].append(float(summary["p99_s"]))
        entry["per_repetition_final_flush_s"].append(float(run["final_flush_s"]))
        entry["per_repetition_wall_s"].append(float(run["write_wall_s"]))
        entry["writes_over_the_ceiling"] += int(summary["intervals_over_the_ceiling"])
        entry["writes_observed"] += int(summary["count"])

    for entry in by_arm.values():
        maxima = entry["per_repetition_max_write_s"]
        entry["worst_write_s"] = max(maxima)
        entry["worst_write_over_the_ceiling"] = max(maxima) / ceiling
        entry["best_repetition_max_write_s"] = min(maxima)
        entry["median_of_repetition_medians_s"] = _quantile(
            entry["per_repetition_median_write_s"], 0.5)
        entry["median_of_repetition_p99_s"] = _quantile(
            entry["per_repetition_p99_write_s"], 0.5)
        entry["worst_final_flush_s"] = max(entry["per_repetition_final_flush_s"])
        entry["worst_final_flush_over_the_ceiling"] = (
            max(entry["per_repetition_final_flush_s"]) / ceiling)
        entry["worst_single_interval_s"] = max(
            max(maxima), max(entry["per_repetition_final_flush_s"]))
        entry["worst_single_interval_over_the_ceiling"] = (
            entry["worst_single_interval_s"] / ceiling)
        entry["every_repetition_stayed_under_the_ceiling"] = bool(
            entry["worst_single_interval_s"] <= ceiling)

    shipped = {name: entry for name, entry in by_arm.items() if name in SHIPPED_ARMS}
    if not shipped:
        raise Round0254WritebackError("R0254 stall verdict needs at least one shipped arm")
    best = min(shipped.values(), key=lambda entry: entry["worst_single_interval_s"])
    worst = max(shipped.values(), key=lambda entry: entry["worst_single_interval_s"])
    baseline = shipped.get(ARM_FSYNC_2GIB)

    bounded = [
        entry for entry in shipped.values()
        if entry["every_repetition_stayed_under_the_ceiling"]
    ]
    return {
        "schema": "round0254-write-stall-verdict-v1",
        "registered_ceiling_s": ceiling,
        "stall_size_bytes": STALL_SIZE_BYTES,
        "write_block_bytes": WRITE_BLOCK_BYTES,
        "by_arm": by_arm,
        "shipped_arms": sorted(shipped),
        "best_shipped_arm": best["arm"],
        "best_shipped_worst_single_interval_s": best["worst_single_interval_s"],
        "best_shipped_worst_single_interval_over_the_ceiling":
            best["worst_single_interval_over_the_ceiling"],
        "worst_shipped_arm": worst["arm"],
        "worst_shipped_worst_single_interval_s": worst["worst_single_interval_s"],
        "worst_shipped_worst_single_interval_over_the_ceiling":
            worst["worst_single_interval_over_the_ceiling"],
        "r0253_shipped_cadence_worst_single_interval_s": (
            None if baseline is None else baseline["worst_single_interval_s"]),
        "improvement_over_the_r0253_cadence": (
            None if baseline is None or best["worst_single_interval_s"] <= 0
            else baseline["worst_single_interval_s"] / best["worst_single_interval_s"]),
        "arms_whose_every_repetition_stayed_under_the_ceiling":
            sorted(entry["arm"] for entry in bounded),
        "something_bounds_the_stall_below_the_ceiling": bool(bounded),
        "what_this_is_not": (
            "Not a distribution-free bound and not a guarantee. Each arm is a "
            "finite number of repetitions of one file size on one device on one "
            "day; the statistic reported is the maximum observed, which is a "
            "lower bound on the true maximum and rises with the number of draws. "
            "An arm listed in "
            "`arms_whose_every_repetition_stayed_under_the_ceiling` has not been "
            "shown to be bounded — it has been shown not to have breached yet."
        ),
    }


def dirty_page_settings() -> dict[str, Any]:
    """The vm sysctls that govern writeback, read and published, never written.

    Tuning these is the fourth candidate remedy and this round does not attempt
    it: they are global, root-owned, and shared with every other job on this
    workstation.  A measurement taken under a modified `vm.dirty_*` would not
    describe the machine the ladder runs on, and changing them is not a change a
    round may make on the owner's behalf.
    """
    names = (
        "dirty_ratio", "dirty_background_ratio", "dirty_bytes",
        "dirty_background_bytes", "dirty_expire_centisecs",
        "dirty_writeback_centisecs",
    )
    values: dict[str, Any] = {}
    for name in names:
        try:
            with open(f"/proc/sys/vm/{name}", "r", encoding="utf-8") as handle:
                values[name] = handle.read().strip()
        except OSError:
            values[name] = None
    return {
        "schema": "round0254-dirty-page-settings-v1",
        "sysctl": values,
        "read_only": True,
        "why_they_are_not_tuned": (
            "Global root-owned settings on a shared workstation. A round that "
            "changed them would measure a machine the ladder does not run on, "
            "and would change every other job's I/O behaviour for the duration."
        ),
    }


# --------------------------------------------------------------------------- #
# the guard, and its positive control, which call the SAME function
# --------------------------------------------------------------------------- #


def write_loop_polls(source: str | None = None) -> dict[str, Any]:
    """Does the write loop contain an abort read, structurally?

    `source` exists so the positive control can plant the defect and run it
    through **this** function rather than re-implementing the walk in the test
    body.  review-0253-01 §I blocked R0253's equivalent claim for exactly that:
    `test_positive_control_a_write_loop_without_a_poll_is_caught` parsed a
    planted string and then walked it itself, so it would have passed had the
    guard returned `True` unconditionally.  With `source=None` this reads its own
    module's shipped writer, which is what the queue guard checks.
    """
    text = textwrap.dedent(
        inspect.getsource(write_arm) if source is None else source
    )
    tree = ast.parse(text)
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        raise Round0254WritebackError("R0254 write-loop guard was given no function")
    function = functions[0]
    write_loops = [
        loop for loop in ast.walk(function)
        if isinstance(loop, (ast.While, ast.For))
        and any(isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "write"
                for inner in ast.walk(loop))
    ]
    polled: list[int] = []
    for loop in write_loops:
        for statement in loop.body:
            for node in ast.walk(statement):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "poll"):
                    polled.append(int(getattr(node, "lineno", -1)))
    return {
        "schema": "round0254-write-loop-poll-guard-v1",
        "checked": "the shipped writer" if source is None else "planted source",
        "write_loops_found": len(write_loops),
        "poll_calls_inside_a_write_loop": len(polled),
        "poll_call_linenos": sorted(polled),
        "the_write_loop_polls": bool(write_loops) and bool(polled),
        "declared_sites": list(ABORT_POLL_SITES),
    }


def assert_write_loop_polls(source: str | None = None) -> dict[str, Any]:
    guard = write_loop_polls(source)
    if not guard["the_write_loop_polls"]:
        raise Round0254WritebackError(
            "R0254: the write loop contains no abort read. This is the "
            "review-0252-01 §H shape: an interval no census can see, measured at "
            "144.50039366001147 s = 57.55x the registered ceiling on R0252's top "
            "rung."
        )
    return guard


__all__ = [
    "ABORT_POLL_SITES",
    "ABORT_POLL_SITE_BLOCK",
    "ABORT_POLL_SITE_FINAL",
    "ABORT_POLL_SITE_FLUSH",
    "ARMS",
    "ARM_FSYNC_2GIB",
    "ARM_FSYNC_256MIB",
    "ARM_FSYNC_32MIB",
    "ARM_FSYNC_FINAL_ONLY",
    "ARM_O_DIRECT",
    "ARM_SYNC_FILE_RANGE",
    "ARM_UNPOLLED_CONTROL",
    "CADENCE_256MIB",
    "CADENCE_2GIB",
    "CADENCE_32MIB",
    "ROUND_ID",
    "Round0254WritebackError",
    "SHIPPED_ARMS",
    "STALL_SIZE_BYTES",
    "WRITE_BLOCK_BYTES",
    "arm_schedule",
    "assert_write_loop_polls",
    "dirty_page_settings",
    "interval_summary",
    "registered_ceiling_s",
    "stall_verdict",
    "sync_file_range",
    "write_arm",
    "write_loop_polls",
]
