"""R0253 — the write path, which polled nothing and therefore emitted nothing.

review-0252-01 §H, the round's severe finding:

    `_write_sized_file` contains no abort read and the gate is not constructed
    until after it returns, so the census structurally cannot see the interval.
    The shipped node still has it -- the top rung's write is
    `144.50039366001147` s `= 57.55x` the ceiling.  Neither the round nor the
    result reports this number.

R0252's own attempt-1 cooperative stop took `93.135091` s `= 37.09x` the ceiling
for exactly this reason: the flag was written while the node was inside the write
loop, and the first gate read did not happen until the loop had finished.

This module is the write loop with a poll between blocks.  Byte semantics are
R0252's, unchanged and deliberately so, because the round's control arm is
R0252's writer and an arm that wrote different bytes would not be a control:

* the same `WRITE_BLOCK_BYTES` 64 MiB pseudorandom block from
  `np.random.default_rng(seed)`, reused for every block,
* the same `os.write` loop with no `fallocate`, no `ftruncate` and no seek past
  EOF, so every byte is issued to the device and the file cannot be sparse,
* the same `os.fsync` + `POSIX_FADV_DONTNEED` every `WRITE_FLUSH_EVERY_BYTES`,
  and once more at the end.

The only difference between the two arms is one global load and one comparison
per 64 MiB block.  Both arms are required to produce the identical digest at the
identical size, which is the receipt that the poll changed nothing.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap
import time
from typing import Any

import numpy as np


class Round0253WriteError(RuntimeError):
    """The registered R0253 write-path contract changed."""


#: 64 MiB.  R0252's value, kept so the arms differ only in the poll.
WRITE_BLOCK_BYTES = 64 << 20
#: 2 GiB.  R0252's value.
WRITE_FLUSH_EVERY_BYTES = 2 << 30

ABORT_POLL_SITE_WRITE_BLOCK = "round0253_write_path.write_sized_file block"
ABORT_POLL_SITE_WRITE_FLUSH = "round0253_write_path.write_sized_file flush"

#: Asserted by a contract test against this module's own AST, so a deleted call
#: site is a test failure rather than a silently restored 144.5 s interval.
ABORT_POLL_SITES: tuple[str, ...] = (
    ABORT_POLL_SITE_WRITE_BLOCK,
    ABORT_POLL_SITE_WRITE_FLUSH,
)

ARM_UNPOLLED = "unpolled"
ARM_POLLED = "polled"


def write_sized_file(
    path: str,
    total_bytes: int,
    *,
    seed: int,
    poll: Any = None,
    digest: bool = True,
) -> dict[str, Any]:
    """Create a real, fully-allocated file of exactly `total_bytes`.

    With `poll=None` this is R0252's `_write_sized_file` byte for byte and read
    for read -- the control arm.  With a callable `poll` it reads the abort hook
    once per 64 MiB block and once per flush, which is the fix.

    `digest` hashes the block stream as it is written, at no extra IO, so the two
    arms can be shown to have produced identical content without a second pass.
    """
    if total_bytes <= 0:
        raise Round0253WriteError(f"R0253 write needs a positive size, got {total_bytes}")
    if poll is not None and not callable(poll):
        raise Round0253WriteError("R0253 write-path poll must be callable or None")
    rng = np.random.default_rng(seed)
    block = rng.integers(0, 256, size=WRITE_BLOCK_BYTES, dtype=np.uint8).tobytes()
    hasher = hashlib.sha256() if digest else None
    started = time.monotonic()
    written = 0
    since_flush = 0
    blocks = 0
    flushes = 0
    polls = 0
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        while written < total_bytes:
            if poll is not None:
                poll(ABORT_POLL_SITE_WRITE_BLOCK)
                polls += 1
            take = min(WRITE_BLOCK_BYTES, total_bytes - written)
            view = block if take == WRITE_BLOCK_BYTES else block[:take]
            offset = 0
            while offset < take:
                offset += os.write(fd, view[offset:])
            if hasher is not None:
                hasher.update(view)
            written += take
            blocks += 1
            since_flush += take
            if since_flush >= WRITE_FLUSH_EVERY_BYTES:
                if poll is not None:
                    poll(ABORT_POLL_SITE_WRITE_FLUSH)
                    polls += 1
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                since_flush = 0
                flushes += 1
        if poll is not None:
            poll(ABORT_POLL_SITE_WRITE_FLUSH)
            polls += 1
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        flushes += 1
    finally:
        os.close(fd)
    wall = time.monotonic() - started
    observed = os.path.getsize(path)
    if observed != total_bytes:
        raise Round0253WriteError(
            f"R0253 wrote {observed} B to {path}, asked for {total_bytes}"
        )
    stat = os.stat(path)
    return {
        "path": path,
        "bytes": observed,
        "arm": ARM_POLLED if poll is not None else ARM_UNPOLLED,
        "write_wall_s": wall,
        "write_throughput_bytes_per_s": (observed / wall) if wall > 0 else None,
        "block_bytes": WRITE_BLOCK_BYTES,
        "blocks_written": blocks,
        "flushes": flushes,
        "abort_reads_inside_the_write": polls,
        "content_sha256": None if hasher is None else hasher.hexdigest(),
        "allocated_blocks_512b": int(stat.st_blocks),
        "fully_allocated": bool(stat.st_blocks * 512 >= observed),
        "sparse_by_construction": False,
        "how_allocation_is_evidenced": (
            "os.stat().st_blocks read from the inode, not a hard-coded literal. "
            "review-0252-01 §B.2 refused R0252's `sparse: false` field as evidence "
            "for exactly that reason and accepted st_blocks; this reports only the "
            "stat."
        ),
    }


# --------------------------------------------------------------------------- #
# the guard: the write loop polls, and a test plants the defect
# --------------------------------------------------------------------------- #


def write_loop_polls() -> dict[str, Any]:
    """Read this module's own AST: does the write loop contain an abort read?

    Structural, not textual: the poll call must sit INSIDE the `while` that
    issues the writes.  A poll before the loop is the R0252 shape -- one read,
    then a 144.5 s interval -- and this returns False for it.
    """
    source = textwrap.dedent(inspect.getsource(write_sized_file))
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise Round0253WriteError("R0253 write-path guard could not parse the writer")
    loops = [node for node in ast.walk(function) if isinstance(node, ast.While)]
    write_loops = [
        loop for loop in loops
        if any(isinstance(inner, ast.Call)
               and isinstance(inner.func, ast.Attribute)
               and inner.func.attr == "write"
               for inner in ast.walk(loop))
    ]
    polled = []
    for loop in write_loops:
        for inner in loop.body:
            for node in ast.walk(inner):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "poll":
                    polled.append(int(getattr(node, "lineno", -1)))
    return {
        "schema": "round0253-write-loop-poll-guard-v1",
        "write_loops_found": len(write_loops),
        "poll_calls_inside_a_write_loop": len(polled),
        "the_write_loop_polls": bool(write_loops) and bool(polled),
        "declared_sites": list(ABORT_POLL_SITES),
    }


def assert_write_loop_polls() -> dict[str, Any]:
    guard = write_loop_polls()
    if not guard["the_write_loop_polls"]:
        raise Round0253WriteError(
            "R0253: the write loop contains no abort read. This is the "
            "review-0252-01 §H shape: a single interval no census can see, "
            "measured at 144.50039366001147 s = 57.55x the registered ceiling on "
            "R0252's top rung."
        )
    return guard


__all__ = [
    "ABORT_POLL_SITES",
    "ABORT_POLL_SITE_WRITE_BLOCK",
    "ABORT_POLL_SITE_WRITE_FLUSH",
    "ARM_POLLED",
    "ARM_UNPOLLED",
    "Round0253WriteError",
    "WRITE_BLOCK_BYTES",
    "WRITE_FLUSH_EVERY_BYTES",
    "assert_write_loop_polls",
    "write_loop_polls",
    "write_sized_file",
]
