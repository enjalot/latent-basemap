"""Full-content identities used by queue admission and scientific artifacts.

The helpers in this module deliberately avoid sampled fingerprints.  Admission
artifacts are relatively small compared with a failed training queue, so every
declared file is streamed in full and directories are represented by an ordered
list of their member signatures.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Iterable


# --------------------------------------------------------------------------- #
# cooperative-abort reads between hash chunks (R0252, additive)
# --------------------------------------------------------------------------- #
#
# Every loop below already streams in bounded chunks.  Until R0252 none of them
# looked at anything between chunks, so one `sha256` over a large substrate was a
# single interval in which a node could not observe an abort request: R0251
# measured 1.3006 s over 3,072,000,128 B and projected 65.03 s at a 100M
# substrate, i.e. ~26x the registered 2.5109531834854018 s ceiling.
#
# The hook is a module-level callable, None by default.  With no hook installed
# each site costs one global load and one comparison per chunk (~366 of them on
# the 3 GB substrate, ~18,750 at 100M), which is not measurable against an 8 MiB
# read.  **Nothing about the digest changes**: every byte is still read in order
# and fed to the same `hashlib.sha256`, the no-follow open and the before/after
# `fstat` identity comparison are untouched, and there is no cached, partial or
# short-circuited digest anywhere.  The only new outcome is that a poll may raise,
# in which case the caller gets an exception and *no* digest at all.

ABORT_POLL_SITE_FILE_CHUNK = "artifact_identity.sha256_file chunk"
ABORT_POLL_SITE_IDENTITY_CHUNK = "artifact_identity.regular_file_identity chunk"
ABORT_POLL_SITE_ARRAY_CHUNK = "artifact_identity.ordered_array_sha256 chunk"

#: The five-tuple a test asserts against, so a removed call site is a failure.
ABORT_POLL_SITES: tuple[str, ...] = (
    ABORT_POLL_SITE_FILE_CHUNK,
    ABORT_POLL_SITE_IDENTITY_CHUNK,
    ABORT_POLL_SITE_ARRAY_CHUNK,
)

#: Installed by node code for the duration of a stage; None everywhere else.
abort_poll = None


def set_abort_poll(poll):
    """Install (or clear, with None) the between-chunk abort read; return the old."""
    global abort_poll
    if poll is not None and not callable(poll):
        raise TypeError("artifact_identity abort poll must be callable or None")
    previous = abort_poll
    abort_poll = poll
    return previous


def _poll_abort(where: str) -> None:
    """Read the installed abort hook, if any. Allocates nothing, returns nothing."""
    poll = abort_poll
    if poll is not None:
        poll(where)


def canonical_json(value) -> bytes:
    """Return the one JSON encoding used for content-bound controller fields."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def matches_cited_display_value(sealed_value: float, cited_constant: float,
                               *, decimals: int = 6) -> bool:
    """Compare a full-precision sealed value against a cited DISPLAY constant.

    The display-precision convention (a recurring cross-check pitfall in this
    program): a round file / REPORT / descriptive reference cites a metric to a
    fixed number of decimals (e.g. ``0.644414``), while the sealed artifact stores
    the same quantity at full float precision (``0.6444139401628924``).  A drift
    cross-check that compares them must round the SEALED value to the CITED
    precision *before* the equality test — comparing full precision directly
    against a 6-decimal literal can never be equal even when nothing has drifted.

    This is the round-vs-literal display-precision rule.  It first surfaced as the
    R0264 node-1 setup-fix bug (a computed floor rounded to 4 places compared
    against a 6-decimal literal), then again in the R0265 gate's R0264 legacy
    drift-check.  Centralising it here — a single ``round(sealed, decimals) ==
    cited`` — so a third round does not rediscover it.  A drift below
    ``10 ** -decimals`` is below the display precision and not meaningful for this
    kind of descriptive cross-check.

    Returns True when the sealed value, rounded to ``decimals`` places, equals the
    cited constant exactly.
    """
    return round(float(sealed_value), int(decimals)) == cited_constant


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            _poll_abort(ABORT_POLL_SITE_FILE_CHUNK)
            h.update(chunk)
    return h.hexdigest()


def _regular_file_identity(path: str | os.PathLike,
                           chunk_size: int = 8 << 20) -> tuple[int, str]:
    """Hash one regular, unaliased file through a no-follow descriptor."""
    raw = os.fspath(path)
    before = os.lstat(raw)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"unsupported input kind at regular-file boundary: {raw}")
    if before.st_nlink != 1:
        raise ValueError(f"hard-linked input is not admissible: {raw}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(raw, flags)
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or
                (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
            raise RuntimeError(f"input identity changed while opening: {raw}")
        digest = hashlib.sha256()
        while True:
            _poll_abort(ABORT_POLL_SITE_IDENTITY_CHUNK)
            chunk = os.read(fd, chunk_size)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                         "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, key) != getattr(after, key) for key in stable_fields):
            raise RuntimeError(f"input changed while hashing: {raw}")
        return int(after.st_size), digest.hexdigest()
    finally:
        os.close(fd)


def _directory_members(path: str) -> list[dict]:
    """Return every directory and file; reject links and special members."""
    root = os.path.abspath(path)
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"directory identity target is not a directory: {root}")
    members: list[dict] = []

    def visit(directory: str) -> None:
        before = os.lstat(directory)
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeError(f"directory changed during identity walk: {directory}")
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda value: value.name)
        for entry in entries:
            member = os.path.join(directory, entry.name)
            relative = os.path.relpath(member, root).replace(os.sep, "/")
            observed = os.lstat(member)
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"directory identity contains symlink: {member}")
            if stat.S_ISDIR(observed.st_mode):
                identity = sha256_bytes(canonical_json({
                    "kind": "directory", "relative_path": relative}))
                members.append({
                    "path": member, "resolved_path": member,
                    "relative_path": relative, "kind": "directory",
                    "bytes": 0, "sha256": identity,
                })
                visit(member)
            elif stat.S_ISREG(observed.st_mode):
                size, digest = _regular_file_identity(member)
                members.append({
                    "path": member, "resolved_path": member,
                    "relative_path": relative, "kind": "file",
                    "bytes": size, "sha256": digest,
                })
            else:
                kind = ("fifo" if stat.S_ISFIFO(observed.st_mode) else
                        "socket" if stat.S_ISSOCK(observed.st_mode) else
                        "device" if (stat.S_ISCHR(observed.st_mode) or
                                     stat.S_ISBLK(observed.st_mode)) else "unsupported")
                raise ValueError(
                    f"directory identity contains unsupported {kind} member: {member}")
        after = os.lstat(directory)
        if ((after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns) !=
                (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)):
            raise RuntimeError(f"directory changed during identity walk: {directory}")

    visit(root)
    return sorted(members, key=lambda value: value["relative_path"])


def ordered_array_sha256(array, row_chunk: int = 65536) -> str:
    """Hash an array's dtype, shape, and every value in row order.

    Chunking keeps lazy/memmapped matrices out of RAM while still making an
    unsampled-row mutation or row permutation change the identity.
    """
    import numpy as np

    shape = tuple(int(v) for v in array.shape)
    dtype = np.dtype(array.dtype if hasattr(array, "dtype") else np.asarray(array[:1]).dtype)
    h = hashlib.sha256()
    h.update(canonical_json({"shape": shape, "dtype": dtype.str}))
    n = len(array)
    for start in range(0, n, row_chunk):
        _poll_abort(ABORT_POLL_SITE_ARRAY_CHUNK)
        rows = np.ascontiguousarray(np.asarray(array[start:start + row_chunk]))
        h.update(rows.tobytes(order="C"))
    return h.hexdigest()


def path_signature(path: str | os.PathLike) -> dict:
    """Return a readable, full identity for a file, symlink, or directory."""
    raw = os.fspath(path)
    resolved = os.path.realpath(raw)
    if not os.path.lexists(raw):
        raise FileNotFoundError(raw)
    if os.path.islink(raw):
        target = os.readlink(raw)
        resolved_signature = path_signature(resolved)
        return {
            "path": os.path.abspath(raw),
            "resolved_path": resolved,
            "kind": "symlink",
            "target": target,
            "resolved_signature": resolved_signature,
            "sha256": sha256_bytes(canonical_json({
                "target": target,
                "resolved_kind": resolved_signature["kind"],
                "resolved_sha256": resolved_signature["sha256"],
            })),
        }
    if os.path.isfile(raw):
        size, digest = _regular_file_identity(raw)
        return {
            "path": os.path.abspath(raw),
            "resolved_path": resolved,
            "kind": "file",
            "bytes": size,
            "sha256": digest,
        }
    if os.path.isdir(raw):
        members = _directory_members(raw)
        payload = [{k: v for k, v in item.items()
                    if k in {"relative_path", "kind", "bytes", "sha256"}}
                   for item in members]
        return {
            "path": os.path.abspath(raw),
            "resolved_path": resolved,
            "kind": "directory",
            "bytes": int(sum(int(item.get("bytes", 0)) for item in members
                             if item["kind"] == "file")),
            "members": members,
            "sha256": sha256_bytes(canonical_json(payload)),
        }
    raise ValueError(f"unsupported input kind: {raw}")


def signatures(paths: Iterable[str | os.PathLike]) -> list[dict]:
    return [path_signature(path) for path in paths]


def expected_input_signature(path: str | os.PathLike, *, repo_root: str | os.PathLike | None = None) -> dict:
    """Return the compact, canonical signature stored in a gate-bound queue.

    Queue inputs intentionally support regular files and directories only.  A
    symlink can otherwise change target while retaining its declared spelling,
    which makes the meaning of "canonical path" needlessly ambiguous at an
    admission boundary.  Directory members are ordered by relative path and
    bind every member byte; empty directories are valid and bind an empty list.
    """
    raw = os.fspath(path)
    if not os.path.isabs(raw):
        if repo_root is None:
            raise ValueError(f"relative input path needs repo_root: {raw}")
        raw = os.path.join(os.fspath(repo_root), raw)
    raw = os.path.abspath(raw)
    if os.path.islink(raw):
        raise ValueError(f"unsupported queue input kind symlink: {path}")
    canonical = os.path.realpath(raw)
    sig = path_signature(canonical)
    if sig["kind"] == "symlink":
        raise ValueError(f"unsupported queue input kind symlink: {path}")
    if sig["kind"] == "file":
        return {
            "canonical_path": canonical,
            "kind": "file",
            "bytes": int(sig["bytes"]),
            "sha256": sig["sha256"],
        }
    if sig["kind"] != "directory":
        raise ValueError(f"unsupported queue input kind {sig['kind']}: {path}")
    members = []
    for member in sig["members"]:
        if member["kind"] not in {"file", "directory"}:
            raise ValueError(f"unsupported queue directory member kind: {member!r}")
        members.append({
            "relative_path": member["relative_path"],
            "kind": member["kind"],
            "bytes": int(member["bytes"]),
            "sha256": member["sha256"],
        })
    return {
        "canonical_path": canonical,
        "kind": "directory",
        "bytes": int(sum(member["bytes"] for member in members
                         if member["kind"] == "file")),
        "sha256": sig["sha256"],
        "members": members,
    }


def expected_input_signatures(paths: Iterable[str | os.PathLike], *,
                              repo_root: str | os.PathLike | None = None) -> list[dict]:
    """Create deterministic pre-gate signatures and reject canonical aliases."""
    out = [expected_input_signature(path, repo_root=repo_root) for path in paths]
    canonical = [item["canonical_path"] for item in out]
    duplicates = sorted({path for path in canonical if canonical.count(path) > 1})
    if duplicates:
        raise ValueError(f"duplicate canonical queue input path(s): {duplicates}")
    return out


_GIT_EXECUTABLE = "/usr/bin/git"


def _closed_git_environment() -> dict[str, str]:
    return {
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def git_checkout_state(repo: str | os.PathLike) -> dict:
    root = os.path.realpath(os.fspath(repo))

    def git(*args: str, check: bool = True) -> str:
        proc = subprocess.run([_GIT_EXECUTABLE, "-C", root, *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=_closed_git_environment(), close_fds=True)
        if check and proc.returncode:
            raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout

    head = git("rev-parse", "HEAD").strip()
    symbolic = git("symbolic-ref", "-q", "HEAD", check=False).strip() or None
    porcelain = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "repo": root,
        "head": head,
        "detached": symbolic is None,
        "symbolic_ref": symbolic,
        "clean": porcelain == "",
        "porcelain": porcelain.splitlines(),
        "dirty_tree_digest": sha256_bytes(porcelain.encode("utf-8")),
    }


def is_ancestor(repo: str | os.PathLike, ancestor: str, descendant: str) -> bool:
    proc = subprocess.run(
        [_GIT_EXECUTABLE, "-C", os.fspath(repo), "merge-base", "--is-ancestor",
         ancestor, descendant], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=_closed_git_environment(), close_fds=True)
    return proc.returncode == 0
