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
import subprocess
from pathlib import Path
from typing import Iterable


def canonical_json(value) -> bytes:
    """Return the one JSON encoding used for content-bound controller fields."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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
        return {
            "path": os.path.abspath(raw),
            "resolved_path": resolved,
            "kind": "file",
            "bytes": int(os.path.getsize(raw)),
            "sha256": sha256_file(raw),
        }
    if os.path.isdir(raw):
        members = []
        for member in sorted(Path(raw).rglob("*")):
            if member.is_file() or member.is_symlink():
                sig = path_signature(member)
                sig["relative_path"] = str(member.relative_to(raw))
                members.append(sig)
        payload = [{k: v for k, v in item.items()
                    if k in {"relative_path", "kind", "bytes", "sha256", "target"}}
                   for item in members]
        return {
            "path": os.path.abspath(raw),
            "resolved_path": resolved,
            "kind": "directory",
            "members": members,
            "sha256": sha256_bytes(canonical_json(payload)),
        }
    raise ValueError(f"unsupported input kind: {raw}")


def signatures(paths: Iterable[str | os.PathLike]) -> list[dict]:
    return [path_signature(path) for path in paths]


def git_checkout_state(repo: str | os.PathLike) -> dict:
    root = os.path.realpath(os.fspath(repo))

    def git(*args: str, check: bool = True) -> str:
        proc = subprocess.run(["git", "-C", root, *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        ["git", "-C", os.fspath(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode == 0
