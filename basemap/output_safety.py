"""Fail-closed helpers for new, certifying runtime outputs.

Round 0005 never repairs a result by overwriting it.  These helpers publish a
complete file with a same-directory hard-link (which is atomic and refuses an
existing destination) and create output roots with ``mkdir`` rather than
``makedirs(..., exist_ok=True)`` at the final path.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import tempfile
from typing import Any


DATA_ROOT = "/data"


def _under_data(path: str) -> bool:
    try:
        return os.path.commonpath([DATA_ROOT, os.path.abspath(path)]) == DATA_ROOT
    except ValueError:
        return False


def canonical_data_path(path: str | os.PathLike, *, label: str = "runtime path",
                        leaf_may_exist: bool = True) -> str:
    """Validate /data containment and reject every symlinked existing ancestor."""
    value = os.fspath(path)
    if not os.path.isabs(value):
        raise ValueError(f"{label} must be absolute")
    absolute = os.path.abspath(value)
    if not _under_data(absolute):
        raise ValueError(f"{label} must be contained by {DATA_ROOT}: {absolute}")
    current = DATA_ROOT
    relative = os.path.relpath(absolute, DATA_ROOT)
    for position, piece in enumerate(relative.split(os.sep)):
        if piece in {"", ".", ".."}:
            raise ValueError(f"{label} has a noncanonical component: {absolute}")
        current = os.path.join(current, piece)
        if not os.path.lexists(current):
            break
        state = os.lstat(current)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"{label} has a symlinked ancestor: {current}")
        is_leaf = position == len(relative.split(os.sep)) - 1
        if not is_leaf and not stat.S_ISDIR(state.st_mode):
            raise ValueError(f"{label} has a nondirectory ancestor: {current}")
        if is_leaf and not leaf_may_exist:
            raise FileExistsError(f"refuse existing {label}: {absolute}")
    if os.path.realpath(absolute) != absolute:
        raise ValueError(f"{label} has a noncanonical resolved spelling: {absolute}")
    return absolute


def _open_data_directory(path: str, *, create: bool,
                         mode: int = 0o755) -> int:
    """Open a /data directory through no-follow dirfds; caller owns the fd."""
    canonical_data_path(path, label="output parent")
    relative = os.path.relpath(path, DATA_ROOT)
    fd = os.open(DATA_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        for piece in relative.split(os.sep):
            if piece in {"", "."}:
                continue
            if create:
                try:
                    os.mkdir(piece, mode, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(
                piece, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=fd)
            state = os.fstat(next_fd)
            if not stat.S_ISDIR(state.st_mode):
                os.close(next_fd)
                raise ValueError(f"output parent component is not a directory: {piece}")
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise


def _ensure_data_directory(path: str, *, mode: int = 0o755) -> None:
    """Create a /data directory chain with dirfd + O_NOFOLLOW at every hop."""
    fd = _open_data_directory(path, create=True, mode=mode)
    os.close(fd)


def ensure_data_directory(path: str | os.PathLike, *,
                          label: str = "output directory",
                          mode: int = 0o755) -> str:
    """Create/validate a contained directory chain without following symlinks."""
    absolute = canonical_data_path(path, label=label)
    _ensure_data_directory(absolute, mode=mode)
    return absolute


def _prepare_parent(path: str) -> str:
    parent = os.path.dirname(path) or "."
    if _under_data(path):
        _ensure_data_directory(parent)
    else:
        # Non-/data behavior exists for small CPU library fixtures only. Every
        # Round 0005 builder validates /data before reaching this helper.
        os.makedirs(parent, exist_ok=True)
    return parent


def refuse_existing(path: str | os.PathLike, *, label: str = "output") -> str:
    """Return an absolute spelling and reject files, directories, or symlinks."""
    absolute = os.path.abspath(os.fspath(path))
    if _under_data(absolute):
        canonical_data_path(absolute, label=label)
    if os.path.lexists(absolute):
        raise FileExistsError(f"refuse existing {label}: {absolute}")
    return absolute


def create_fresh_directory(path: str | os.PathLike, *, label: str = "output root",
                           mode: int = 0o755) -> str:
    """Create one new directory while allowing already-created parent dirs."""
    absolute = refuse_existing(path, label=label)
    parent = _prepare_parent(absolute)
    if _under_data(absolute):
        parent_fd = _open_data_directory(parent, create=False)
        try:
            os.mkdir(os.path.basename(absolute), mode, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    else:
        os.mkdir(absolute, mode)
    return absolute


def require_empty_directory(path: str | os.PathLike, *, label: str = "directory") -> str:
    """Validate an existing directory that has not received any output yet."""
    absolute = os.path.abspath(os.fspath(path))
    if _under_data(absolute):
        canonical_data_path(absolute, label=label)
    if not os.path.isdir(absolute) or os.path.islink(absolute):
        raise ValueError(f"{label} must be a regular directory: {absolute}")
    entries = os.listdir(absolute)
    if entries:
        raise FileExistsError(f"refuse nonempty {label}: {absolute}: {sorted(entries)[:5]}")
    return absolute


def _new_temp_file(destination: str, *, mode: int = 0o600):
    """Create a sibling temporary file while retaining the trusted parent fd."""
    parent = _prepare_parent(destination)
    if not _under_data(destination):
        fd, tmp = tempfile.mkstemp(
            prefix=f".{os.path.basename(destination)}.tmp.", dir=parent)
        return fd, tmp, None, None
    parent_fd = _open_data_directory(parent, create=False)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    for _ in range(128):
        name = f".{os.path.basename(destination)}.tmp.{secrets.token_hex(16)}"
        try:
            fd = os.open(name, flags, mode, dir_fd=parent_fd)
            return fd, f"/proc/self/fd/{parent_fd}/{name}", parent_fd, name
        except FileExistsError:
            continue
    os.close(parent_fd)
    raise FileExistsError("could not allocate a collision-proof sibling temporary file")


def _cleanup_temp(tmp: str, parent_fd: int | None, temp_name: str | None) -> None:
    try:
        if parent_fd is not None and temp_name is not None:
            os.unlink(temp_name, dir_fd=parent_fd)
        else:
            os.unlink(tmp)
    except OSError:
        pass
    finally:
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _publish_temp_new(tmp: str, destination: str, *, immutable: bool,
                      parent_fd: int | None = None,
                      temp_name: str | None = None) -> str:
    """Atomically link a complete temp file into a destination that must not exist."""
    if immutable:
        if parent_fd is not None and temp_name is not None:
            os.chmod(temp_name, 0o444, dir_fd=parent_fd, follow_symlinks=False)
        else:
            os.chmod(tmp, 0o444)
    try:
        if parent_fd is not None and temp_name is not None:
            canonical_data_path(destination, label="atomic destination")
            os.link(temp_name, os.path.basename(destination),
                    src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                    follow_symlinks=False)
            os.fsync(parent_fd)
        else:
            os.link(tmp, destination, follow_symlinks=False)
            directory_fd = os.open(
                os.path.dirname(destination) or ".",
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        _cleanup_temp(tmp, parent_fd, temp_name)
    return destination


def atomic_write_new_bytes(path: str | os.PathLike, payload: bytes, *,
                           immutable: bool = False, mode: int = 0o644) -> str:
    """Atomically publish bytes without any last-write-wins behavior."""
    destination = refuse_existing(path)
    fd, tmp, parent_fd, temp_name = _new_temp_file(destination)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return _publish_temp_new(
            tmp, destination, immutable=immutable, parent_fd=parent_fd,
            temp_name=temp_name)
    except Exception:
        _cleanup_temp(tmp, parent_fd, temp_name)
        raise


def atomic_write_new_json(path: str | os.PathLike, value: Any, *,
                          immutable: bool = False, indent: int = 2) -> str:
    payload = (json.dumps(value, indent=indent, sort_keys=True, ensure_ascii=False)
               + "\n").encode("utf-8")
    return atomic_write_new_bytes(path, payload, immutable=immutable)


def atomic_save_new_npy(path: str | os.PathLike, value, *, immutable: bool = False) -> str:
    """Stream a NumPy array to a fresh atomically-published ``.npy`` file."""
    import numpy as np

    destination = refuse_existing(path)
    fd, tmp, parent_fd, temp_name = _new_temp_file(destination)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, value)
            handle.flush()
            os.fsync(handle.fileno())
        return _publish_temp_new(
            tmp, destination, immutable=immutable, parent_fd=parent_fd,
            temp_name=temp_name)
    except Exception:
        _cleanup_temp(tmp, parent_fd, temp_name)
        raise


def atomic_save_new_npz(path: str | os.PathLike, *, immutable: bool = False,
                        compressed: bool = False, **arrays) -> str:
    """Stream a NumPy archive and atomically publish it at an absent path."""
    import numpy as np

    destination = refuse_existing(path)
    fd, tmp, parent_fd, temp_name = _new_temp_file(destination)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        return _publish_temp_new(
            tmp, destination, immutable=immutable, parent_fd=parent_fd,
            temp_name=temp_name)
    except Exception:
        _cleanup_temp(tmp, parent_fd, temp_name)
        raise


def atomic_copy_new(source: str | os.PathLike, destination: str | os.PathLike, *,
                    immutable: bool = False, chunk_size: int = 8 << 20) -> str:
    """Copy one regular file and atomically publish it at a fresh destination."""
    source_path = os.path.abspath(os.fspath(source))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)
    destination_path = refuse_existing(destination)
    fd, tmp, parent_fd, temp_name = _new_temp_file(destination_path)
    try:
        with open(source_path, "rb") as src, os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst, length=chunk_size)
            dst.flush()
            os.fsync(dst.fileno())
        return _publish_temp_new(
            tmp, destination_path, immutable=immutable, parent_fd=parent_fd,
            temp_name=temp_name)
    except Exception:
        _cleanup_temp(tmp, parent_fd, temp_name)
        raise


def atomic_build_new_file(path: str | os.PathLike, writer, *,
                          immutable: bool = False) -> str:
    """Let ``writer(temp_path)`` build a file, then atomically publish it fresh."""
    destination = refuse_existing(path)
    fd, tmp, parent_fd, temp_name = _new_temp_file(destination)
    os.close(fd)
    try:
        writer(tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        return _publish_temp_new(
            tmp, destination, immutable=immutable, parent_fd=parent_fd,
            temp_name=temp_name)
    except Exception:
        _cleanup_temp(tmp, parent_fd, temp_name)
        raise
