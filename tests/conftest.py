from __future__ import annotations

import contextlib
import os
import json
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from basemap.release_preflight import (CACHE_KEYS, _canonical_freeze_sha,
                                       issue_release_preflight_receipt)


REPO_ROOT = Path(__file__).resolve().parents[1]

#: Never copied into a materialised release.  A checkout of this repo carries
#: its virtualenv *inside* the tree (``.venv``, ~9.4 GB, ~39k files), and a
#: fixture that copies the tree per test therefore paid ~14 GB and minutes of
#: Git hashing per test — the reason three rounds abandoned a full suite run
#: mid-way.  No test executes the copied tree's interpreter: every test runs
#: under ``sys.executable`` from the outer environment and only ever needs the
#: *sources* (import closure, tracked-file set, Git identity) of the copy.
SOURCE_RELEASE_EXCLUDES = (
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "*.pyc",
)


def force_remove_tree(path: str | os.PathLike) -> None:
    """``rm -rf`` that first restores owner write permission on every entry.

    A copied release inherits ``r--`` files and ``r-x`` directories from the
    virtualenv and from Git's object store, so a plain ``rmtree``/``rm -rf``
    fails part-way and a reported reclaim silently never happened (R0239).
    Clearing the modes is part of the deletion here, not a runbook step.
    """
    root = os.fspath(path)
    if not os.path.lexists(root):
        return
    os.chmod(root, os.stat(root).st_mode | stat.S_IRWXU)
    for parent, directories, files in os.walk(root, topdown=True):
        for name in directories:
            target = os.path.join(parent, name)
            if not os.path.islink(target):
                os.chmod(target, os.stat(target).st_mode | stat.S_IRWXU)
        for name in files:
            target = os.path.join(parent, name)
            if not os.path.islink(target):
                os.chmod(target, os.stat(target).st_mode | stat.S_IWUSR | stat.S_IRUSR)
    shutil.rmtree(root)


def copy_source_release(destination: str | os.PathLike,
                        source: str | os.PathLike = REPO_ROOT) -> Path:
    """Copy the repo's *source* tree only; never its in-repo virtualenv."""
    path = Path(destination)
    shutil.copytree(
        source, path, symlinks=True,
        ignore=shutil.ignore_patterns(*SOURCE_RELEASE_EXCLUDES))
    return path


@contextlib.contextmanager
def source_release_copies(parent: str | os.PathLike):
    """Hand out source-only repo copies under ``parent`` and remove them after.

    Removing at teardown (rather than at basetemp expiry) keeps the peak cost
    of a full suite at one copy instead of one copy per test.
    """
    created: list[Path] = []

    def make(name: str) -> Path:
        destination = copy_source_release(Path(parent) / name)
        created.append(destination)
        return destination

    try:
        yield make
    finally:
        for destination in created:
            force_remove_tree(destination)


@pytest.fixture
def source_release_copy(tmp_path):
    """Factory for detached source-only repo copies, removed at teardown."""
    with source_release_copies(tmp_path) as make:
        yield make


@pytest.fixture
def fresh_data_root():
    """Collision-proof /data root; intentionally retained after the test run."""
    parent = os.environ.get(
        "BASEMAP_TEST_DATA_PARENT",
        "/data/latent-basemap/runs/round-0005/pytest")
    os.makedirs(parent, exist_ok=True)
    path = os.path.join(parent, f"{os.getpid()}-{uuid.uuid4().hex}")
    os.mkdir(path)
    return path


@pytest.fixture
def clean_release_evidence(tmp_path, fresh_data_root):
    """One replayable pushed-release receipt backed by a tiny detached repo."""
    repo = tmp_path / "release-repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email",
                           "fixture@example.invalid"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name",
                           "Fixture"])
    (repo / "tracked.txt").write_text("release\n")
    subprocess.check_call(["git", "-C", str(repo), "add", "tracked.txt"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "release"])
    release = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    remote = tmp_path / "release-remote.git"
    subprocess.check_call(["git", "init", "--bare", "-q", str(remote)])
    subprocess.check_call(["git", "-C", str(repo), "remote", "add", "origin", str(remote)])
    subprocess.check_call([
        "git", "-C", str(repo), "push", "-q", "origin", f"{release}:refs/heads/main"])
    pushed_ref = "refs/remotes/origin/main"
    subprocess.check_call(["git", "-C", str(repo), "fetch", "-q", "origin", "main"])
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", "-q", release])

    root = Path(fresh_data_root) / "release-evidence"
    root.mkdir()
    freeze = root / "freeze.txt"
    freeze.write_text("fixture-package==1\n")
    environment = root / "environment.json"
    environment.write_text(json.dumps({
        "freeze_file": str(freeze),
        "freeze_sha256": _canonical_freeze_sha(str(freeze)),
        "identity_sha256": "e" * 64,
        "venv_path": sys.prefix,
    }) + "\n")
    cache = {"PYTHONDONTWRITEBYTECODE": "1"}
    for key in CACHE_KEYS:
        path = root / "cache" / key.lower()
        path.mkdir(parents=True)
        cache[key] = str(path)
    receipt_path = root / "release-preflight.json"
    receipt = issue_release_preflight_receipt(
        str(receipt_path), integration_repo=str(repo), release_sha=release,
        implementation_commits=[release], pushed_ref=pushed_ref,
        run_checkout=str(repo), environment_manifest=str(environment),
        cache_environment=cache)
    return {
        "repo": str(repo), "root": str(root), "release_sha": release,
        "receipt_path": str(receipt_path), "receipt": receipt,
    }


@pytest.fixture
def round0005_clean_release(tmp_path, fresh_data_root, source_release_copy):
    """Clean detached copy containing the complete fixture runtime closure."""
    repo = source_release_copy("round0005-clean-release")
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email",
                           "fixture@example.invalid"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Fixture"])
    subprocess.check_call(["git", "-C", str(repo), "add", "-f", "-A"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "fixture release"])
    release = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    remote = tmp_path / "round0005-remote.git"
    subprocess.check_call(["git", "init", "--bare", "-q", str(remote)])
    subprocess.check_call(["git", "-C", str(repo), "remote", "add", "origin", str(remote)])
    subprocess.check_call([
        "git", "-C", str(repo), "push", "-q", "origin", f"{release}:refs/heads/main"])
    pushed_ref = "refs/remotes/origin/main"
    subprocess.check_call(["git", "-C", str(repo), "fetch", "-q", "origin", "main"])
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", "-q", release])
    evidence = Path(fresh_data_root) / "round0005-clean-release"
    evidence.mkdir()
    freeze = evidence / "freeze.txt"
    freeze.write_text("fixture-package==1\n")
    environment = evidence / "environment.json"
    environment.write_text(json.dumps({
        "freeze_file": str(freeze),
        "freeze_sha256": _canonical_freeze_sha(str(freeze)),
        "identity_sha256": "e" * 64,
        "venv_path": sys.prefix,
    }) + "\n")
    args = SimpleNamespace(
        repo_root=str(repo), integration_repo=str(repo), release_sha=release,
        implementation_commit=[release], pushed_ref=pushed_ref,
        round_sha256="a" * 64, environment_manifest=str(environment))
    return args, evidence
