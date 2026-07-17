from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from basemap.release_preflight import (CACHE_KEYS, _canonical_freeze_sha,
                                       issue_release_preflight_receipt)


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
def round0005_clean_release(tmp_path, fresh_data_root):
    """Clean detached copy containing the complete fixture runtime closure."""
    source = Path(__file__).resolve().parents[1]
    repo = tmp_path / "round0005-clean-release"
    shutil.copytree(
        source, repo, symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"))
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
