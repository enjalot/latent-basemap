"""Controls for the materialised-release copy and its forced removal.

Two hazards are pinned here, both observed for real on ``gsv``:

* the per-test copy of the repository used to include the in-repo ``.venv``
  (~9.4 GB), which made a full suite cost tens of GB and about an hour; and
* deleting such a copy with a plain ``rm -rf``/``rmtree`` fails on the ``r--``
  files and ``r-x`` directories it inherits, so a reported reclaim silently did
  not happen (R0239).  The ``chmod``-before-delete now lives in the code, and
  the negative control below proves it is load-bearing.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (REPO_ROOT, SOURCE_RELEASE_EXCLUDES,  # noqa: E402
                      copy_source_release, force_remove_tree,
                      source_release_copies)


def _plant_read_only_tree(root: Path) -> Path:
    """Build the shape a copied venv + Git object store presents on removal."""
    nested = root / "lib" / "python3.12" / "site-packages" / "pkg"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("x = 1\n")
    (root / "objects").mkdir()
    (root / "objects" / "loose").write_bytes(b"object")
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o444)
    root.chmod(0o500)
    return root


def test_force_remove_tree_deletes_a_planted_read_only_tree(tmp_path):
    root = _plant_read_only_tree(tmp_path / "read-only-release")
    force_remove_tree(root)
    assert not root.exists()


def test_plain_rmtree_cannot_delete_that_tree(tmp_path):
    """Negative control: without the chmod the removal really does fail."""
    root = _plant_read_only_tree(tmp_path / "read-only-release")
    with pytest.raises(PermissionError):
        shutil.rmtree(root)
    assert root.exists(), "the failed removal must have left the tree behind"
    force_remove_tree(root)
    assert not root.exists()


def test_force_remove_tree_is_idempotent_and_keeps_symlinked_targets(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("kept\n")
    root = tmp_path / "with-symlink"
    (root / "inner").mkdir(parents=True)
    (root / "inner" / "link").symlink_to(outside)
    (root / "inner").chmod(0o500)
    force_remove_tree(root)
    assert not root.exists()
    assert outside.read_text() == "kept\n"
    force_remove_tree(root)  # a second removal is a no-op, not an error


def test_source_copy_excludes_environments_caches_and_git(tmp_path):
    source = tmp_path / "source"
    (source / ".venv" / "lib").mkdir(parents=True)
    (source / ".venv" / "lib" / "big.so").write_bytes(b"0" * 4096)
    (source / "venv").mkdir()
    (source / "venv" / "pyvenv.cfg").write_text("home = /usr\n")
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00")
    (source / "basemap").mkdir()
    (source / "basemap" / "kept.py").write_text("kept = True\n")
    (source / "basemap" / "stale.pyc").write_bytes(b"\x00")

    destination = copy_source_release(tmp_path / "copy", source=source)

    assert (destination / "basemap" / "kept.py").read_text() == "kept = True\n"
    for excluded in (".venv", "venv", ".git", "__pycache__",
                     Path("basemap") / "stale.pyc"):
        assert not (destination / excluded).exists(), excluded
    assert ".venv" in SOURCE_RELEASE_EXCLUDES


def test_materialised_release_is_source_complete_and_venv_free(
        source_release_copy):
    """The copy stays a complete source release: only the venv is dropped."""
    repo = source_release_copy("release-copy")
    assert not (repo / ".venv").exists()
    assert not (repo / ".git").exists()
    for member in ("basemap/run_controller.py",
                   "basemap/source_closure.py",
                   "basemap/pumap/parametric_umap/models/mlp.py",
                   "experiments/run_round0005_fixture.py"):
        assert (repo / member).is_file(), member
    if (REPO_ROOT / ".venv").is_dir():
        planted = sum(1 for _ in (REPO_ROOT / ".venv").rglob("*"))
        assert planted > 0
        assert not any(repo.rglob(".venv"))


def test_copied_release_is_isolated_from_the_source_checkout(source_release_copy):
    """Mutating the copy must never reach the checkout the suite runs from."""
    repo = source_release_copy("isolation-copy")
    target = repo / "basemap" / "pumap" / "parametric_umap" / "models" / "mlp.py"
    original = (REPO_ROOT / "basemap/pumap/parametric_umap/models/mlp.py").read_text()
    target.write_text(target.read_text() + "\n# post-approval drift\n")
    assert target.read_text() != original
    assert (REPO_ROOT / "basemap/pumap/parametric_umap/models/mlp.py").read_text() == original
    assert not os.path.samefile(target.parent, REPO_ROOT / "basemap/pumap/parametric_umap/models")


def test_release_copies_are_removed_even_when_read_only(tmp_path):
    """The teardown removes the copy, including entries turned read-only."""
    with source_release_copies(tmp_path) as make:
        repo = make("teardown-copy")
        assert (repo / "basemap" / "run_controller.py").is_file()
        frozen = repo / "basemap" / "run_controller.py"
        frozen.chmod(0o444)
        (repo / "basemap").chmod(0o500)
    assert not repo.exists()
