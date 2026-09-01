"""Shared sys.path bootstrap for sandbox scripts — kills the recurring parents[1]/parents[2] bug.

THE BUG CLASS (2026-08-31 evolbench abort, 2026-08-2x perf_bench gate-2, and others):
a sandbox script at `<repo>/experiments/sandbox/foo.py` needs the REPO ROOT on sys.path to do
`from basemap...`. The repo root is `parents[2]`, NOT `parents[1]` (that is `experiments/`).
Scripts that import `knobs_2m` got the root for free — knobs_2m.py does the parents[2] insert as a
module-load side effect — so a wrong `parents[1]` went unnoticed. Scripts importing `from basemap...`
DIRECTLY (no knobs_2m first) crashed at runtime with ModuleNotFoundError. py_compile does NOT catch
this (imports are not executed), so an IMPORT-SMOKE is mandatory for every new script (see below).

USE (top of main(), or module level):
    from pathlib import Path; import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # make _paths itself importable
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP   # now resolves

IMPORT-SMOKE (run as part of committing any script that imports basemap/knobs):
    .venv/bin/python -c "import ast,sys; ast.parse(open('experiments/sandbox/foo.py').read())"  # syntax
    .venv/bin/python experiments/sandbox/foo.py --help  2>&1 | head   # or a guarded import path
  — the real check is that `from basemap...` executes; py_compile is NOT sufficient.
"""
from pathlib import Path
import sys


def repo_root() -> Path:
    """Repo root (latent-basemap/). This file is <repo>/experiments/sandbox/_paths.py -> parents[2]."""
    return Path(__file__).resolve().parents[2]


def ensure_paths() -> Path:
    """Idempotently put repo root, <repo>/experiments, and the sandbox dir on sys.path.
    Superset of every historical ad-hoc insert, so it is always safe to swap one in for another."""
    root = repo_root()
    sandbox = Path(__file__).resolve().parent
    for p in (str(root), str(root / "experiments"), str(sandbox)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
