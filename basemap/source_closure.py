"""Deterministic tracked Python import closure for the Round 0005 runtime.

The old queue used a hand-maintained list and consequently omitted package
initializers and modules imported below ``parametric_umap.core``.  This module
walks local imports from the exact controller and six canonical node scripts,
adds every existing package initializer, verifies that every member is tracked,
and returns content signatures suitable for the manifest-global registry.
"""
from __future__ import annotations

import ast
import os
import subprocess
from collections import deque
from typing import Iterable

from .artifact_identity import canonical_json, expected_input_signature, sha256_bytes


ROUND0005_RUNTIME_ENTRYPOINTS = (
    "basemap/gate_preparation.py",
    "basemap/queue_admission.py",
    "basemap/release_preflight.py",
    "basemap/round0005_program.py",
    "basemap/roundwatch_gate.py",
    "basemap/run_controller.py",
    "basemap/source_closure.py",
    "experiments/prepare_round0005_gate.py",
    "experiments/prepare_round0005_queue.py",
    "experiments/score_complete_panel.py",
    "experiments/compare_panel_cache.py",
    "experiments/round0005_performance_gate.py",
    "experiments/calibrate_jina_embedding.py",
    "experiments/run_round0005_seal_canary.py",
)

# This is a second, target-specific closure rather than a widening of the
# historical Round-0005 program.  It binds the one Round-0014 node executable
# that the queue will invoke after owner approval.
ROUND0014_RUNTIME_ENTRYPOINTS = (
    "basemap/gate_preparation.py",
    "basemap/queue_admission.py",
    "basemap/release_preflight.py",
    "basemap/round0014_admission.py",
    "basemap/round0014_program.py",
    "basemap/round0014_staging.py",
    "basemap/round0014_transform.py",
    "basemap/roundwatch_gate.py",
    "basemap/run_controller.py",
    "basemap/source_closure.py",
    "experiments/run_round0014_node.py",
)

# The additive Round-0015 closure keeps the reviewed scientific implementation
# but binds only its new exact-service construction, target wrapper, staging,
# and lease-release behavior.
ROUND0015_RUNTIME_ENTRYPOINTS = (
    "basemap/gate_preparation.py",
    "basemap/queue_admission.py",
    "basemap/release_preflight.py",
    "basemap/round0014_program.py",
    "basemap/round0014_transform.py",
    "basemap/round0015_admission.py",
    "basemap/round0015_program.py",
    "basemap/round0015_service.py",
    "basemap/round0015_staging.py",
    "basemap/roundwatch_gate.py",
    "basemap/run_controller.py",
    "basemap/source_closure.py",
    "experiments/run_round0014_node.py",
    "experiments/run_round0015_node.py",
)

# Dynamic imports and package-level exports cannot all be discovered from an
# AST edge.  Keep this list narrowly focused and assert it is present in the
# generated closure.  In particular these are the omissions called out by the
# Round 0005 audits.
ROUND0005_REQUIRED_DYNAMIC_SOURCES = (
    "basemap/pumap/parametric_umap/__init__.py",
    "basemap/pumap/parametric_umap/core.py",
    "basemap/pumap/parametric_umap/perf.py",
    "basemap/pumap/parametric_umap/models/__init__.py",
    "basemap/pumap/parametric_umap/models/mlp.py",
    "basemap/pumap/parametric_umap/datasets/__init__.py",
    "basemap/pumap/parametric_umap/datasets/covariates_datasets.py",
    "basemap/pumap/parametric_umap/datasets/edge_dataset.py",
    "basemap/pumap/parametric_umap/datasets/edge_list_dataset.py",
    "basemap/pumap/parametric_umap/utils/__init__.py",
    "basemap/pumap/parametric_umap/utils/data_prefetcher.py",
    "basemap/pumap/parametric_umap/utils/graph.py",
    "basemap/pumap/parametric_umap/utils/losses.py",
)
SOURCE_CLOSURE_GIT = "/usr/bin/git"


def _closed_git_environment() -> dict[str, str]:
    return {
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _module_candidates(root: str, module: str) -> list[str]:
    stem = os.path.join(root, *module.split("."))
    return [stem + ".py", os.path.join(stem, "__init__.py")]


def _resolve_module(root: str, module: str) -> str | None:
    if not module:
        return None
    for candidate in _module_candidates(root, module):
        if os.path.isfile(candidate) and not os.path.islink(candidate):
            return os.path.relpath(candidate, root).replace(os.sep, "/")
    return None


def _module_name(relative: str) -> tuple[str, bool]:
    parts = relative.removesuffix(".py").split("/")
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _package_initializers(root: str, relative: str) -> list[str]:
    pieces = relative.split("/")[:-1]
    result = []
    for end in range(1, len(pieces) + 1):
        candidate = "/".join([*pieces[:end], "__init__.py"])
        absolute = os.path.join(root, candidate)
        if os.path.isfile(absolute) and not os.path.islink(absolute):
            result.append(candidate)
    return result


def _local_imports(root: str, relative: str) -> set[str]:
    absolute = os.path.join(root, relative)
    with open(absolute, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=relative)
    current_module, is_package = _module_name(relative)
    current_parts = current_module.split(".") if current_module else []
    package_parts = current_parts if is_package else current_parts[:-1]
    resolved: set[str] = set()

    def add(module: str) -> None:
        candidate = _resolve_module(root, module)
        if candidate:
            resolved.add(candidate)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pieces = alias.name.split(".")
                # Importing a package executes each initializer on the path.
                for end in range(1, len(pieces) + 1):
                    add(".".join(pieces[:end]))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                if keep < 0:
                    continue
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            add(base)
            for alias in node.names:
                if alias.name != "*":
                    add(".".join(part for part in (base, alias.name) if part))
    return resolved


def runtime_source_closure(repo_root: str,
                           entrypoints: Iterable[str] = ROUND0005_RUNTIME_ENTRYPOINTS
                           ) -> tuple[str, ...]:
    """Return the sorted full local import closure for the exact runtime."""
    root = os.path.realpath(repo_root)
    seeds = [*entrypoints, *ROUND0005_REQUIRED_DYNAMIC_SOURCES]
    pending = deque(dict.fromkeys(seeds))
    seen: set[str] = set()
    while pending:
        relative = pending.popleft().replace(os.sep, "/")
        if relative in seen:
            continue
        absolute = os.path.join(root, relative)
        if not os.path.isfile(absolute) or os.path.islink(absolute):
            raise FileNotFoundError(f"Round 0005 runtime source is missing: {absolute}")
        seen.add(relative)
        for initializer in _package_initializers(root, relative):
            if initializer not in seen:
                pending.append(initializer)
        for imported in sorted(_local_imports(root, relative)):
            if imported not in seen:
                pending.append(imported)
    missing_dynamic = sorted(set(ROUND0005_REQUIRED_DYNAMIC_SOURCES) - seen)
    if missing_dynamic:
        raise RuntimeError(f"generated runtime closure omitted required modules: {missing_dynamic}")
    return tuple(sorted(seen))


def _tracked_paths(repo_root: str) -> set[str]:
    proc = subprocess.run(
        [SOURCE_CLOSURE_GIT, "-C", repo_root, "ls-files", "-z"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=_closed_git_environment(), close_fds=True)
    if proc.returncode:
        raise RuntimeError(
            f"git ls-files failed while generating source closure: "
            f"{proc.stderr.decode(errors='replace').strip()}")
    return {value.decode("utf-8") for value in proc.stdout.split(b"\0") if value}


def source_closure_receipt(
        repo_root: str, *, require_tracked: bool = True,
        entrypoints: Iterable[str] = ROUND0005_RUNTIME_ENTRYPOINTS) -> dict:
    """Generate signatures and one identity for the complete runtime closure."""
    root = os.path.realpath(repo_root)
    entrypoints = tuple(entrypoints)
    if entrypoints != ROUND0005_RUNTIME_ENTRYPOINTS:
        raise RuntimeError(
            "production source closure entrypoints must equal "
            "ROUND0005_RUNTIME_ENTRYPOINTS exactly and in order")
    return _source_closure_receipt(
        root, require_tracked=require_tracked, entrypoints=entrypoints,
        schema="round0005_runtime_source_closure.v3")


def round0014_source_closure_receipt(
        repo_root: str, *, require_tracked: bool = True) -> dict:
    """Bind the exact tracked runtime for the one Round 0014 queue."""
    return _source_closure_receipt(
        os.path.realpath(repo_root), require_tracked=require_tracked,
        entrypoints=ROUND0014_RUNTIME_ENTRYPOINTS,
        schema="round0014-runtime-source-closure-v1")


def round0015_source_closure_receipt(
        repo_root: str, *, require_tracked: bool = True) -> dict:
    """Bind the exact tracked runtime for the one Round 0015 queue."""
    return _source_closure_receipt(
        os.path.realpath(repo_root), require_tracked=require_tracked,
        entrypoints=ROUND0015_RUNTIME_ENTRYPOINTS,
        schema="round0015-runtime-source-closure-v1")


def _source_closure_receipt(
        repo_root: str, *, require_tracked: bool,
        entrypoints: tuple[str, ...], schema: str) -> dict:
    """Implementation shared only with the explicit CPU-fixture closure."""
    root = os.path.realpath(repo_root)
    closure = runtime_source_closure(root, entrypoints)
    if require_tracked:
        tracked = _tracked_paths(root)
        untracked = sorted(set(closure) - tracked)
        if untracked:
            raise RuntimeError(
                f"Round 0005 runtime closure contains untracked sources: {untracked}")
    members = [
        {"relative_path": relative,
         "signature": expected_input_signature(os.path.join(root, relative))}
        for relative in closure
    ]
    body = {
        "schema": schema,
        "repo_root": root,
        "entrypoints": list(entrypoints),
        "members": members,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _source_closure_receipt_fixture_only(
        repo_root: str, *, entrypoints: Iterable[str]) -> dict:
    """Private CUDA-hidden fixture closure; never valid for production admission."""
    values = tuple(entrypoints)
    if values == ROUND0005_RUNTIME_ENTRYPOINTS:
        raise RuntimeError("fixture source closure must be unmistakably fixture-only")
    return _source_closure_receipt(
        repo_root, require_tracked=True, entrypoints=values,
        schema="round0005_fixture_runtime_source_closure.v1")


def validate_source_closure_receipt(recorded: dict, *, repo_root: str) -> dict:
    entrypoints = recorded.get("entrypoints") if isinstance(recorded, dict) else None
    if tuple(entrypoints or ()) != ROUND0005_RUNTIME_ENTRYPOINTS:
        raise RuntimeError(
            "Round 0005 source closure entrypoints differ from the canonical "
            "ROUND0005_RUNTIME_ENTRYPOINTS tuple")
    current = source_closure_receipt(repo_root, require_tracked=True)
    if current != recorded:
        raise RuntimeError(
            "Round 0005 tracked runtime import closure changed: "
            f"expected={recorded!r} observed={current!r}")
    return current


def validate_round0014_source_closure_receipt(
        recorded: dict, *, repo_root: str) -> dict:
    """Reopen the exact Round 0014 source closure without accepting variants."""
    if (not isinstance(recorded, dict) or
            recorded.get("schema") != "round0014-runtime-source-closure-v1" or
            tuple(recorded.get("entrypoints") or ()) !=
            ROUND0014_RUNTIME_ENTRYPOINTS):
        raise RuntimeError("Round 0014 source closure identity/entrypoints changed")
    current = round0014_source_closure_receipt(
        repo_root, require_tracked=True)
    if current != recorded:
        raise RuntimeError("Round 0014 tracked runtime import closure changed")
    return current


def validate_round0015_source_closure_receipt(
        recorded: dict, *, repo_root: str) -> dict:
    """Reopen the exact Round 0015 source closure without variants."""
    if (not isinstance(recorded, dict) or
            recorded.get("schema") != "round0015-runtime-source-closure-v1" or
            tuple(recorded.get("entrypoints") or ()) !=
            ROUND0015_RUNTIME_ENTRYPOINTS):
        raise RuntimeError("Round 0015 source closure identity/entrypoints changed")
    current = round0015_source_closure_receipt(
        repo_root, require_tracked=True)
    if current != recorded:
        raise RuntimeError("Round 0015 tracked runtime import closure changed")
    return current


def _validate_source_closure_receipt_fixture_only(
        recorded: dict, *, repo_root: str, entrypoints: Iterable[str]) -> dict:
    """Private validator for the clean-release six-node CPU integration fixture."""
    expected_entrypoints = tuple(entrypoints)
    if (not isinstance(recorded, dict) or
            recorded.get("schema") != "round0005_fixture_runtime_source_closure.v1" or
            tuple(recorded.get("entrypoints") or ()) != expected_entrypoints):
        raise RuntimeError("fixture source closure identity/entrypoints are invalid")
    current = _source_closure_receipt_fixture_only(
        repo_root, entrypoints=expected_entrypoints)
    if current != recorded:
        raise RuntimeError("fixture tracked runtime import closure changed")
    return current
