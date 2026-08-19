"""Pinned adapter for the upstream ParamRepulsor implementation.

This module does not reproduce ParamRepulsor.  It verifies and calls the
authors' ``parampacmap`` package at one immutable upstream commit.  The source
closure hashes below cover the estimator, loss, model, pair construction, and
dataloader files used by the run.  A VCS install from another commit, a local
editable checkout, or modified installed source is refused.

The comparison uses the upstream defaults.  The only estimator settings added
by this study are an explicit seed and verbose progress logging.  Both are
recorded in the recipe; neither changes an upstream algorithmic default.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes


ROUND_ID = "0270"
STUDY_ID = "minilm-2m-upstream-paramrepulsor-v1"
ROWS = 2_000_000
DIMENSION = 384
SEEDS: tuple[int, ...] = (42, 43, 44)
CANONICAL_SEED = 42

UPSTREAM_REPOSITORY = "https://github.com/hyhuang00/ParamRepulsor"
UPSTREAM_COMMIT = "be8df72b1ac9041be3aae3d99f16f0d392b492dc"
UPSTREAM_VERSION = "0.1.1rc0"
UPSTREAM_LICENSE = "Apache-2.0"
PYTHON_VERSION = (3, 10)

RECIPE_SCHEMA = "baseline-upstream-paramrepulsor-2m-recipe-v1"
CHECKPOINT_SCHEMA = "baseline-upstream-paramrepulsor-checkpoint-v1"
CAPABILITY_TEMPLATE = "minilm-mixed-2m-upstream-paramrepulsor-seed{seed}-v1"

# SHA-256 at UPSTREAM_COMMIT.  These files are the algorithmic import closure
# reached by ParamPaCMAP.fit/transform for the ANN baseline.
UPSTREAM_SOURCE_CLOSURE: dict[str, str] = {
    "__init__.py": "0af750b32df1280c584f4984f5c7d2676f3e5a4f07dd4da05ca9ff2e2c3f8aee",
    "parampacmap.py": "aa6ce66d32a7cbcbc2fc96f0e45244769ca492ed3eab4f47ba9a211bccfaea15",
    "training.py": "4d610de904c9169c21b4298422e29716d2144ec3240297e0a0e1f82efa2323df",
    "models/__init__.py": "47e4ddadbd6ce67d6051eab3b9fc67e825072858634b7a9c39317cb52820e29c",
    "models/dataset.py": "5dda156efb7533d3046b3d8b3d89e3f4994e3114e30995624754d587bfebbd61",
    "models/module.py": "2740ae1f04bd6cecbfad6f38904887a695217e05fb2ab46eb402b14ede0e41e0",
    "utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "utils/data.py": "462c06b331e9b841c7b16dac63c5e1a1c7b25b720c169bb1e7f3f06fb0fb9cf1",
    "utils/utils.py": "3210ae73cab015cd0e017ac6d3664070be680e775d8ac49a51c16b95b3c28960",
}

# The authors' published CUDA 12.4 lock at UPSTREAM_COMMIT.  The local lock
# adds tqdm only because importing latent-basemap's shared evaluator reaches
# its ParametricUMAP module; tqdm does not enter the upstream training path.
EXPECTED_ENVIRONMENT = {
    "annoy": "1.17.3",
    "joblib": "1.4.2",
    "llvmlite": "0.43.0",
    "numba": "0.60.0",
    "numpy": "2.0.2",
    "parampacmap": UPSTREAM_VERSION,
    "scikit-learn": "1.5.2",
    "scipy": "1.14.1",
    "torch": "2.5.1+cu124",
    "tqdm": "4.67.1",
}

UPSTREAM_DEFAULTS: dict[str, Any] = {
    "n_components": 2,
    "n_neighbors": 10,
    "n_FP": 20,
    "n_MN": 5,
    "distance": "euclidean",
    "optim_type": "Adam",
    "lr": 1e-3,
    "lr_schedule": None,
    "apply_pca": True,
    "apply_scale": None,
    "model_dict": {"backbone": "ANN", "layer_size": [100, 100, 100]},
    "intermediate_snapshots": [],
    "loss_weight": [1, 1, 1],
    "batch_size": 1024,
    "data_reshape": None,
    "num_epochs": 450,
    "weight_schedule": "paramrep_weight_schedule",
    "const_schedule": "paramrep_const_schedule",
    "num_workers": 1,
    "dtype": "torch.float32",
    "embedding_init": "pca",
    "save_pairs": False,
}


class ParamRepulsorBaselineError(RuntimeError):
    """The upstream implementation, environment, or recipe is not pinned."""


def capability_for_seed(seed: int) -> str:
    seed = int(seed)
    if seed not in SEEDS:
        raise ParamRepulsorBaselineError(
            f"ParamRepulsor seed {seed!r} is not in the registered family {SEEDS}"
        )
    return CAPABILITY_TEMPLATE.format(seed=seed)


def recipe(seed: int = CANONICAL_SEED) -> dict[str, Any]:
    seed = int(seed)
    return {
        "schema": RECIPE_SCHEMA,
        "round_id": ROUND_ID,
        "study_id": STUDY_ID,
        "capability": capability_for_seed(seed),
        "seed": seed,
        "rows": ROWS,
        "dimension": DIMENSION,
        "implementation": {
            "kind": "unmodified_upstream_package",
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "package": "parampacmap",
            "version": UPSTREAM_VERSION,
            "license": UPSTREAM_LICENSE,
        },
        "estimator": copy.deepcopy(UPSTREAM_DEFAULTS),
        "study_overrides": {
            "seed": seed,
            "verbose": True,
            "reason": "reproducibility and progress logging; no algorithmic default changed",
        },
        "input": {
            "substrate": "sealed R0216 MiniLM mixed 2M ordered substrate",
            "row_order_preserved": True,
            "normalization": "already L2-normalized fp32 MiniLM embeddings",
            "additional_preprocessing": "upstream default PCA to 100 dimensions",
        },
        "evaluation": "R0265 instruments on the same R0216 substrate and held-out reserve",
        "gate_registerable_here": False,
    }


def assert_registered_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        seed = int(value["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParamRepulsorBaselineError("ParamRepulsor recipe has no integer seed") from exc
    expected = recipe(seed)
    if canonical_json(value) != canonical_json(expected):
        raise ParamRepulsorBaselineError(
            "ParamRepulsor recipe differs from the pinned upstream-default recipe"
        )
    return expected


def seed_invariant_sha256(value: Mapping[str, Any]) -> str:
    checked = copy.deepcopy(assert_registered_recipe(value))
    checked["seed"] = None
    checked["capability"] = CAPABILITY_TEMPLATE.format(seed="{seed}")
    checked["study_overrides"]["seed"] = None
    return sha256_bytes(canonical_json(checked))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream_environment() -> dict[str, Any]:
    """Verify the Python lock, VCS origin, and installed source closure."""
    if sys.version_info[:2] != PYTHON_VERSION:
        raise ParamRepulsorBaselineError(
            f"ParamRepulsor environment requires Python {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}, "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )
    observed_versions: dict[str, str] = {}
    for distribution, expected in EXPECTED_ENVIRONMENT.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ParamRepulsorBaselineError(
                f"required ParamRepulsor distribution {distribution!r} is absent"
            ) from exc
        observed_versions[distribution] = observed
        if observed != expected:
            raise ParamRepulsorBaselineError(
                f"{distribution}=={observed}, expected pinned {expected}"
            )

    distribution = importlib.metadata.distribution("parampacmap")
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise ParamRepulsorBaselineError(
            "parampacmap has no PEP 610 direct_url.json; install the pinned VCS requirement"
        )
    try:
        direct_url = json.loads(direct_url_text)
    except ValueError as exc:
        raise ParamRepulsorBaselineError("parampacmap direct_url.json is invalid") from exc
    vcs = dict(direct_url.get("vcs_info") or {})
    if vcs.get("vcs") != "git" or vcs.get("commit_id") != UPSTREAM_COMMIT:
        raise ParamRepulsorBaselineError(
            "parampacmap is not installed from the pinned upstream git commit"
        )
    source_url = str(direct_url.get("url") or "").removesuffix(".git")
    if source_url != UPSTREAM_REPOSITORY:
        raise ParamRepulsorBaselineError(
            f"parampacmap source URL {source_url!r} is not {UPSTREAM_REPOSITORY!r}"
        )

    import parampacmap

    package_root = Path(parampacmap.__file__).resolve().parent
    observed_files: dict[str, Any] = {}
    for relative, expected_sha in UPSTREAM_SOURCE_CLOSURE.items():
        path = package_root / relative
        if not path.is_file():
            raise ParamRepulsorBaselineError(f"upstream source file is absent: {path}")
        observed_sha = _sha256_file(path)
        if observed_sha != expected_sha:
            raise ParamRepulsorBaselineError(
                f"upstream source drift in {relative}: {observed_sha} != {expected_sha}"
            )
        observed_files[relative] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": observed_sha,
        }
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "versions": observed_versions,
        "direct_url": direct_url,
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "version": UPSTREAM_VERSION,
        "source_closure": observed_files,
        "every_source_digest_matches": True,
    }


def new_reducer(value: Mapping[str, Any]):
    """Construct the authors' estimator with every upstream default explicit."""
    checked = assert_registered_recipe(value)
    import torch
    from parampacmap import ParamPaCMAP
    from parampacmap.parampacmap import (
        paramrep_const_schedule,
        paramrep_weight_schedule,
    )

    params = checked["estimator"]
    reducer = ParamPaCMAP(
        n_components=params["n_components"],
        n_neighbors=params["n_neighbors"],
        n_FP=params["n_FP"],
        n_MN=params["n_MN"],
        distance=params["distance"],
        optim_type=params["optim_type"],
        lr=params["lr"],
        lr_schedule=params["lr_schedule"],
        apply_pca=params["apply_pca"],
        apply_scale=params["apply_scale"],
        model_dict=copy.deepcopy(params["model_dict"]),
        intermediate_snapshots=list(params["intermediate_snapshots"]),
        loss_weight=list(params["loss_weight"]),
        batch_size=params["batch_size"],
        data_reshape=params["data_reshape"],
        num_epochs=params["num_epochs"],
        verbose=bool(checked["study_overrides"]["verbose"]),
        weight_schedule=paramrep_weight_schedule,
        const_schedule=paramrep_const_schedule,
        num_workers=params["num_workers"],
        dtype=torch.float32,
        embedding_init=params["embedding_init"],
        seed=checked["seed"],
        save_pairs=params["save_pairs"],
    )
    return reducer


def save_checkpoint(
    reducer: Any,
    path: str,
    *,
    recipe_value: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> None:
    """Write a trusted, content-bound upstream reducer checkpoint.

    The whole upstream estimator is needed because the default PCA projector is
    part of out-of-sample transform.  Model and loss tensors are moved to CPU
    for portability, then restored so the caller can continue using the model.
    """
    import torch
    from basemap.output_safety import atomic_build_new_file

    checked = assert_registered_recipe(recipe_value)
    original_device = reducer.device
    _move_reducer_tensors(reducer, torch.device("cpu"))
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "recipe": checked,
        "seed_invariant_sha256": seed_invariant_sha256(checked),
        "upstream_environment": dict(environment),
        "reducer": reducer,
        "trusted_pickle_warning": (
            "torch.load uses pickle; load only this content-bound experiment artifact"
        ),
    }
    try:
        atomic_build_new_file(
            path,
            lambda temporary: torch.save(payload, temporary),
            immutable=True,
        )
    finally:
        _move_reducer_tensors(reducer, original_device)


def _move_reducer_tensors(reducer: Any, device: Any) -> None:
    """Move upstream state, including its three unregistered loss constants."""
    import torch

    target = torch.device(device)
    reducer.model.to(target)
    reducer.loss.to(target)
    for name in ("nnloss", "fploss", "mnloss"):
        component = getattr(reducer.loss, name)
        component.const = component.const.to(target)
    reducer.device = target


def load_checkpoint(path: str, *, device: str = "cuda") -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ParamRepulsorBaselineError("not a pinned ParamRepulsor baseline checkpoint")
    checked = assert_registered_recipe(payload.get("recipe") or {})
    if payload.get("seed_invariant_sha256") != seed_invariant_sha256(checked):
        raise ParamRepulsorBaselineError("ParamRepulsor checkpoint recipe digest changed")
    environment = dict(payload.get("upstream_environment") or {})
    if environment.get("commit") != UPSTREAM_COMMIT:
        raise ParamRepulsorBaselineError("ParamRepulsor checkpoint came from another commit")
    reducer = payload.get("reducer")
    if reducer is None or reducer.__class__.__module__ != "parampacmap.parampacmap":
        raise ParamRepulsorBaselineError("ParamRepulsor checkpoint has another estimator type")
    target = torch.device(device)
    _move_reducer_tensors(reducer, target)
    return reducer, checked


__all__ = [
    "CANONICAL_SEED",
    "CHECKPOINT_SCHEMA",
    "DIMENSION",
    "EXPECTED_ENVIRONMENT",
    "PYTHON_VERSION",
    "RECIPE_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "SEEDS",
    "STUDY_ID",
    "UPSTREAM_COMMIT",
    "UPSTREAM_DEFAULTS",
    "UPSTREAM_LICENSE",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_SOURCE_CLOSURE",
    "UPSTREAM_VERSION",
    "ParamRepulsorBaselineError",
    "assert_registered_recipe",
    "capability_for_seed",
    "load_checkpoint",
    "new_reducer",
    "recipe",
    "save_checkpoint",
    "seed_invariant_sha256",
    "verify_upstream_environment",
]
