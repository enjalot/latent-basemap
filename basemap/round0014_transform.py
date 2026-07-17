"""Authenticated production transform for the single Round 0014 treatment."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes)
from .minilm_input_pack import (TRANSFORM_EXECUTION_SPEC_SEAL, PackError,
                                build_transform_execution_spec,
                                stream_transform_to_npy_chunks)
from .round0014_program import (ISSUED_BASE, TRAIN_CONFIG, TRAIN_CONFIG_SHA256,
                                raw_source_map)


_ACTIVE_MODEL = None
_ACTIVE_CONFIG: dict[str, Any] | None = None


def production_transform(block: np.ndarray) -> np.ndarray:
    """Transform one bounded raw-fp32 block using the authenticated model."""
    if _ACTIVE_MODEL is None or _ACTIVE_CONFIG is None:
        raise RuntimeError("Round 0014 production transform model is not authenticated")
    array = np.asarray(block)
    if array.ndim != 2 or array.shape[1] != 384 or array.dtype.str != "<f4":
        raise ValueError("Round 0014 production transform requires bounded <f4 (N,384)")
    output = _ACTIVE_MODEL.transform(
        array, batch_size=_ACTIVE_CONFIG["model_batch_rows"])
    result = np.asarray(output, dtype="<f4")
    if result.shape != (len(array), 2) or not np.isfinite(result).all():
        raise RuntimeError("Round 0014 production transform emitted invalid coordinates")
    return result


def _template_config(*, release_sha: str,
                     train_output_relative_path: str) -> dict[str, Any]:
    if not isinstance(train_output_relative_path, str) or \
            train_output_relative_path.startswith("/") or ".." in \
            train_output_relative_path.split("/"):
        raise ValueError("trained-model output binding must be release/queue relative")
    transform = TRAIN_CONFIG["transform"]
    return {
        "schema": "round0014-transform-config-template-v1",
        "release_sha": release_sha,
        "production_config_sha256": TRAIN_CONFIG_SHA256,
        "trained_model": {
            "producer_node": "train_seed42_30m",
            "controller_output_relative_path": train_output_relative_path,
            "content_binding": "runtime-full-sha256-before-transform-spec-seal",
            "pre_gate_hash_available": False,
        },
        "architecture": TRAIN_CONFIG["model"],
        "model_weight_dtype": transform["model_weight_dtype"],
        "input_dtype": transform["input_dtype"],
        "output_dtype": transform["output_dtype"],
        "output_dimension": transform["output_dimension"],
        "model_batch_rows": transform["model_batch_rows"],
        "read_block_rows": transform["read_block_rows"],
        "rows_per_output_chunk": transform["rows_per_output_chunk"],
        "normalization": transform["normalization"],
        "centering": transform["centering"],
        "stochastic_options": transform["stochastic_options"],
    }


def build_transform_template(*, release_root: str, release_sha: str,
                             train_output_relative_path: str) -> dict[str, Any]:
    if release_sha == ISSUED_BASE:
        raise ValueError("Round 0014 transform template requires the implementation release")
    config = _template_config(
        release_sha=release_sha,
        train_output_relative_path=train_output_relative_path)
    spec = build_transform_execution_spec(
        production_transform, release_root=release_root,
        release_commit=release_sha, transform_config=config)
    body = {
        "schema": "round0014-transform-spec-template-v1",
        "release_sha": release_sha,
        "template_config": config,
        "template_config_sha256": sha256_bytes(canonical_json(config)),
        "execution_spec": spec,
        "execution_spec_sha256": spec[TRANSFORM_EXECUTION_SPEC_SEAL],
        "runtime_resolution": {
            "model_hash_source": "controller-certified train output",
            "resolution_order": [
                "reopen controller-certified model output",
                "hash exact model bytes",
                "validate architecture/dtype/dimensions",
                "seal actual transform execution spec",
                "authenticate callable/source/code/config/release",
                "resolve coordinate destination",
            ],
        },
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def validate_transform_template(path: str, *, release_root: str,
                                release_sha: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    identity = value.get("identity_sha256") if isinstance(value, dict) else None
    body = {key: value[key] for key in value if key != "identity_sha256"} \
        if isinstance(value, dict) else {}
    if identity != sha256_bytes(canonical_json(body)):
        raise ValueError("Round 0014 transform template seal changed")
    expected = build_transform_template(
        release_root=release_root, release_sha=release_sha,
        train_output_relative_path="artifacts/train/model.pt")
    if value != expected:
        raise ValueError("Round 0014 transform callable/source/config/release changed")
    return value


def _actual_config(*, template: dict[str, Any], model_signature: dict[str, Any]) \
        -> dict[str, Any]:
    config = dict(template["template_config"])
    config["schema"] = "round0014-transform-config-v1"
    config["trained_model"] = {
        "producer_node": "train_seed42_30m",
        "artifact": model_signature,
        "sha256": model_signature["sha256"],
        "content_binding": "runtime-full-sha256-before-transform-spec-seal",
    }
    return config


def authenticate_model_and_build_spec(*, model_path: str, template_path: str,
                                      release_root: str,
                                      release_sha: str) -> dict[str, Any]:
    """Reopen the sole trained model and bind it into the actual spec."""
    global _ACTIVE_MODEL, _ACTIVE_CONFIG
    if _ACTIVE_MODEL is not None or _ACTIVE_CONFIG is not None:
        raise RuntimeError("Round 0014 production transform is one-use per process")
    template = validate_transform_template(
        template_path, release_root=release_root, release_sha=release_sha)
    model_signature = expected_input_signature(model_path)
    if model_signature["kind"] != "file":
        raise ValueError("Round 0014 trained model must be one regular file")
    from .pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
    expected_model = TRAIN_CONFIG["model"]
    observed = {
        "architecture": model.architecture,
        "input_dimension": model.input_dim,
        "hidden_dimension": model.hidden_dim,
        "hidden_layers": model.n_layers,
        "output_dimension": model.n_components,
        "use_batchnorm": model.use_batchnorm,
        "use_dropout": model.use_dropout,
        "low_dim_kernel": model.low_dim_kernel,
        "a": model.a,
        "b": model.b,
    }
    parameter_dtypes = sorted({str(parameter.dtype)
                               for parameter in model.model.parameters()})
    if (observed != expected_model or model.positive_target_mode != "binary" or
            parameter_dtypes != ["torch.float32"]):
        raise RuntimeError(
            "Round 0014 trained-model architecture/dtype changed: "
            f"architecture={observed!r} dtypes={parameter_dtypes!r}")
    config = _actual_config(template=template, model_signature=model_signature)
    spec = build_transform_execution_spec(
        production_transform, release_root=release_root,
        release_commit=release_sha, transform_config=config)
    _ACTIVE_MODEL = model
    _ACTIVE_CONFIG = config
    return {
        "schema": "round0014-actual-transform-spec-v1",
        "model_signature": model_signature,
        "transform_config": config,
        "transform_config_sha256": sha256_bytes(canonical_json(config)),
        "execution_spec": spec,
        "execution_spec_sha256": spec[TRANSFORM_EXECUTION_SPEC_SEAL],
    }


def stream_production_coordinates(*, model_path: str, template_path: str,
                                  release_root: str, release_sha: str,
                                  output_root: str) -> dict[str, Any]:
    """Authenticate first, then let the public pack API resolve the destination."""
    actual = authenticate_model_and_build_spec(
        model_path=model_path, template_path=template_path,
        release_root=release_root, release_sha=release_sha)
    config = actual["transform_config"]
    implementation = sha256_bytes(canonical_json({
        "release_sha": release_sha,
        "execution_spec_sha256": actual["execution_spec_sha256"],
        "model_sha256": actual["model_signature"]["sha256"],
    }))
    capability = stream_transform_to_npy_chunks(
        raw_source_map(), output_root, production_transform,
        transform_id="round0014-production-model-transform-v1",
        transform_implementation_sha256=implementation,
        transform_config=config,
        transform_execution_spec=actual["execution_spec"],
        release_root=release_root,
        output_dim=2, output_dtype="<f4",
        rows_per_chunk=config["rows_per_output_chunk"],
        read_block_rows=config["read_block_rows"])
    return {"actual_transform": actual, "stream_capability": capability}
