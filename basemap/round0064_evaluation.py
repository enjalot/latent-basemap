"""Shared, content-bound evaluation primitives for the matched 30M/60M rung.

Round 0064 evaluates two independently trained checkpoints.  This module keeps
the two important identities structural:

* a checkpoint is usable only with its sealed successful train receipt; and
* scientific rows are the retained representatives of the substrate being
  scored, while the coordinate stream remains addressable by every substrate
  row.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .int8_eligibility import load_int8_eligibility
from .round0036_pipeline import (
    DIMENSION,
    EncodedInt8Array,
    Round0036Error,
)
from .round0040_program import (
    RepresentativeArrayView,
    RepresentativeRowSelector,
)


ROUND_ID = "0064"
MODEL_SPECS = {
    "r0061-30m": {
        "round_id": "0061",
        "receipt_schema": "round0061-train-receipt-v1",
        "config_schema": "round0055-production-config-v1",
        "rows": 30_000_000,
        "retained_rows": 29_781_754,
        "updates": 500_003,
        "sampler_class": "HostInt8Balanced30mCanonicalSampler",
    },
    "r0063-60m": {
        "round_id": "0063",
        "receipt_schema": "round0063-train-receipt-v1",
        "config_schema": "round0052-production-config-v1",
        "rows": 60_000_000,
        "retained_rows": 59_399_288,
        "updates": 997_248,
        "sampler_class": "HostInt8BalancedCanonicalSampler",
    },
    "r0068-45m": {
        "round_id": "0068",
        "receipt_schema": "round0068-train-receipt-v1",
        "config_schema": "round0068-production-config-v1",
        "rows": 45_000_000,
        "retained_rows": 44_598_360,
        "updates": 748_757,
        "sampler_class": "HostInt8SelectedCanonicalSampler",
    },
    "r0075-90m": {
        "round_id": "0075",
        "receipt_schema": "round0075-train-receipt-v1",
        "config_schema": "round0075-production-config-v1",
        "rows": 90_000_000,
        "retained_rows": 88_945_313,
        "updates": 1_493_293,
        "sampler_class": "HostInt8Balanced90mCanonicalSampler",
    },
    "r0079-120m": {
        "round_id": "0079",
        "receipt_schema": "round0079-train-receipt-v1",
        "config_schema": "round0079-production-config-v1",
        "rows": 120_000_000,
        "retained_rows": 118_067_492,
        "updates": 1_982_221,
        "sampler_class": "HostInt8Balanced120mCanonicalSampler",
    },
}


class Round0064Error(Round0036Error):
    """The matched-scale evaluation contract was violated."""


def expected_retained_rows_for_scale(row_count: int) -> int:
    """Return the one registered representative count for a scale universe."""
    matches = {
        int(spec["retained_rows"])
        for spec in MODEL_SPECS.values()
        if int(spec["rows"]) == int(row_count)
    }
    if len(matches) != 1:
        raise Round0064Error(
            f"scale substrate row count {row_count} is not registered exactly"
        )
    return next(iter(matches))


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {
        **value,
        "identity_sha256": sha256_bytes(canonical_json(value)),
    }


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {
        key: item
        for key, item in value.items()
        if key != "identity_sha256"
    }
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0064Error(f"{label} identity seal is invalid")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_train_bundle(
    *,
    label: str,
    model_path: str,
    model_sha256: str,
    train_receipt_path: str,
    train_receipt_sha256: str,
) -> dict[str, Any]:
    """Authenticate one exact registered scale model-and-receipt tuple."""
    try:
        spec = MODEL_SPECS[label]
    except KeyError as exc:
        raise Round0064Error(f"unknown scale model label {label!r}") from exc
    if not _valid_sha256(model_sha256) or not _valid_sha256(
        train_receipt_sha256
    ):
        raise Round0064Error("model/train receipt SHA-256 is malformed")
    model = expected_input_signature(model_path)
    receipt_signature = expected_input_signature(train_receipt_path)
    if model["sha256"] != model_sha256:
        raise Round0064Error(f"{label} model bytes changed")
    if receipt_signature["sha256"] != train_receipt_sha256:
        raise Round0064Error(f"{label} train receipt bytes changed")
    with open(train_receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label=f"{label} train receipt")
    config = receipt.get("production_config")
    config_sha256 = receipt.get("production_config_sha256")
    accounting = receipt.get("train_accounting") or {}
    runtime = receipt.get("exact_execution_receipt") or {}
    row_universe = (config or {}).get("row_universe") or {}
    model_config = (config or {}).get("model") or {}
    expected_stamp = (config or {}).get("execution", {}).get(
        "expected_pipeline_stamp"
    ) or {}
    updates = int(spec["updates"])
    stamp_fields = (
        "pipeline",
        "sampler_class",
        "x_residency",
        "positive_sampling",
        "negative_sampling",
        "positive_source_count",
        "valid_canonical_edge_count",
    )
    if (
        receipt.get("schema") != spec["receipt_schema"]
        or receipt.get("round_id") != spec["round_id"]
        or receipt.get("model") != model
        or not isinstance(config, dict)
        or config.get("schema") != spec["config_schema"]
        or config_sha256 != sha256_bytes(canonical_json(config))
        or row_universe.get("rows") != spec["rows"]
        or row_universe.get("input_dimension") != DIMENSION
        or model_config.get("architecture") != "residual_bottleneck"
        or model_config.get("input_dimension") != DIMENSION
        or model_config.get("hidden_dimension") != 2048
        or model_config.get("output_dimension") != 2
        or accounting.get("budget_satisfied") is not True
        or accounting.get("positive_lr_optimizer_steps") != updates
        or accounting.get("optimizer_steps_attempted") != updates
        or accounting.get("optimizer_steps_succeeded") != updates
        or accounting.get("amp_overflow_skips") != 0
        or accounting.get("nonfinite_loss_skips") != 0
        or accounting.get("nonfinite_gradient_skips") != 0
        or accounting.get("stop_reason") != "lr_horizon"
        or runtime.get("sampler_class") != spec["sampler_class"]
        or any(runtime.get(key) != expected_stamp.get(key)
               for key in stamp_fields)
        or receipt.get("retry_count") != 0
    ):
        raise Round0064Error(
            f"{label} train/model execution contract is incomplete"
        )
    return {
        "label": label,
        "spec": dict(spec),
        "model": model,
        "train_receipt": receipt_signature,
        "receipt": receipt,
        "production_config": config,
        "production_config_sha256": config_sha256,
    }


def load_train_model(
    bundle: Mapping[str, Any],
    *,
    device: str = "cuda",
) -> Any:
    """Load the exact finite state dict with the receipt-bound architecture."""
    import torch
    from experiments.run_round0034_node import _exact_model

    checkpoint = torch.load(
        bundle["model"]["canonical_path"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "state_dict",
        "production_config",
        "production_config_sha256",
    }:
        raise Round0064Error("scale checkpoint fields changed")
    if (
        checkpoint["production_config"] != bundle["production_config"]
        or checkpoint["production_config_sha256"]
        != bundle["production_config_sha256"]
    ):
        raise Round0064Error("checkpoint and train receipt configs differ")
    state = checkpoint["state_dict"]
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(name, str)
            or not torch.is_tensor(value)
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
            for name, value in state.items()
        )
    ):
        raise Round0064Error("checkpoint state is not finite fp32")
    instance = _exact_model(bundle["production_config"])
    instance.device = str(device)
    instance._init_model(input_dim=DIMENSION)
    expected = instance.model.state_dict()
    if set(state) != set(expected) or any(
        tuple(state[name].shape) != tuple(expected[name].shape)
        for name in state
    ):
        raise Round0064Error("checkpoint state keys/shapes changed")
    instance.model.load_state_dict(state, strict=True)
    instance.model.to(device)
    instance.model.eval()
    instance.is_fitted = True
    return instance


def load_substrate(
    *,
    int8_path: str,
    int8_sha256: str,
    scales_path: str,
    scales_sha256: str,
    eligibility_path: str,
    eligibility_sha256: str,
    row_count: int,
) -> tuple[
    EncodedInt8Array,
    RepresentativeRowSelector,
    RepresentativeArrayView,
    dict[str, Any],
]:
    """Load one exact int8 substrate and its representative-only view."""
    encoded = EncodedInt8Array.from_files(
        int8_path=int8_path,
        int8_sha256=int8_sha256,
        scales_path=scales_path,
        scales_sha256=scales_sha256,
        row_count=row_count,
        dimension=DIMENSION,
    )
    eligibility = load_int8_eligibility(
        eligibility_path,
        expected_sha256=eligibility_sha256,
        row_count=row_count,
    )
    selector = RepresentativeRowSelector(
        eligibility["excluded_rows"],
        row_count=row_count,
        source=eligibility["signature"],
        policy=(
            "exact within-subset zero/duplicate exclusion; first ordered "
            "family member is the retained representative"
        ),
    )
    expected_retained = expected_retained_rows_for_scale(row_count)
    if selector.retained_count != expected_retained:
        raise Round0064Error("scale substrate retained-row accounting changed")
    return (
        encoded,
        selector,
        RepresentativeArrayView(encoded, selector),
        eligibility,
    )


def retained_identity(
    encoded: EncodedInt8Array,
    selector: RepresentativeRowSelector,
    eligibility: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Complete scientific identity for a compact representative universe."""
    base = encoded.scientific_identity()
    signature = eligibility["signature"]
    return {
        "data_identity": {
            "kind": "ordered_shards",
            "shape": [selector.retained_count, DIMENSION],
            "dtype": "<f4",
            "shards": [
                *base["shards"],
                {
                    "position": len(base["shards"]),
                    "name": os.path.basename(signature["canonical_path"]),
                    "bytes": int(signature["bytes"]),
                    "sha256": signature["sha256"],
                },
            ],
        },
        "convention": {
            "row_order": f"compact ascending {label} retained row IDs",
            "selector": selector.identity(),
            "distance": "squared L2 on fp32 dequantized int8 rows",
            "self_exclusion": True,
            "anchor_namespace": "compact retained-row positions",
        },
    }
