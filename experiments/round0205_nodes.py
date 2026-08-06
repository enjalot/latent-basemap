"""Publish the reviewed FineWeb-2M v0 bundle to the local map registry."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_copy_new,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0205_v0_registry import (
    BUNDLE_SCHEMA,
    CANDIDATE_ID,
    EXPECTED_COORDINATES_SHA256,
    EXPECTED_TRAIN_RECEIPT_SHA256,
    INPUT_DIMENSION,
    MAP_DEFINITION_SCHEMA,
    OUTPUT_DIMENSION,
    PUBLICATION_SCHEMA,
    ROUND_ID,
    ROWS,
    Round0205Error,
    canonical_metrics,
    named_ood_failures,
)


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    try:
        return prompt_contract.read_sealed(path, label=label)
    except Exception as error:
        raise Round0205Error(f"{label} is missing, changed, or unsealed") from error


def _validate_coordinates(path: str) -> None:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, OUTPUT_DIMENSION) or array.dtype != np.dtype("<f4"):
        raise Round0205Error("canonical coordinates changed shape or dtype")
    for start in range(0, ROWS, 250_000):
        if not np.isfinite(array[start : start + 250_000]).all():
            raise Round0205Error("canonical coordinates contain non-finite values")


def _require_bundle(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    bundle = _read_sealed(path, label="accepted R0204 release bundle")
    canonical = bundle.get("canonical_artifact") or {}
    coordinates = canonical.get("coordinates") or {}
    train_receipt = canonical.get("train_receipt") or {}
    if (
        bundle.get("schema") != BUNDLE_SCHEMA
        or bundle.get("round_id") != "0204"
        or bundle.get("candidate_id") != CANDIDATE_ID
        or canonical.get("canonical_seed") != 42
        or canonical.get("rows") != ROWS
        or canonical.get("dimension") != INPUT_DIMENSION
        or canonical.get("embedding_convention") != "Document: "
        or coordinates.get("sha256") != EXPECTED_COORDINATES_SHA256
        or train_receipt.get("sha256") != EXPECTED_TRAIN_RECEIPT_SHA256
        or (bundle.get("release_actions") or {}).get(
            "local_registry_round_authorized_by_campaign"
        )
        is not True
        or (bundle.get("release_actions") or {}).get(
            "huggingface_upload_authorized"
        )
        is not False
    ):
        raise Round0205Error("accepted R0204 release contract changed")
    prompt_contract.verify_signature(coordinates, label="canonical coordinates")
    prompt_contract.verify_signature(train_receipt, label="canonical train receipt")
    prompt_contract.verify_signature(bundle.get("model_card"), label="model card")
    canonical_metrics(bundle)
    named_ood_failures(bundle)
    return bundle, signature


def _require_score(path: str, coordinates: Mapping[str, Any]) -> dict[str, Any]:
    score = _read_sealed(path, label="canonical R0115 score")
    if (
        score.get("schema") != "round0113-prompt-arm-score-v1"
        or score.get("round_id") != "0115"
        or score.get("arm") != "document"
        or (score.get("coordinates") or {}).get("training") != dict(coordinates)
        or (score.get("panel") or {}).get("n") != ROWS
        or (score.get("panel") or {}).get("n_dims_hi") != INPUT_DIMENSION
        or (score.get("panel") or {}).get("n_dims_lo") != OUTPUT_DIMENSION
        or not all((score.get("execution_gates") or {}).values())
    ):
        raise Round0205Error("canonical R0115 score contract changed")
    return score


def _require_train_receipt(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != EXPECTED_TRAIN_RECEIPT_SHA256:
        raise Round0205Error("canonical train receipt bytes changed")
    receipt = _read_sealed(path, label="canonical R0115 train receipt")
    accounting = receipt.get("train_accounting") or {}
    execution = receipt.get("exact_execution_receipt") or {}
    model = receipt.get("model") or {}
    if (
        receipt.get("schema") != "round0113-train-receipt-v1"
        or receipt.get("round_id") != "0115"
        or receipt.get("arm") != "document"
        or receipt.get("training_performed") is not True
        or receipt.get("optimizer_updates") != 500_000
        or accounting.get("optimizer_steps_succeeded") != 500_000
        or accounting.get("amp_overflow_skips") != 0
        or accounting.get("nonfinite_gradient_skips") != 0
        or accounting.get("nonfinite_loss_skips") != 0
        or execution.get("pipeline") != "host_weighted_jina_prompt_contrast"
        or execution.get("sampler_class") != "PromptWeightedJinaSampler"
        or execution.get("positive_sampling")
        != "fuzzy_weight_proportional_with_replacement_via_exact_uniform_envelope_rejection"
        or execution.get("feature_residency")
        != "host-contiguous-compact-fp16-memmap"
        or not isinstance(model.get("sha256"), str)
    ):
        raise Round0205Error("canonical R0115 execution receipt changed")
    prompt_contract.verify_signature(model, label="canonical seed-42 model")
    return receipt, signature


def _write_definition(
    *,
    active: Mapping[str, Any],
    output: str,
    bundle: Mapping[str, Any],
    bundle_signature: Mapping[str, Any],
    score: Mapping[str, Any],
    score_path: str,
    train_receipt: Mapping[str, Any],
    train_signature: Mapping[str, Any],
) -> str:
    coordinate_root = ensure_data_directory(os.path.join(output, "coordinates"))
    chunk_root = ensure_data_directory(os.path.join(coordinate_root, "chunk-00000"))
    source_coordinates = (bundle.get("canonical_artifact") or {})["coordinates"]
    copied_path = os.path.join(chunk_root, "coordinates.npy")
    atomic_copy_new(
        str(source_coordinates["canonical_path"]), copied_path, immutable=True
    )
    copied_signature = expected_input_signature(copied_path)
    if copied_signature["sha256"] != EXPECTED_COORDINATES_SHA256:
        raise Round0205Error("local coordinate copy is not byte-identical")
    _validate_coordinates(copied_path)

    card_source = str((bundle.get("model_card") or {})["canonical_path"])
    card_path = os.path.join(output, "README.md")
    atomic_copy_new(card_source, card_path, immutable=True)
    card_signature = expected_input_signature(card_path)
    if card_signature["sha256"] != (bundle.get("model_card") or {}).get("sha256"):
        raise Round0205Error("local model-card copy is not byte-identical")

    production_config_path = prompt_contract.verify_signature(
        train_receipt.get("production_config"), label="production config"
    )
    with open(production_config_path, encoding="utf-8") as handle:
        production_config = json.load(handle)
    if (
        production_config.get("schema") != "round0113-production-config-v1"
        or production_config.get("round_id") != "0115"
        or production_config.get("arm") != "document"
    ):
        raise Round0205Error("canonical production config changed")
    config = production_config.get("config") or {}
    model_config = config.get("model") or {}
    execution = train_receipt.get("exact_execution_receipt") or {}
    metrics = canonical_metrics(bundle)
    definition = prompt_contract.seal({
        "schema": MAP_DEFINITION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "map_id": CANDIDATE_ID,
        "candidate_id": CANDIDATE_ID,
        "training_round": "0115",
        "evaluation_round": "0115",
        "release_bundle_round": "0204",
        "release_bundle": dict(bundle_signature),
        "coordinates": copied_signature,
        "source_coordinates": dict(source_coordinates),
        "coordinate_layout": "single immutable chunk in registry row order",
        "model_card": card_signature,
        "train_receipt": dict(train_signature),
        "score": expected_input_signature(score_path),
        "model": dict(train_receipt["model"]),
        "population": {
            "corpus": "FineWeb English frozen R0113 population",
            "rows": ROWS,
            "input_dimension": INPUT_DIMENSION,
            "output_dimension": OUTPUT_DIMENSION,
            "embedding_model": "jinaai/jina-embeddings-v5-text-nano",
            "embedding_convention": "Document: ",
            "canonical_seed": 42,
        },
        "architecture": {
            "name": model_config.get("architecture"),
            "hidden_dimension": model_config.get("hidden_dimension"),
            "hidden_layers": model_config.get("hidden_layers"),
            "low_dim_kernel": model_config.get("low_dim_kernel"),
            "optimizer_updates": train_receipt.get("optimizer_updates"),
            "amp_dtype": (train_receipt.get("train_accounting") or {}).get(
                "amp_dtype"
            ),
        },
        "actual_pipeline": {
            "pipeline": execution.get("pipeline"),
            "sampler_class": execution.get("sampler_class"),
            "positive_sampling": execution.get("positive_sampling"),
            "feature_residency": execution.get("feature_residency"),
        },
        "metrics": {
            **metrics,
            "formula_version": (score.get("panel") or {}).get("formula_version"),
            "all_six_seed42_gates_pass": True,
            "all_four_seeds_pass_all_six_gates": True,
        },
        "limitations": {
            "canonical_seed42_named_ood_failures": named_ood_failures(bundle),
            "canonical_seed42_named_ood_failure_count": 7,
            "named_ood_probe_count": 11,
            "universal_ood_quality_claim": False,
            "method_winner_claim": False,
            "sae_readiness_claim": False,
        },
        "release_scope": {
            "local_registry_ready": True,
            "local_v0_release_registered": True,
            "intended_use": "exploratory FineWeb-English under Document: convention",
            "production_readiness_claimed": False,
            "huggingface_upload_authorized": False,
            "huggingface_upload_performed": False,
            "external_publication_performed": False,
        },
        "training_performed": False,
    })
    definition_path = os.path.join(output, "map-definition.json")
    atomic_write_new_json(definition_path, definition, immutable=True)
    return definition_path


def _publish_registry(
    *, definition_path: str, output: str, started: float
) -> dict[str, Any]:
    from experiments import map_registry

    registry = map_registry.scan()
    matches = [
        item for item in registry.get("maps", [])
        if item.get("map_id") == CANDIDATE_ID
    ]
    if len(matches) != 1 or matches[0].get("round_id") != ROUND_ID:
        raise Round0205Error("registry scan did not discover exactly one v0 map")
    entry = matches[0]
    if (
        entry.get("local_v0_release_registered") is not True
        or entry.get("production_ready") is not False
        or entry.get("universal_ood_ready") is not False
        or (entry.get("coordinates") or {}).get("receipt_sha256")
        != EXPECTED_COORDINATES_SHA256
        or (entry.get("panel") or {}).get("decision_checks_all_pass") is not True
    ):
        raise Round0205Error("discovered v0 registry entry changed scope or identity")

    snapshot = map_registry.write_registry(registry)
    if snapshot is None:
        raise Round0205Error("registry promotion did not mint a new immutable snapshot")
    snapshot_signature = expected_input_signature(str(snapshot))
    map_registry.publish(registry)
    page = map_registry.SITE_DIR / f"round-{ROUND_ID}" / "index.html"
    if not page.is_file() or CANDIDATE_ID not in page.read_text(encoding="utf-8"):
        raise Round0205Error("local map-registry page was not published")

    receipt = prompt_contract.seal({
        "schema": PUBLICATION_SCHEMA,
        "round_id": ROUND_ID,
        "map_id": CANDIDATE_ID,
        "map_definition": expected_input_signature(definition_path),
        "immutable_registry_snapshot": snapshot_signature,
        "mutable_registry_view_observed": expected_input_signature(
            str(map_registry.REGISTRY_PATH)
        ),
        "local_site_page_observed": expected_input_signature(str(page)),
        "site_url": f"{map_registry.SITE_URL}/round-{ROUND_ID}/",
        "registry_entry": entry,
        "checks": {
            "exact_map_id_discovered_once": True,
            "immutable_snapshot_minted": True,
            "local_site_page_published": True,
            "production_readiness_not_claimed": True,
            "universal_ood_readiness_not_claimed": True,
            "huggingface_upload_performed": False,
        },
        "training_performed": False,
        "wall_s": time.monotonic() - started,
    })
    receipt_path = os.path.join(output, "registry-publication.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return receipt


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if (
        active.get("manifest", {}).get("round_id") != ROUND_ID
        or job.get("action") != "register_v0_locally"
    ):
        raise Round0205Error("R0205 handler received another round or action")
    started = time.monotonic()
    bundle, bundle_signature = _require_bundle(str(job["r0204_bundle"]))
    coordinates = (bundle.get("canonical_artifact") or {})["coordinates"]
    score = _require_score(str(job["r0115_score"]), coordinates)
    train_path = str(coordinates.get("canonical_path", "")).replace(
        "/evaluation/coordinates.npy", "/train/train-receipt.json"
    )
    if train_path != str(job["r0115_train_receipt"]):
        raise Round0205Error("R0115 score/train path relation changed")
    train_receipt, train_signature = _require_train_receipt(train_path)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0205 local registry artifact"
    )
    definition_path = _write_definition(
        active=active,
        output=output,
        bundle=bundle,
        bundle_signature=bundle_signature,
        score=score,
        score_path=str(job["r0115_score"]),
        train_receipt=train_receipt,
        train_signature=train_signature,
    )
    _publish_registry(
        definition_path=definition_path, output=output, started=started
    )


__all__ = ["run_job"]
