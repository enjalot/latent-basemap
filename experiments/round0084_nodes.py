"""Train and evaluate one seed-43 balanced-90M sensitivity replicate."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0064_evaluation import (
    MODEL_SPECS,
    seal,
    validate_seal,
    validate_train_bundle,
)
from basemap.round0071_substrate import (
    ELIGIBILITY_SUMMARY,
    ROW_COUNT,
    validate_substrate,
)
from basemap.round0075_training import SUCCESSFUL_UPDATES
from basemap.round0084_program import (
    BASELINE_SEED,
    CONFIG_SCHEMA,
    FULL_KEY,
    MATCHED_KEY,
    MODEL_LABEL,
    PANEL_SCHEMA,
    ROUND_ID,
    SEED,
    TRAIN_RECEIPT_SCHEMA,
    Round0084ProgramError,
    seed43_config_from_seed42,
)
from experiments import round0064_nodes as evaluator
from experiments import round0075_nodes as trainer


MAP_LABELS = {
    MATCHED_KEY: "r0084-balanced-90m-seed43-on-matched-30m",
    FULL_KEY: "r0084-balanced-90m-seed43",
}


class Round0084Error(Round0084ProgramError):
    """The paired seed-sensitivity execution contract was violated."""


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0084Error(f"{path} is not a JSON object")
    return value


def _configure_evaluator() -> None:
    evaluator.ROUND_ID = ROUND_ID
    evaluator.MAP_LABELS = MAP_LABELS
    MODEL_SPECS[MODEL_LABEL] = {
        "round_id": ROUND_ID,
        # The exact R0075 trainer receipt format is intentionally reused;
        # round_id, config schema, seed, release, model, and seal disambiguate
        # the new execution.
        "receipt_schema": TRAIN_RECEIPT_SCHEMA,
        "config_schema": CONFIG_SCHEMA,
        "rows": ROW_COUNT,
        "retained_rows": ELIGIBILITY_SUMMARY["retained_row_count"],
        "updates": SUCCESSFUL_UPDATES,
        "sampler_class": "HostInt8Balanced90mCanonicalSampler",
    }


def _baseline_bundle(job: Mapping[str, Any]) -> dict[str, Any]:
    return validate_train_bundle(
        label="r0075-90m",
        model_path=str(job["baseline_model"]),
        model_sha256=str(job["baseline_model_sha256"]),
        train_receipt_path=str(job["baseline_train_receipt"]),
        train_receipt_sha256=str(job["baseline_train_receipt_sha256"]),
    )


def _config_builder(job: Mapping[str, Any]):
    baseline = _baseline_bundle(job)

    def build(**kwargs: Any) -> tuple[dict[str, Any], str]:
        return seed43_config_from_seed42(
            baseline["production_config"],
            **kwargs,
        )

    return build


def run_train(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    """Run the reviewed R0075 trainer while changing only its random seed."""
    substrate = validate_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    baseline = _baseline_bundle(job)
    graph_path = str(job["canonical_graph_manifest"])
    graph = _read_json(graph_path)
    graph_signature = expected_input_signature(graph_path)
    _config, config_sha256 = seed43_config_from_seed42(
        baseline["production_config"],
        graph_manifest=graph,
        graph_manifest_path=graph_signature["canonical_path"],
        graph_manifest_sha256=graph_signature["sha256"],
        substrate_manifest=substrate["manifest"],
        substrate_manifest_path=substrate["signature"]["canonical_path"],
        substrate_manifest_sha256=substrate["signature"]["sha256"],
    )
    if (
        job.get("train_config_sha256") != config_sha256
        or active["manifest"].get("release_sha")
        != job.get("release_sha")
    ):
        raise Round0084Error("R0084 queue/config identity changed")

    previous = {
        "round_id": trainer.ROUND_ID,
        "seed": trainer.SEED,
        "builder": trainer.train_config_from_capabilities,
    }
    trainer.ROUND_ID = ROUND_ID
    trainer.SEED = SEED
    trainer.train_config_from_capabilities = _config_builder(job)
    try:
        receipt = trainer.run_train(active, job)
    finally:
        trainer.ROUND_ID = previous["round_id"]
        trainer.SEED = previous["seed"]
        trainer.train_config_from_capabilities = previous["builder"]
    if (
        receipt.get("round_id") != ROUND_ID
        or receipt.get("seed") != SEED
        or (receipt.get("production_config") or {}).get("schema")
        != CONFIG_SCHEMA
        or (receipt.get("production_config") or {}).get(
            "optimizer", {}
        ).get("seed")
        != SEED
    ):
        raise Round0084Error("R0084 train receipt did not bind seed 43")
    return receipt


def _late_bind_model(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    model_path = str(job["model_path"])
    receipt_path = str(job["train_receipt_path"])
    model = expected_input_signature(model_path)
    receipt_signature = expected_input_signature(receipt_path)
    receipt = _read_json(receipt_path)
    body = {
        key: value for key, value in receipt.items()
        if key != "identity_sha256"
    }
    treatment = (
        (receipt.get("production_config") or {})
        .get("execution", {})
        .get("seed_sensitivity_treatment", {})
    )
    if (
        receipt.get("schema") != TRAIN_RECEIPT_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("release_sha") != active["manifest"]["release_sha"]
        or receipt.get("model") != model
        or receipt.get("seed") != SEED
        or treatment.get("baseline_seed") != BASELINE_SEED
        or treatment.get("treatment_seed") != SEED
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
    ):
        raise Round0084Error("late-bound R0084 model/receipt changed")
    return {
        **job,
        "model_sha256": model["sha256"],
        "train_receipt_sha256": receipt_signature["sha256"],
    }


def _metrics(panel: Mapping[str, Any]) -> dict[str, float]:
    scientific = panel["panel"]
    purity = scientific["purity"]
    return {
        "ffr": float(scientific["ffr"]),
        "density_legacy_diagnostic": float(scientific["density"]),
        "purity_k256": float(purity["k256"]),
        "purity_k1024": float(purity["k1024"]),
        "projection_ffr": float(panel["projection"]["proj_ffr"]),
        "recall_at_10": float(panel["recall_at_10"]),
        "recall_at_50": float(panel["recall_at_50"]),
    }


def _load_panel(path: str, *, schema: str, key: str) -> dict[str, Any]:
    panel = _read_json(path)
    validate_seal(panel, label=f"R0084 {key} panel")
    if panel.get("schema") != schema or panel.get("map_key") != key:
        raise Round0084Error(f"panel identity changed for {key}")
    return panel


def _same_universe(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("eligibility") == right.get("eligibility")
        and left.get("scientific_universe") == right.get("scientific_universe")
        and left.get("panel", {}).get("n") == right.get("panel", {}).get("n")
        and left.get("panel", {}).get("anchor_hash")
        == right.get("panel", {}).get("anchor_hash")
        and left.get("panel", {}).get("provenance", {}).get(
            "hiD_reference_key"
        )
        == right.get("panel", {}).get("provenance", {}).get(
            "hiD_reference_key"
        )
    )


def run_comparison(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0084 paired seed-sensitivity comparison",
    )
    panels = {
        "seed42_matched": _load_panel(
            str(job["seed42_matched_panel"]),
            schema="round0076-registered-panel-v1",
            key="r0075-90m-on-30m",
        ),
        "seed43_matched": _load_panel(
            str(job["seed43_matched_panel"]),
            schema=PANEL_SCHEMA,
            key=MATCHED_KEY,
        ),
        "seed42_full": _load_panel(
            str(job["seed42_full_panel"]),
            schema="round0076-registered-panel-v1",
            key="r0075-90m-on-90m",
        ),
        "seed43_full": _load_panel(
            str(job["seed43_full_panel"]),
            schema=PANEL_SCHEMA,
            key=FULL_KEY,
        ),
    }
    if (
        not _same_universe(
            panels["seed42_matched"], panels["seed43_matched"]
        )
        or not _same_universe(panels["seed42_full"], panels["seed43_full"])
    ):
        raise Round0084Error("seed panels do not share exact universes")

    rows: dict[str, Any] = {}
    for universe in ("matched", "full"):
        baseline = _metrics(panels[f"seed42_{universe}"])
        treatment = _metrics(panels[f"seed43_{universe}"])
        rows[universe] = {
            metric: {
                "seed42": baseline[metric],
                "seed43": treatment[metric],
                "signed_delta_seed43_minus_seed42": round(
                    treatment[metric] - baseline[metric], 6
                ),
                "absolute_delta": round(
                    abs(treatment[metric] - baseline[metric]), 6
                ),
            }
            for metric in baseline
        }
    full_checks = {
        seed: {
            key: value
            for key, value in panels[f"{seed}_full"][
                "decision_checks"
            ].items()
            if key != "density_at_least_0_60"
        }
        for seed in ("seed42", "seed43")
    }
    body = {
        "schema": "round0084-seed43-sensitivity-contrast-v1",
        "round_id": ROUND_ID,
        "baseline_seed": BASELINE_SEED,
        "treatment_seed": SEED,
        "paired_metric_contrasts": rows,
        "full_90m_non_density_checks": full_checks,
        "legacy_density_absolute_check_reported_not_gating": {
            seed: panels[f"{seed}_full"]["decision_checks"].get(
                "density_at_least_0_60"
            )
            for seed in ("seed42", "seed43")
        },
        "interpretation": {
            "one_paired_seed_contrast": True,
            "estimates_variance": False,
            "establishes_error_bar": False,
            "changes_ladder_decision": False,
            "future_statistical_noise_claim_requires_more_seeds": True,
        },
        "panels": {
            key: expected_input_signature(str(job[f"{key}_panel"]))
            for key in panels
        },
        "training_performed": True,
    }
    receipt = seal(body)
    path = os.path.join(output, "seed43-sensitivity-contrast.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise RuntimeError("R0084 handler requires its exact round and job")
    action = str(job.get("action"))
    if action == "train":
        return run_train(active, job)
    _configure_evaluator()
    if action == "transform":
        return evaluator.run_transform(active, _late_bind_model(active, job))
    if action == "panel":
        return evaluator.run_panel(active, _late_bind_model(active, job))
    if action == "comparison":
        return run_comparison(active, job)
    raise RuntimeError(f"unknown R0084 action {action!r}")
