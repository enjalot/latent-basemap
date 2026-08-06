"""Execute the R0203 h2048 quarter/half matched-low-dose ladder."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0187_composition_nested_ladder import RUNG_ROWS
from basemap.round0188_composition_boundary_seed43 import train_checks_close
from basemap.round0203_h2048_nested_dose_ladder import (
    CAPABILITY,
    EVALUATION_SCHEMA,
    HIDDEN_DIMENSION,
    HOST_RSS_LIMIT_GIB,
    ROUND_ID,
    RUNGS,
    SEED,
    SYNTHESIS_SCHEMA,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    Round0203Error,
    ladder_summary,
    successful_updates_for_edges,
    train_config,
    train_schema,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0202_nodes as delegate


ALLOWED_ACTIONS = {
    "train_h2048_low_dose_rung",
    "evaluate_h2048_low_dose_rung",
    "synthesize_h2048_low_dose_ladder",
}


def _signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:
        raise Round0203Error(f"{label} is unavailable or changed") from error


def _read(path: str, *, label: str) -> dict[str, Any]:
    try:
        return prompt_contract.read_sealed(path, label=label)
    except Exception as error:
        raise Round0203Error(f"{label} is missing or unsealed") from error


def _configure_delegate() -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": SEED,
        "RUNGS": RUNGS,
        "HIDDEN_DIMENSION": HIDDEN_DIMENSION,
        "HOST_RSS_LIMIT_GIB": HOST_RSS_LIMIT_GIB,
        "EVALUATION_SCHEMA": EVALUATION_SCHEMA,
        "SYNTHESIS_SCHEMA": SYNTHESIS_SCHEMA,
        "TARGET_POSITIVE_DRAWS_PER_EDGE": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "Round0202Error": Round0203Error,
        "successful_updates_for_edges": successful_updates_for_edges,
        "train_config": train_config,
        "train_schema": train_schema,
    }
    for name, value in bindings.items():
        setattr(delegate, name, value)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_delegate()
    delegate.run_train(active, job)


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_delegate()
    delegate.run_evaluate(active, job)


def run_synthesize(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0203Error("synthesis handler received another queue")
    cells: dict[str, dict[str, float]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    economics: dict[str, dict[str, Any]] = {}
    for rung in RUNGS:
        evaluation_path = os.path.join(
            str(job["evaluation_outputs"][rung]), "common-core-evaluation.json"
        )
        evaluation = _read(evaluation_path, label=f"R0203 {rung} evaluation")
        if (
            evaluation.get("schema") != EVALUATION_SCHEMA
            or evaluation.get("round_id") != ROUND_ID
            or evaluation.get("rung") != rung
            or int(evaluation.get("seed", -1)) != SEED
            or not all((evaluation.get("execution_checks") or {}).values())
        ):
            raise Round0203Error(f"R0203 {rung} evaluation contract changed")
        cells[rung] = {
            key: float(value)
            for key, value in (evaluation.get("primary_metrics") or {}).items()
        }
        evaluations[rung] = _signature(
            evaluation_path, label=f"R0203 {rung} evaluation"
        )
        train_path = os.path.join(
            str(job["train_outputs"][rung]), "train-receipt.json"
        )
        train = _read(train_path, label=f"R0203 {rung} train")
        graph = _read(
            str(job["graph_manifests"][rung]), label=f"accepted {rung} graph"
        )
        economics[rung] = {
            "retained_rows": RUNG_ROWS[rung],
            "directed_edges": int(graph["directed_edge_count"]),
            "successful_updates": int(train["optimizer_updates"]),
            "achieved_positive_draws_per_edge": float(
                train["consumed_positive_draws_per_edge"]
            ),
            "updates_per_s": float(train["steady_updates_per_s"]),
            "train_wall_s": float(train["train_wall_s"]),
            "train_receipt": _signature(train_path, label=f"R0203 {rung} train"),
        }

    full_evaluation_path = str(job["r0184_full_evaluation"])
    full_evaluation = _read(
        full_evaluation_path, label="accepted R0184 full evaluation from R0191"
    )
    if (
        full_evaluation.get("schema")
        != "round0191-r0184-h2048-common-core-evaluation-v1"
        or full_evaluation.get("round_id") != "0191"
        or full_evaluation.get("rung") != "full"
        or int(full_evaluation.get("seed", -1)) != SEED
        or not all((full_evaluation.get("execution_checks") or {}).values())
    ):
        raise Round0203Error("accepted R0184 full evaluation changed")
    cells["full"] = {
        key: float(value)
        for key, value in (full_evaluation.get("primary_metrics") or {}).items()
    }
    evaluations["full"] = _signature(
        full_evaluation_path, label="accepted R0184 full evaluation"
    )
    full_train_path = str(job["r0184_full_train"])
    full_train = _read(full_train_path, label="accepted R0184 full train")
    accounting = full_train.get("train_accounting") or {}
    if (
        full_train.get("schema")
        != "round0184-prompted-8m-dose-midpoint-train-receipt-v1"
        or full_train.get("round_id") != "0184"
        or int(full_train.get("optimizer_updates", -1)) != 1_000_000
        or int(accounting.get("positive_lr_optimizer_steps", -1)) != 1_000_000
        or not train_checks_close(full_train.get("train_checks"))
    ):
        raise Round0203Error("accepted R0184 full train changed")
    economics["full"] = {
        "retained_rows": 7_952_419,
        "directed_edges": 603_086_368,
        "successful_updates": 1_000_000,
        "achieved_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "updates_per_s": float(full_train["steady_updates_per_s"]),
        "train_wall_s": float(full_train["train_wall_s"]),
        "train_receipt": _signature(full_train_path, label="accepted R0184 full train"),
    }
    summary = ladder_summary(cells)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0203 h2048 low-dose ladder synthesis"
    )
    receipt = prompt_contract.seal({
        "schema": SYNTHESIS_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "summary": summary,
        "evaluations": evaluations,
        "training_economics": economics,
        "scientific_scope": {
            "width": HIDDEN_DIMENSION,
            "seed": SEED,
            "rungs": ["quarter", "half", "full"],
            "quarter_half_trained_here": True,
            "full_endpoint": "byte-exact accepted R0184; evaluation from R0191",
            "population_graph_composition_common_core_frozen": True,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "primary_registered_metric": "pile_ffr",
            "density_role": "diagnostic-only",
            "cross_width_selector": "deferred to campaign Track A3",
        },
        "training_performed_in_synthesis_node": False,
    })
    atomic_write_new_json(
        os.path.join(output, "h2048-low-dose-ladder-summary.json"),
        receipt,
        immutable=True,
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise Round0203Error(f"R0203 does not authorize action {action!r}")
    actions = {
        "train_h2048_low_dose_rung": run_train,
        "evaluate_h2048_low_dose_rung": run_evaluate,
        "synthesize_h2048_low_dose_ladder": run_synthesize,
    }
    actions[action](active, job)


__all__ = ["run_job"]
