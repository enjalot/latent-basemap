"""Execute the R0191 full-rung h4096 width-only contrast."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0184_prompted_8m_dose_midpoint import (
    scale_train_config as h2048_train_config,
)
from basemap.round0188_composition_boundary_seed43 import train_checks_close
from basemap.round0191_full_width_contrast import (
    CAPABILITY,
    H2048_EVALUATION_SCHEMA,
    H4096_EVALUATION_SCHEMA,
    HOST_RSS_LIMIT_GIB,
    ROUND_ID,
    SEED,
    SUCCESSFUL_UPDATES,
    SYNTHESIS_SCHEMA,
    TRAIN_SCHEMA,
    Round0191Error,
    h4096_train_config,
    width_decision,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0180_nodes as r0180_nodes
from experiments import round0188_nodes as base


GRAPH_SCHEMA = "round0171-prompted-8m-fuzzy-graph-v1"
PRODUCTION_CONFIG_SCHEMA = "round0191-full-h4096-production-config-v1"
GRAPH_INDEX_DESCRIPTION = r0180_nodes.GRAPH_INDEX_DESCRIPTION
ALLOWED_ACTIONS = {
    "train_full_h4096",
    "evaluate_width_arm",
    "synthesize_width_contrast",
}


def _configure_q2(_rung: str, _job: Mapping[str, Any]) -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": SEED,
        "SUCCESSFUL_UPDATES": SUCCESSFUL_UPDATES,
        "HOST_RSS_LIMIT_GIB": HOST_RSS_LIMIT_GIB,
        "Round0166Error": Round0191Error,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "TRAIN_SCHEMA": TRAIN_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "GRAPH_INDEX_DESCRIPTION": GRAPH_INDEX_DESCRIPTION,
        "GRAPH_REFERENCE_ROW_ORDER": "R0165 frozen-prefix prompted compact order",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": "R0165 compact IDs",
        "GRAPH_SOURCE_ROUND_ID": "0171",
        "GRAPH_BUILT_IN_ROUND": False,
        "POPULATION_READER": None,
        "MIN_SCALE_ROWS_EXCLUSIVE": 0,
        "scale_train_config": h4096_train_config,
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0191Error("train handler received another queue")
    _configure_q2("full", job)
    q2.run_train(active, job)


def _signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:
        raise Round0191Error(f"{label} is missing or changed") from error


def _read(path: str, *, label: str) -> dict[str, Any]:
    try:
        return prompt_contract.read_sealed(path, label=label)
    except Exception as error:
        raise Round0191Error(f"{label} is missing or unsealed") from error


def _load_width_model(
    job: Mapping[str, Any], rung: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if rung != "full":
        raise Round0191Error("R0191 evaluates only the full rung")
    arm = str(job.get("width_arm") or "")
    if arm not in {"h4096", "r0184_h2048"}:
        raise Round0191Error("width evaluation arm changed")
    _configure_q2(rung, job)
    population, population_signature = q2._read_population(job)
    graph_path = str(job["graph_manifest"])
    graph_signature = _signature(graph_path, label="accepted full graph manifest")
    graph = _read(graph_path, label="accepted full graph manifest")
    fixed = ((graph.get("search_qualification") or {}).get("cells") or {}).get(
        "64", {}
    )
    if (
        graph.get("schema") != GRAPH_SCHEMA
        or graph.get("round_id") != "0171"
        or graph.get("population") != population_signature
        or int(graph.get("retained_rows", -1)) != int(population["retained_rows"])
        or int(graph.get("dimension", -1)) != 768
        or int(graph.get("k", -1)) != 50
        or int(graph.get("directed_edge_count", -1)) <= 0
        or int((graph.get("search_qualification") or {}).get("selected_nprobe", -1))
        != 64
        or fixed.get("passed") is not True
        or (graph.get("search_qualification") or {}).get("index")
        != GRAPH_INDEX_DESCRIPTION
    ):
        raise Round0191Error("accepted full graph contract changed")

    if arm == "h4096":
        train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
        expected_schema = TRAIN_SCHEMA
        expected_round = ROUND_ID
        config_builder = h4096_train_config
    else:
        train_path = str(job["r0184_train_receipt"])
        expected_schema = "round0184-prompted-8m-dose-midpoint-train-receipt-v1"
        expected_round = "0184"
        config_builder = h2048_train_config
    train_signature = _signature(train_path, label=f"{arm} train receipt")
    train = _read(train_path, label=f"{arm} train receipt")
    config, config_sha = config_builder(
        graph_signature=graph["graph"],
        graph_manifest_signature=graph_signature,
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=int(population["retained_rows"]),
    )
    expected_draws = SUCCESSFUL_UPDATES * prompt_contract.POSITIVE_ROWS_PER_UPDATE
    accounting = train.get("train_accounting") or {}
    if arm == "h4096":
        positive_draws_close = int(train.get("consumed_positive_draws", -1)) == expected_draws
    else:
        # R0184 was sealed before consumed_positive_draws became a top-level
        # receipt field. Bind the same quantity through its immutable exact
        # accounting rather than weakening the accepted historical premise.
        positive_draws_close = (
            int(accounting.get("positive_lr_optimizer_steps", -1))
            == SUCCESSFUL_UPDATES
            and int(accounting.get("pipeline_endpoint_gather_calls", -1))
            == SUCCESSFUL_UPDATES
            and int(accounting.get("pipeline_source_rows_gathered", -1))
            == SUCCESSFUL_UPDATES * prompt_contract.BATCH_SIZE
            and int(accounting.get("pipeline_destination_rows_gathered", -1))
            == SUCCESSFUL_UPDATES * prompt_contract.BATCH_SIZE
        )
    if (
        train.get("schema") != expected_schema
        or train.get("round_id") != expected_round
        or int(train.get("training_seed", -1)) != SEED
        or train.get("population") != population_signature
        or train.get("graph_manifest") != graph_signature
        or train.get("production_config_sha256") != config_sha
        or int(train.get("optimizer_updates", -1)) != SUCCESSFUL_UPDATES
        or not positive_draws_close
        or not train_checks_close(train.get("train_checks"))
    ):
        raise Round0191Error(f"{arm} train receipt changed")
    model_path = prompt_contract.verify_signature(train["model"], label=f"{arm} model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
    expected = config["model"]
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
    if observed != expected:
        raise Round0191Error(f"{arm} model architecture changed")
    return model, train, train_signature


def _configure_evaluator(arm: str) -> None:
    schema = H4096_EVALUATION_SCHEMA if arm == "h4096" else H2048_EVALUATION_SCHEMA
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": SEED,
        "EVALUATION_SCHEMA": schema,
        "Round0188Error": Round0191Error,
        "_configure_q2": _configure_q2,
        "_load_seed43_model": _load_width_model,
    }
    for name, value in bindings.items():
        setattr(base, name, value)


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    arm = str(job.get("width_arm") or "")
    if arm not in {"h4096", "r0184_h2048"}:
        raise Round0191Error("width evaluation arm changed")
    _configure_evaluator(arm)
    base.run_evaluate(active, job)


def run_synthesize(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0191Error("synthesis handler received another queue")
    track_a_path = str(job["r0190_synthesis"])
    track_a_signature = _signature(track_a_path, label="accepted R0190 synthesis")
    track_a_receipt = _read(track_a_path, label="accepted R0190 synthesis")
    if (
        track_a_receipt.get("schema") != "round0190-three-seed-boundary-synthesis-v1"
        or track_a_receipt.get("round_id") != "0190"
    ):
        raise Round0191Error("R0190 synthesis contract changed")

    metrics: dict[str, dict[str, float]] = {}
    evaluation_signatures: dict[str, dict[str, Any]] = {}
    for arm, schema in (
        ("h4096", H4096_EVALUATION_SCHEMA),
        ("r0184_h2048", H2048_EVALUATION_SCHEMA),
    ):
        path = os.path.join(
            str(job["evaluation_outputs"][arm]), "common-core-evaluation.json"
        )
        receipt = _read(path, label=f"{arm} common-core evaluation")
        if (
            receipt.get("schema") != schema
            or receipt.get("round_id") != ROUND_ID
            or receipt.get("rung") != "full"
            or int(receipt.get("seed", -1)) != SEED
            or not (receipt.get("execution_checks") or {})
            or not all(receipt["execution_checks"].values())
        ):
            raise Round0191Error(f"{arm} evaluation contract changed")
        metrics[arm] = {
            key: float(value)
            for key, value in (receipt.get("primary_metrics") or {}).items()
        }
        evaluation_signatures[arm] = _signature(path, label=f"{arm} evaluation")
    decision = width_decision(
        track_a=track_a_receipt["decision"],
        h4096_metrics=metrics["h4096"],
        h2048_metrics=metrics["r0184_h2048"],
    )
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train = _read(train_path, label="R0191 h4096 train receipt")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0191 width synthesis"
    )
    receipt = prompt_contract.seal({
        "schema": SYNTHESIS_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "decision": decision,
        "r0190_synthesis": track_a_signature,
        "evaluations": evaluation_signatures,
        "training_economics": {
            "hidden_dimension": 4096,
            "successful_updates": int(train["optimizer_updates"]),
            "updates_per_s": float(train["steady_updates_per_s"]),
            "train_wall_s": float(train["train_wall_s"]),
            "achieved_positive_draws_per_edge": float(
                train["consumed_positive_draws_per_edge"]
            ),
        },
        "h4096_train_receipt": _signature(train_path, label="R0191 train receipt"),
        "r0184_h2048_train_receipt": _signature(
            str(job["r0184_train_receipt"]), label="R0184 train receipt"
        ),
        "scope": {
            "width_only": True,
            "hidden_dimension_change": [2048, 4096],
            "population_graph_seed_dose_schedule_sampler_optimizer_precision_frozen": True,
            "additional_width_cells_authorized": False,
        },
        "training_performed_in_synthesis_node": False,
    })
    atomic_write_new_json(
        os.path.join(output, "width-decision.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise Round0191Error(f"R0191 does not authorize action {action!r}")
    actions = {
        "train_full_h4096": run_train,
        "evaluate_width_arm": run_evaluate,
        "synthesize_width_contrast": run_synthesize,
    }
    actions[action](active, job)


__all__ = ["run_job"]
