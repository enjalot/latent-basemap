"""Execute the R0202 h4096 quarter/half matched-dose ladder."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0187_composition_nested_ladder import (
    DIMENSION,
    RUNG_ROWS,
    NestedScalePromptTrainingInput,
)
from basemap.round0188_composition_boundary_seed43 import train_checks_close
from basemap.round0202_h4096_nested_dose_ladder import (
    CAPABILITY,
    EVALUATION_SCHEMA,
    HIDDEN_DIMENSION,
    HOST_RSS_LIMIT_GIB,
    ROUND_ID,
    RUNGS,
    SEED,
    SYNTHESIS_SCHEMA,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    Round0202Error,
    ladder_summary,
    successful_updates_for_edges,
    train_config,
    train_schema,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0187_nodes as r0187_nodes
from experiments import round0188_nodes as evaluator


GRAPH_INDEX_DESCRIPTION = r0187_nodes.GRAPH_INDEX_DESCRIPTION
ALLOWED_ACTIONS = {
    "train_h4096_nested_rung",
    "evaluate_h4096_nested_rung",
    "synthesize_h4096_nested_ladder",
}


def _signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:
        raise Round0202Error(f"{label} is unavailable or changed") from error


def _read(path: str, *, label: str) -> dict[str, Any]:
    try:
        return prompt_contract.read_sealed(path, label=label)
    except Exception as error:
        raise Round0202Error(f"{label} is missing or unsealed") from error


def _configure_q2(rung: str, job: Mapping[str, Any]) -> None:
    if rung not in RUNGS:
        raise Round0202Error("h4096 nested execution rung changed")
    graph = _read(
        str(job["graph_manifest"]), label=f"accepted R0187 {rung} graph"
    )
    updates = successful_updates_for_edges(int(graph.get("directed_edge_count", -1)))
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": SEED,
        "SUCCESSFUL_UPDATES": updates,
        "HOST_RSS_LIMIT_GIB": HOST_RSS_LIMIT_GIB,
        "Round0166Error": Round0202Error,
        "GRAPH_SCHEMA": f"round0187-composition-nested-fuzzy-graph-{rung}-v1",
        "TRAIN_SCHEMA": train_schema(rung),
        "PRODUCTION_CONFIG_SCHEMA": f"round0202-{rung}-h4096-production-config-v1",
        "GRAPH_INDEX_DESCRIPTION": GRAPH_INDEX_DESCRIPTION,
        "GRAPH_REFERENCE_ROW_ORDER": f"R0187 {rung} canonical nested order",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": f"R0187 {rung} compact IDs",
        "GRAPH_SOURCE_ROUND_ID": "0187",
        "GRAPH_BUILT_IN_ROUND": False,
        "POPULATION_READER": r0187_nodes._population_reader,
        "MIN_SCALE_ROWS_EXCLUSIVE": 0,
        "ScalePromptTrainingInput": NestedScalePromptTrainingInput,
        "scale_train_config": lambda **kwargs: train_config(rung=rung, **kwargs),
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0202Error("train handler received another queue")
    rung = str(job.get("rung") or "")
    _configure_q2(rung, job)
    q2.run_train(active, job)


def _load_model(
    job: Mapping[str, Any], rung: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    _configure_q2(rung, job)
    population, population_signature = q2._read_population(job)
    graph_path = str(job["graph_manifest"])
    graph_signature = _signature(graph_path, label=f"accepted R0187 {rung} graph")
    graph = _read(graph_path, label=f"accepted R0187 {rung} graph")
    fixed = ((graph.get("search_qualification") or {}).get("cells") or {}).get(
        "64", {}
    )
    if (
        graph.get("schema")
        != f"round0187-composition-nested-fuzzy-graph-{rung}-v1"
        or graph.get("round_id") != "0187"
        or graph.get("population") != population_signature
        or int(graph.get("retained_rows", -1)) != RUNG_ROWS[rung]
        or int(graph.get("dimension", -1)) != DIMENSION
        or int(graph.get("k", -1)) != 50
        or int((graph.get("search_qualification") or {}).get("selected_nprobe", -1))
        != 64
        or fixed.get("passed") is not True
        or (graph.get("search_qualification") or {}).get("index")
        != GRAPH_INDEX_DESCRIPTION
    ):
        raise Round0202Error(f"accepted R0187 {rung} graph contract changed")
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train_signature = _signature(train_path, label=f"R0202 {rung} train receipt")
    train = _read(train_path, label=f"R0202 {rung} train receipt")
    config, config_sha = train_config(
        rung=rung,
        graph_signature=graph["graph"],
        graph_manifest_signature=graph_signature,
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=RUNG_ROWS[rung],
    )
    updates = successful_updates_for_edges(int(graph["directed_edge_count"]))
    expected_draws = updates * prompt_contract.POSITIVE_ROWS_PER_UPDATE
    if (
        train.get("schema") != train_schema(rung)
        or train.get("round_id") != ROUND_ID
        or int(train.get("training_seed", -1)) != SEED
        or train.get("population") != population_signature
        or train.get("graph_manifest") != graph_signature
        or train.get("production_config_sha256") != config_sha
        or int(train.get("optimizer_updates", -1)) != updates
        or int(train.get("consumed_positive_draws", -1)) != expected_draws
        or float(train.get("requested_positive_draws_per_edge", float("nan")))
        != TARGET_POSITIVE_DRAWS_PER_EDGE
        or not train_checks_close(train.get("train_checks"))
    ):
        raise Round0202Error(f"R0202 {rung} train receipt changed")
    model_path = prompt_contract.verify_signature(
        train["model"], label=f"R0202 {rung} model"
    )
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
    if observed != expected or model.hidden_dim != HIDDEN_DIMENSION:
        raise Round0202Error(f"R0202 {rung} model architecture changed")
    return model, train, train_signature


def _configure_evaluator() -> None:
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": SEED,
        "RUNGS": RUNGS,
        "EVALUATION_SCHEMA": EVALUATION_SCHEMA,
        "Round0188Error": Round0202Error,
        "_configure_q2": _configure_q2,
        "_load_seed43_model": _load_model,
    }
    for name, value in bindings.items():
        setattr(evaluator, name, value)


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_evaluator()
    evaluator.run_evaluate(active, job)


def run_synthesize(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0202Error("synthesis handler received another queue")
    cells: dict[str, dict[str, float]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    economics: dict[str, dict[str, Any]] = {}
    for rung in RUNGS:
        evaluation_path = os.path.join(
            str(job["evaluation_outputs"][rung]), "common-core-evaluation.json"
        )
        evaluation = _read(evaluation_path, label=f"R0202 {rung} evaluation")
        if (
            evaluation.get("schema") != EVALUATION_SCHEMA
            or evaluation.get("round_id") != ROUND_ID
            or evaluation.get("rung") != rung
            or int(evaluation.get("seed", -1)) != SEED
            or not all((evaluation.get("execution_checks") or {}).values())
        ):
            raise Round0202Error(f"R0202 {rung} evaluation contract changed")
        cells[rung] = {
            key: float(value)
            for key, value in (evaluation.get("primary_metrics") or {}).items()
        }
        evaluations[rung] = _signature(
            evaluation_path, label=f"R0202 {rung} evaluation"
        )
        train_path = os.path.join(
            str(job["train_outputs"][rung]), "train-receipt.json"
        )
        train = _read(train_path, label=f"R0202 {rung} train")
        economics[rung] = {
            "retained_rows": RUNG_ROWS[rung],
            "directed_edges": int(
                _read(str(job["graph_manifests"][rung]), label=f"{rung} graph")[
                    "directed_edge_count"
                ]
            ),
            "successful_updates": int(train["optimizer_updates"]),
            "achieved_positive_draws_per_edge": float(
                train["consumed_positive_draws_per_edge"]
            ),
            "updates_per_s": float(train["steady_updates_per_s"]),
            "train_wall_s": float(train["train_wall_s"]),
            "train_receipt": _signature(train_path, label=f"R0202 {rung} train"),
        }

    full_evaluation_path = str(job["r0191_full_evaluation"])
    full_evaluation = _read(full_evaluation_path, label="accepted R0191 full evaluation")
    if (
        full_evaluation.get("schema")
        != "round0191-full-h4096-common-core-evaluation-v1"
        or full_evaluation.get("round_id") != "0191"
        or full_evaluation.get("rung") != "full"
        or int(full_evaluation.get("seed", -1)) != SEED
        or not all((full_evaluation.get("execution_checks") or {}).values())
    ):
        raise Round0202Error("accepted R0191 full evaluation changed")
    cells["full"] = {
        key: float(value)
        for key, value in (full_evaluation.get("primary_metrics") or {}).items()
    }
    evaluations["full"] = _signature(
        full_evaluation_path, label="accepted R0191 full evaluation"
    )
    full_train_path = str(job["r0191_full_train"])
    full_train = _read(full_train_path, label="accepted R0191 full train")
    if (
        full_train.get("schema") != "round0191-full-h4096-width-train-receipt-v1"
        or full_train.get("round_id") != "0191"
        or int(full_train.get("optimizer_updates", -1)) != 1_000_000
        or not train_checks_close(full_train.get("train_checks"))
    ):
        raise Round0202Error("accepted R0191 full train changed")
    economics["full"] = {
        "retained_rows": 7_952_419,
        "directed_edges": 603_086_368,
        "successful_updates": int(full_train["optimizer_updates"]),
        "achieved_positive_draws_per_edge": float(
            full_train["consumed_positive_draws_per_edge"]
        ),
        "updates_per_s": float(full_train["steady_updates_per_s"]),
        "train_wall_s": float(full_train["train_wall_s"]),
        "train_receipt": _signature(full_train_path, label="accepted R0191 full train"),
    }
    summary = ladder_summary(cells)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0202 h4096 nested ladder synthesis"
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
            "full_endpoint": "byte-exact accepted R0191",
            "population_graph_composition_common_core_frozen": True,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "primary_registered_metric": "pile_ffr",
            "density_role": "diagnostic-only",
            "cross_width_selector": "deferred to campaign Track A3",
        },
        "training_performed_in_synthesis_node": False,
    })
    atomic_write_new_json(
        os.path.join(output, "h4096-ladder-summary.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise Round0202Error(f"R0202 does not authorize action {action!r}")
    actions = {
        "train_h4096_nested_rung": run_train,
        "evaluate_h4096_nested_rung": run_evaluate,
        "synthesize_h4096_nested_ladder": run_synthesize,
    }
    actions[action](active, job)


__all__ = ["run_job"]
