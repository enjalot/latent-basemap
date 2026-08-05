"""Execute the R0189 seed-44 half-to-full composition-boundary replay."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0187_composition_nested_ladder import PRIMARY_METRICS
from basemap.round0189_composition_boundary_seed44 import (
    CAPABILITY,
    EVALUATION_SCHEMA,
    ROUND_ID,
    RUNGS,
    SEED,
    SYNTHESIS_SCHEMA,
    Round0189Error,
    boundary_decision,
    successful_updates_for_edges,
    train_config,
    train_schema,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0187_nodes as r0187_nodes
from experiments import round0188_nodes as base


ALLOWED_ACTIONS = {
    "train_seed44_boundary_rung",
    "evaluate_seed44_boundary_rung",
    "synthesize_seed44_boundary",
}


def _configure_q2(rung: str, job: Mapping[str, Any]) -> None:
    """Bind the shared Q2 executor to the seed-44 replay contract."""
    if rung not in RUNGS:
        raise Round0189Error("training rung changed")
    graph = base._read_sealed(
        str(job["graph_manifest"]), label=f"accepted {rung} graph manifest"
    )
    updates = successful_updates_for_edges(int(graph.get("directed_edge_count", -1)))
    meta = base.GRAPH_META[rung]
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": SEED,
        "SUCCESSFUL_UPDATES": updates,
        "HOST_RSS_LIMIT_GIB": 28.0,
        "Round0166Error": Round0189Error,
        "GRAPH_SCHEMA": meta["schema"],
        "TRAIN_SCHEMA": train_schema(rung),
        "PRODUCTION_CONFIG_SCHEMA": f"round0189-{rung}-seed44-production-config-v1",
        "GRAPH_INDEX_DESCRIPTION": meta["index"],
        "GRAPH_REFERENCE_ROW_ORDER": meta["row_order"],
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": meta["anchor_namespace"],
        "GRAPH_SOURCE_ROUND_ID": meta["source_round"],
        "GRAPH_BUILT_IN_ROUND": False,
        "POPULATION_READER": base._half_population_reader if rung == "half" else None,
        "MIN_SCALE_ROWS_EXCLUSIVE": 0,
        "ScalePromptTrainingInput": r0187_nodes.NestedScalePromptTrainingInput,
        "scale_train_config": lambda **kwargs: train_config(rung=rung, **kwargs),
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def _configure_base() -> None:
    """Adapt the authenticated R0188 evaluator without copying its GPU path."""
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": SEED,
        "EVALUATION_SCHEMA": EVALUATION_SCHEMA,
        "SYNTHESIS_SCHEMA": SYNTHESIS_SCHEMA,
        "Round0188Error": Round0189Error,
        "train_config": train_config,
        "train_schema": train_schema,
        "_configure_q2": _configure_q2,
    }
    for name, value in bindings.items():
        setattr(base, name, value)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_base()
    base.run_train(active, job)


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_base()
    base.run_evaluate(active, job)


def run_synthesize(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_base()
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0189Error("synthesis handler received another queue")
    prior_path = str(job["r0187_ladder_decision"])
    prior_signature = base._signature(
        prior_path, label="accepted R0187 ladder decision"
    )
    prior = base._read_sealed(prior_path, label="accepted R0187 ladder decision")
    prior_decision = prior.get("decision") or {}
    if (
        prior.get("schema") != "round0187-composition-nested-ladder-decision-v1"
        or prior.get("round_id") != "0187"
        or prior_decision.get("outcome") != "composition-controlled-size-regression"
        or (prior_decision.get("concordant_material_regression") or {}).get(
            "pile_ffr"
        )
        is not True
    ):
        raise Round0189Error("accepted R0187 decision branch changed")

    seed44: dict[str, dict[str, float]] = {}
    evaluation_signatures: dict[str, dict[str, Any]] = {}
    for rung in RUNGS:
        path = os.path.join(
            str(job["evaluation_outputs"][rung]), "common-core-evaluation.json"
        )
        signature = base._signature(path, label=f"R0189 {rung} evaluation")
        receipt = base._read_sealed(path, label=f"R0189 {rung} evaluation")
        if (
            receipt.get("schema") != EVALUATION_SCHEMA
            or receipt.get("round_id") != ROUND_ID
            or receipt.get("rung") != rung
            or int(receipt.get("seed", -1)) != SEED
            or not all((receipt.get("execution_checks") or {}).values())
        ):
            raise Round0189Error(f"R0189 {rung} evaluation contract changed")
        seed44[rung] = {
            key: float(value)
            for key, value in (receipt.get("primary_metrics") or {}).items()
        }
        evaluation_signatures[rung] = signature
    seed42 = {
        rung: {
            key: float(value)
            for key, value in prior_decision["cells"][rung].items()
        }
        for rung in RUNGS
    }
    decision = boundary_decision(seed42=seed42, seed44=seed44)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0189 seed-44 boundary synthesis"
    )
    receipt = prompt_contract.seal({
        "schema": SYNTHESIS_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "decision": decision,
        "r0187_ladder_decision": prior_signature,
        "evaluations": evaluation_signatures,
        "scientific_scope": {
            "boundary": "half_to_full",
            "registered_metric": "pile_ffr",
            "seed42_source": "accepted R0187",
            "replay_seed": SEED,
            "population_graph_and_dose_reused": True,
            "hidden_dimension": 2048,
            "primary_metric_vector": list(PRIMARY_METRICS),
            "other_seed44_metric_misses_role": "diagnostic-only",
            "aggregate_seed42_seed43_seed44_decision_deferred": True,
        },
        "training_performed_in_synthesis_node": False,
    })
    atomic_write_new_json(
        os.path.join(output, "boundary-decision.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise Round0189Error(f"R0189 does not authorize action {action!r}")
    actions = {
        "train_seed44_boundary_rung": run_train,
        "evaluate_seed44_boundary_rung": run_evaluate,
        "synthesize_seed44_boundary": run_synthesize,
    }
    actions[action](active, job)


__all__ = ["run_job"]
