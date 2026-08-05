"""Execute the R0192 mixed-quarter seed-43/44 family completion."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0192_quarter_seed_family import (
    CAPABILITY,
    EVALUATION_SCHEMA,
    ROUND_ID,
    ROWS,
    RUNG,
    SEEDS,
    SYNTHESIS_SCHEMA,
    Round0192Error,
    seed_family,
    successful_updates_for_edges,
    train_config,
    train_schema,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0187_nodes as r0187_nodes
from experiments import round0188_nodes as base


QUARTER_META = {
    "schema": "round0187-composition-nested-fuzzy-graph-quarter-v1",
    "source_round": "0187",
    "index": r0187_nodes.GRAPH_INDEX_DESCRIPTION,
    "row_order": "R0187 quarter canonical order within corpus",
    "anchor_namespace": "R0187 quarter compact IDs",
}
ALLOWED_ACTIONS = {
    "train_quarter_seed",
    "evaluate_quarter_seed",
    "synthesize_quarter_seed_family",
}


def _seed(job: Mapping[str, Any]) -> int:
    seed = int(job.get("seed", -1))
    if seed not in SEEDS:
        raise Round0192Error("R0192 job seed changed")
    return seed


def _population_reader(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return r0187_nodes._population_reader({**dict(job), "rung": RUNG})


def _configure_q2(rung: str, job: Mapping[str, Any]) -> None:
    if rung != RUNG:
        raise Round0192Error("R0192 trains only the quarter rung")
    seed = _seed(job)
    graph = base._read_sealed(
        str(job["graph_manifest"]), label="accepted quarter graph manifest"
    )
    updates = successful_updates_for_edges(int(graph.get("directed_edge_count", -1)))
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": seed,
        "SUCCESSFUL_UPDATES": updates,
        "HOST_RSS_LIMIT_GIB": 28.0,
        "Round0166Error": Round0192Error,
        "GRAPH_SCHEMA": QUARTER_META["schema"],
        "TRAIN_SCHEMA": train_schema(seed),
        "PRODUCTION_CONFIG_SCHEMA": (
            f"round0192-quarter-seed{seed}-production-config-v1"
        ),
        "GRAPH_INDEX_DESCRIPTION": QUARTER_META["index"],
        "GRAPH_REFERENCE_ROW_ORDER": QUARTER_META["row_order"],
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": QUARTER_META["anchor_namespace"],
        "GRAPH_SOURCE_ROUND_ID": QUARTER_META["source_round"],
        "GRAPH_BUILT_IN_ROUND": False,
        "POPULATION_READER": _population_reader,
        "MIN_SCALE_ROWS_EXCLUSIVE": 0,
        "ScalePromptTrainingInput": r0187_nodes.NestedScalePromptTrainingInput,
        "scale_train_config": lambda **kwargs: train_config(seed=seed, **kwargs),
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def _configure_base(job: Mapping[str, Any]) -> None:
    seed = _seed(job)
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "SEED": seed,
        "RUNGS": (RUNG,),
        "EVALUATION_SCHEMA": EVALUATION_SCHEMA,
        "Round0188Error": Round0192Error,
        "GRAPH_META": {RUNG: QUARTER_META},
        "train_config": lambda *, rung, **kwargs: train_config(
            seed=seed, **kwargs
        ),
        "train_schema": lambda rung: train_schema(seed),
        "_configure_q2": _configure_q2,
    }
    for name, value in bindings.items():
        setattr(base, name, value)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_base(job)
    base.run_train(active, {**dict(job), "rung": RUNG})


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_base(job)
    base.run_evaluate(active, {**dict(job), "rung": RUNG})


def run_synthesize(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0192Error("synthesis handler received another queue")
    evaluations: dict[int, dict[str, Any]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    seed42_path = str(job["r0187_quarter_evaluation"])
    seed42 = base._read_sealed(seed42_path, label="accepted R0187 quarter evaluation")
    seed42_train = base._read_sealed(
        prompt_contract.verify_signature(
            seed42["train_receipt"], label="accepted R0187 quarter train receipt"
        ),
        label="accepted R0187 quarter train receipt",
    )
    if (
        seed42.get("schema")
        != "round0187-composition-nested-common-core-evaluation-v1"
        or seed42.get("round_id") != "0187"
        or seed42.get("rung") != RUNG
        # R0187's evaluation schema predates the explicit seed field; bind
        # seed 42 through its exact sealed train receipt instead.
        or seed42.get("seed") is not None
        or seed42_train.get("round_id") != "0187"
        or int(seed42_train.get("training_seed", -1)) != 42
        or not all((seed42.get("execution_checks") or {}).values())
    ):
        raise Round0192Error("accepted seed-42 quarter evaluation changed")
    evaluations[42] = seed42
    signatures["42"] = base._signature(seed42_path, label="seed42 evaluation")
    for seed in SEEDS:
        path = os.path.join(
            str(job["evaluation_outputs"][str(seed)]),
            "common-core-evaluation.json",
        )
        evaluation = base._read_sealed(path, label=f"seed{seed} quarter evaluation")
        if (
            evaluation.get("schema") != EVALUATION_SCHEMA
            or evaluation.get("round_id") != ROUND_ID
            or evaluation.get("rung") != RUNG
            or int(evaluation.get("seed", -1)) != seed
            or not all((evaluation.get("execution_checks") or {}).values())
        ):
            raise Round0192Error(f"seed {seed} quarter evaluation changed")
        evaluations[seed] = evaluation
        signatures[str(seed)] = base._signature(path, label=f"seed{seed} evaluation")
    family = seed_family(evaluations=evaluations)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0192 quarter seed family"
    )
    receipt = prompt_contract.seal({
        "schema": SYNTHESIS_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "family": family,
        "evaluations": signatures,
        "scope": {
            "rung": RUNG,
            "rows": ROWS,
            "new_training_seeds": list(SEEDS),
            "seed42_source_round": "0187",
            "population_graph_dose_recipe_frozen": True,
            "gate_registration_performed": False,
        },
        "training_performed_in_synthesis_node": False,
    })
    atomic_write_new_json(
        os.path.join(output, "quarter-seed-family.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise Round0192Error(f"R0192 does not authorize action {action!r}")
    actions = {
        "train_quarter_seed": run_train,
        "evaluate_quarter_seed": run_evaluate,
        "synthesize_quarter_seed_family": run_synthesize,
    }
    actions[action](active, job)


__all__ = ["run_job"]
