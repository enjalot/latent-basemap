"""Execute the paired seed-43 raw/drop-only replay for Round 0150."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import CURRENT_GRAPH_CURRENT_HOST
from basemap.round0149_drop_only import TREATMENT as DROP_ONLY
from basemap.round0150_seed_replay import (
    CAPABILITY,
    ROUND_ID,
    SEED,
    Round0150Error,
    build_decision,
    drop_seed43_train_config,
    raw_seed43_train_config,
)
from experiments import round0140_nodes as raw_base
from experiments import round0147_nodes as policy_base
from experiments import round0149_nodes as drop_parent


RAW = CURRENT_GRAPH_CURRENT_HOST
RENDER_SEED = 15_000


def _configure_raw() -> None:
    raw_base.ROUND_ID = ROUND_ID
    raw_base.CAPABILITY = CAPABILITY
    raw_base.SEED = SEED
    raw_base.NEW_CELLS = (RAW,)
    raw_base.RENDER_SEED = RENDER_SEED
    raw_base.ARTIFACT_SCHEMA_PREFIX = "round0150"
    raw_base.host_train_config = raw_seed43_train_config


def _configure_drop() -> None:
    drop_parent._configure_base()
    policy_base.ROUND_ID = ROUND_ID
    policy_base.CAPABILITY = CAPABILITY
    policy_base.SEED = SEED
    policy_base.RENDER_SEED = RENDER_SEED
    policy_base.ARTIFACT_SCHEMA_PREFIX = "round0150"
    policy_base.GRAPH_RECEIPT_ROUND_ID = "0149"
    policy_base.CONTROL_PANEL_ROUND_ID = ROUND_ID
    policy_base.REQUIRE_CONTROL_RESTORATION = False
    policy_base.TREATMENT_ROLE = "drop-only-historical-seed43-replay"
    policy_base.treatment_train_config = drop_seed43_train_config


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0150Error(f"{label} bytes changed")
    return actual


def _read_sealed_path(path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0150Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value, signature


def run_raw_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_raw()
    raw_base.run_host_train(active, job)


def run_drop_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_drop()
    policy_base.run_train(active, job)


def run_raw_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_raw()
    raw_base.run_panel(active, job)


def run_drop_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_drop()
    delegated = dict(job)
    delegated["r0140_panel"] = expected_input_signature(
        os.path.join(str(job["raw_panel_output"]), "functional-bisection.json")
    )
    policy_base.run_functional_panel(active, delegated)


def run_raw_universality(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_drop()
    train_path = os.path.join(str(job["raw_train_output"]), "train-receipt.json")
    train, _signature_value = _read_sealed_path(train_path, label="R0150 raw train")
    if (
        train.get("round_id") != ROUND_ID
        or train.get("release_sha") != active["manifest"]["release_sha"]
        or train.get("cell") != RAW
        or (train.get("exact_execution_receipt") or {}).get("pipeline")
        != "host_weighted_jina_paired"
    ):
        raise Round0150Error("R0150 raw train identity changed")
    delegated = dict(job)
    delegated["map_key"] = RAW
    delegated["model"] = train["model"]
    policy_base.run_universality_panel(active, delegated)


def run_drop_universality(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_drop()
    delegated = dict(job)
    delegated["map_key"] = DROP_ONLY
    policy_base.run_universality_panel(active, delegated)


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0150 seed-replay decision"
    )
    r0149, r0149_signature = _read_sealed_path(
        str(job["r0149_decision"]["canonical_path"]), label="accepted R0149 decision"
    )
    if r0149_signature != dict(job["r0149_decision"]):
        raise Round0150Error("accepted R0149 decision signature changed")

    raw_path = os.path.join(str(job["raw_panel_output"]), "functional-bisection.json")
    raw_panel, raw_signature = _read_sealed_path(raw_path, label="R0150 raw panel")
    drop_path = os.path.join(str(job["drop_panel_output"]), "functional-panel.json")
    drop_panel, drop_signature = _read_sealed_path(drop_path, label="R0150 drop panel")
    if raw_panel.get("round_id") != ROUND_ID or drop_panel.get("round_id") != ROUND_ID:
        raise Round0150Error("R0150 functional panel identity changed")
    raw_cell = raw_panel.get("cells", {}).get(RAW)
    copied_raw = drop_panel.get("cells", {}).get(RAW)
    drop_cell = drop_panel.get("cells", {}).get(DROP_ONLY)
    if (
        not isinstance(raw_cell, Mapping)
        or not isinstance(copied_raw, Mapping)
        or not isinstance(drop_cell, Mapping)
        or dict(raw_cell) != dict(copied_raw)
    ):
        raise Round0150Error("R0150 paired panel control binding changed")

    decision = build_decision(r0149, {RAW: raw_cell, DROP_ONLY: drop_cell})
    universality: dict[str, Any] = {}
    for key, output_root in job["universality_outputs"].items():
        path = os.path.join(str(output_root), "universality-panel.json")
        panel, signature = _read_sealed_path(path, label=f"R0150 {key} universality")
        if (
            panel.get("round_id") != ROUND_ID
            or panel.get("map_key") != key
            or panel.get("role")
            != "diagnostic-only; never part of the restoration selector"
        ):
            raise Round0150Error("R0150 universality identity changed")
        universality[key] = {
            "panel": signature,
            "metrics": {
                name: report["metrics"]
                for name, report in panel.get("probes", {}).items()
            },
        }

    receipt = seal({
        **decision,
        "release_sha": active["manifest"]["release_sha"],
        "r0149_decision": r0149_signature,
        "raw_functional_panel": raw_signature,
        "paired_functional_panel": drop_signature,
        "universality_diagnostic": universality,
        "training_performed": True,
    })
    atomic_write_new_json(os.path.join(output, "decision.json"), receipt, immutable=True)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    handlers = {
        "train_raw_seed43": run_raw_train,
        "train_drop_seed43": run_drop_train,
        "score_raw_functional": run_raw_panel,
        "score_drop_functional": run_drop_panel,
        "score_raw_universality": run_raw_universality,
        "score_drop_universality": run_drop_universality,
        "decide_seed_replay": run_decision,
    }
    handler = handlers.get(action)
    if handler is None:
        raise Round0150Error(f"unknown R0150 action: {action}")
    handler(active, job)
