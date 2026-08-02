"""Execute R0154's raw historical-row seeds 44 and 45."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import CURRENT_GRAPH_CURRENT_HOST
from basemap.round0154_seed_variance import (
    CAPABILITY,
    RAW,
    ROUND_ID,
    SEEDS,
    Round0154Error,
    build_seed_evidence,
    raw_seed_train_config,
)
from experiments import round0140_nodes as raw_base
from experiments.round0153_nodes import _load_frozen_universe, _score_coordinates


def _configure(seed: int) -> None:
    if seed not in SEEDS:
        raise Round0154Error("R0154 seed changed")

    def config_factory(**kwargs: Any) -> tuple[dict[str, Any], str]:
        return raw_seed_train_config(seed=seed, **kwargs)

    raw_base.ROUND_ID = ROUND_ID
    raw_base.CAPABILITY = CAPABILITY
    raw_base.SEED = seed
    raw_base.NEW_CELLS = (RAW,)
    raw_base.RENDER_SEED = 15_400 + seed
    raw_base.ARTIFACT_SCHEMA_PREFIX = f"round0154-seed{seed}"
    raw_base.host_train_config = config_factory


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0154Error(f"{label} bytes changed")
    return actual


def _read_sealed(path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0154Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value, signature


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure(int(job["seed"]))
    raw_base.run_host_train(active, job)


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure(int(job["seed"]))
    raw_base.run_panel(active, job)


def run_density(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0154 density-v2 panel"
    )
    retained, anchors, high_radius, lineage, _frozen = _load_frozen_universe(job)
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "anchor_compact_rows": anchors,
        "anchor_global_rows": retained[anchors],
        "high_radius": high_radius,
    }
    panels = job.get("functional_panels")
    if not isinstance(panels, Mapping) or set(panels) != {
        str(seed) for seed in SEEDS
    }:
        raise Round0154Error("R0154 density panel set changed")
    for seed in SEEDS:
        path = os.path.join(
            str(panels[str(seed)]), "functional-bisection.json"
        )
        panel, panel_signature = _read_sealed(
            path, label=f"R0154 seed {seed} functional panel"
        )
        cell = panel.get("cells", {}).get(RAW)
        if (
            panel.get("round_id") != ROUND_ID
            or not isinstance(cell, Mapping)
            or cell.get("seed") != seed
            or not isinstance(cell.get("coordinates"), Mapping)
        ):
            raise Round0154Error(f"R0154 seed {seed} panel identity changed")
        scored, cell_arrays = _score_coordinates(
            cell["coordinates"],
            retained=retained,
            anchors=anchors,
            high_radius=high_radius,
            workers=int(job.get("cpu_workers", 4)),
            label=f"R0154 raw seed {seed}",
        )
        cells[f"seed{seed}"] = {
            "seed": seed,
            "functional_panel": panel_signature,
            **scored,
        }
        for suffix, value in cell_arrays.items():
            arrays[f"seed{seed}__{suffix}"] = value
    arrays_path = os.path.join(output, "density-v2-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": "round0154-raw-seed-density-v2-panel-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "lineage": lineage,
        "cells": cells,
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "floor_changed": False,
    })
    atomic_write_new_json(
        os.path.join(output, "density-v2.json"), receipt, immutable=True
    )


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0154 seed evidence"
    )
    panels: dict[int, dict[str, Any]] = {}
    panel_signatures: dict[str, Any] = {}
    for seed in SEEDS:
        path = os.path.join(
            str(job["functional_panels"][str(seed)]),
            "functional-bisection.json",
        )
        panel, signature = _read_sealed(path, label=f"R0154 seed {seed} panel")
        panels[seed] = panel
        panel_signatures[f"seed{seed}"] = signature
    density_path = os.path.join(str(job["density_output"]), "density-v2.json")
    density, density_signature = _read_sealed(
        density_path, label="R0154 density-v2 panel"
    )
    if (
        density.get("round_id") != ROUND_ID
        or density.get("release_sha") != active["manifest"]["release_sha"]
    ):
        raise Round0154Error("R0154 density release identity changed")
    evidence = build_seed_evidence(panels, density["cells"])
    train_receipts: dict[str, Any] = {}
    for seed in SEEDS:
        cell = panels[seed]["cells"][RAW]
        train_receipts[f"seed{seed}"] = dict(cell["training"]["train"])
    receipt = seal({
        **evidence,
        "release_sha": active["manifest"]["release_sha"],
        "functional_panels": panel_signatures,
        "density_panel": density_signature,
        "train_receipts": train_receipts,
        "no_graph_build": True,
        "graph_reused_byte_exact": True,
    })
    atomic_write_new_json(
        os.path.join(output, "seed-evidence.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0154Error("R0154 handler received another queue")
    action = str(job.get("action") or "")
    handlers = {
        "train_raw_seed": run_train,
        "score_raw_functional": run_panel,
        "score_raw_density_v2": run_density,
        "seal_seed_evidence": run_decision,
    }
    handler = handlers.get(action)
    if handler is None:
        raise Round0154Error(f"unknown R0154 action: {action}")
    handler(active, job)
