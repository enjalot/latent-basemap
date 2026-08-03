"""Contract tests for the R0186 graph and exact dose planner."""
from __future__ import annotations

import json

from basemap.artifact_identity import expected_input_signature
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0186_prompted_u12_graph import (
    BASELINE_GRAPH_EDGES,
    BASELINE_SUCCESSFUL_UPDATES,
    CAPABILITY,
    DOSE_PLAN_SCHEMA,
    GRAPH_SCHEMA,
    ROUND_ID,
    successful_updates_for_edges,
)
from experiments import round0169_nodes, round0186_nodes


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_update_horizon_uses_exact_ceiling_arithmetic() -> None:
    assert successful_updates_for_edges(BASELINE_GRAPH_EDGES) == 500_000
    assert successful_updates_for_edges(BASELINE_GRAPH_EDGES * 2) == 1_000_000
    edges = BASELINE_GRAPH_EDGES * 2 + 1
    updates = successful_updates_for_edges(edges)
    assert updates == 1_000_001
    assert (
        (updates - 1) * BASELINE_GRAPH_EDGES
        < BASELINE_SUCCESSFUL_UPDATES * edges
        <= updates * BASELINE_GRAPH_EDGES
    )


def test_graph_dispatch_rebinds_q3_kernel(monkeypatch) -> None:
    names = ("ROUND_ID", "CAPABILITY", "GRAPH_SCHEMA", "Round0169Error")
    before = {name: getattr(round0169_nodes, name) for name in names}
    observed = {}
    monkeypatch.setattr(
        round0169_nodes,
        "run_build_graph",
        lambda active, job: observed.update({
            "round": round0169_nodes.ROUND_ID,
            "capability": round0169_nodes.CAPABILITY,
            "schema": round0169_nodes.GRAPH_SCHEMA,
        }),
    )
    try:
        round0186_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}},
            {"action": "build_graph_and_reference"},
        )
        assert observed == {
            "round": ROUND_ID,
            "capability": CAPABILITY,
            "schema": GRAPH_SCHEMA,
        }
    finally:
        for name, value in before.items():
            setattr(round0169_nodes, name, value)


def test_dose_plan_binds_qualified_graph_and_reports_queue_fit(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(round0186_nodes, "ROWS", 10)
    graph_path = tmp_path / "graph.npz"
    graph_path.write_bytes(b"graph")
    graph_signature = expected_input_signature(graph_path)
    edges = BASELINE_GRAPH_EDGES * 2
    manifest = prompt_contract.seal({
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "retained_rows": 10,
        "directed_edge_count": edges,
        "graph": graph_signature,
        "search_qualification": {
            "cells": {
                "64": {
                    "passed": True,
                    "mean_recall_at_49": 0.95,
                    "p10_recall_at_49": 0.85,
                }
            }
        },
    })
    manifest_path = tmp_path / "graph-manifest.json"
    _write_json(manifest_path, manifest)
    output = tmp_path / "output"
    round0186_nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "a" * 40}},
        {
            "action": "derive_dose_plan",
            "graph_manifest": str(manifest_path),
            "outputs": [str(output)],
        },
    )
    with open(output / "dose-plan.json", encoding="utf-8") as handle:
        plan = json.load(handle)
    assert plan["schema"] == DOSE_PLAN_SCHEMA
    assert plan["capabilities"] == [CAPABILITY]
    assert plan["successful_positive_lr_updates"] == 1_000_000
    assert plan["first_whole_update_at_or_above_target"] is True
    assert plan["graph"] == graph_signature
    assert isinstance(
        plan["runtime_projection"][
            "fits_single_eight_gpu_hour_queue_at_reference_rate"
        ],
        bool,
    )
