#!/usr/bin/env python3
"""Materialize, but never launch, the accepted-negative R0149 queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0027_program import CENTROIDS, TRAIN_PATH
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    RESTORATION_FLOORS,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
)
from basemap.round0147_row_policy import TREATMENT as SIZE_PRESERVING_TREATMENT
from basemap.round0149_drop_only import (
    CAPABILITY,
    RAW_PREFIX_EXCLUDED_ROWS,
    RAW_PREFIX_ROWS,
    ROUND_ID,
    ROWS,
    ROW_UNIVERSE,
    TREATMENT,
    derive_drop_only_selection,
    treatment_train_config,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _accepted_review, _frontmatter
from experiments import prepare_round0147_queue as r0147_prep
from experiments.round0147_nodes import training_accounting_mismatches


ROUND_ROOT = "/data/latent-basemap/runs/round-0149"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0149-*.md")
R0147_ROOT = "/data/latent-basemap/runs/round-0147/queue/artifacts"
R0147_DECISION = os.path.join(
    R0147_ROOT,
    "jina-2m-historical-row-policy-duplicate-control-v1",
    "decision.json",
)
R0147_PANEL = os.path.join(
    R0147_ROOT, "functional-row-policy-panel", "functional-panel.json"
)
R0147_SELECTION = os.path.join(
    R0147_ROOT, "historical-eligibility-selection", "selection-receipt.json"
)
R0147_UNIVERSALITY = {
    CURRENT_GRAPH_CURRENT_HOST: os.path.join(
        R0147_ROOT,
        f"universality-{CURRENT_GRAPH_CURRENT_HOST}",
        "universality-panel.json",
    ),
    SIZE_PRESERVING_TREATMENT: os.path.join(
        R0147_ROOT,
        f"universality-{SIZE_PRESERVING_TREATMENT}",
        "universality-panel.json",
    ),
}

GPU_HOURS_MINIMUM = 1.20
GPU_HOURS_EXPECTED = 1.40
GPU_HOURS_P90 = 1.55
GPU_HOURS_MAXIMUM = 2.25


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _read_sealed(path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    value = _read_json(path)
    validate_seal(value, label=label)
    return value, signature


def _issued_round(release_sha: str) -> tuple[str, dict[str, Any]]:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0149 requires exactly one issued round; found {len(candidates)}"
        )
    if _frontmatter(candidates[0]).get("base_commit") != release_sha:
        raise RuntimeError("R0149 issued base_commit differs from release")
    return candidates[0], expected_input_signature(candidates[0])


def _accepted_activation() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    review_inputs = _accepted_review(
        "0147", "jina-2m-historical-row-policy-duplicate-control-v1"
    )
    if (
        os.path.basename(review_inputs[1]["canonical_path"])
        != "result-0147-2026-08-01.md"
        or os.path.basename(review_inputs[2]["canonical_path"])
        != "review-0147-2026-08-01.md"
    ):
        raise RuntimeError("R0149 requires the exact accepted R0147 result/review")
    decision, decision_signature = _read_sealed(
        R0147_DECISION, label="accepted R0147 decision"
    )
    panel, panel_signature = _read_sealed(
        R0147_PANEL, label="accepted R0147 functional panel"
    )
    selection, selection_signature = _read_sealed(
        R0147_SELECTION, label="accepted R0147 selection"
    )
    universality: dict[str, dict[str, Any]] = {}
    for key, path in R0147_UNIVERSALITY.items():
        value, signature = _read_sealed(path, label=f"accepted R0147 {key} OOD")
        if (
            value.get("round_id") != "0147"
            or value.get("map_key") != key
            or value.get("role")
            != "diagnostic-only; never part of the restoration selector"
        ):
            raise RuntimeError("R0147 universality evidence changed")
        universality[key] = signature
    if (
        decision.get("round_id") != "0147"
        or decision.get("capability")
        != "jina-2m-historical-row-policy-duplicate-control-v1"
        or decision.get("outcome")
        != "eligible-historical-row-policy-does-not-restore"
        or decision.get("next_action")
        != "decompose-exclusion-and-replacement-policy-before-scale"
        or decision.get("duplicate_control_compatible_with_restoration")
        is not False
        or decision.get("functional_panel") != panel_signature
        or decision.get("selection_receipt") != selection_signature
        or panel.get("round_id") != "0147"
        or selection.get("round_id") != "0147"
        or selection.get("target_rows") != RAW_PREFIX_ROWS
        or selection.get("selection_summary", {}).get("raw_prefix_excluded_rows")
        != RAW_PREFIX_EXCLUDED_ROWS
    ):
        raise RuntimeError("accepted R0147 negative activation changed")
    for key, signature in universality.items():
        if decision.get("universality_diagnostic", {}).get(key, {}).get("panel") != signature:
            raise RuntimeError("R0147 decision/OOD binding changed")
    return (
        review_inputs,
        decision_signature,
        panel_signature,
        selection_signature,
        universality,
    )


def _pytest_smoke(*, release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0149 pytest checkout is not at the requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0149_drop_only.py",
        "tests/test_round0147_nodes.py",
        "tests/test_round0147_row_policy.py",
        "tests/test_round0104_training.py",
        "tests/test_round0140_subsystem_bisection.py",
        "tests/test_round0142_jina_universality.py",
        "tests/test_panel_v2.py",
    ]
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    receipt = seal({
        "schema": "round0149-release-pytest-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
    })
    validate_seal(receipt, label="R0149 release pytest")
    if completed.returncode != 0:
        raise RuntimeError(f"R0149 release pytest failed:\n{completed.stdout}\n{completed.stderr}")
    return receipt


def _drop_only_cpu_smoke(
    *, parent_selection: Mapping[str, Any]
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("R0149 CPU smoke requires CUDA_VISIBLE_DEVICES='' or '-1'")
    started = time.monotonic()
    arrays_signature = expected_input_signature(
        parent_selection["selection_arrays"]["canonical_path"]
    )
    if arrays_signature != parent_selection["selection_arrays"]:
        raise RuntimeError("R0147 selection arrays changed during R0149 smoke")
    with np.load(arrays_signature["canonical_path"], allow_pickle=False) as archive:
        selected, summary = derive_drop_only_selection(
            {key: np.asarray(archive[key]) for key in archive.files},
            parent_summary=parent_selection["selection_summary"],
        )
    config, config_sha = treatment_train_config(
        graph_signature={
            "canonical_path": "/cpu-smoke/graph.npz",
            "kind": "file",
            "bytes": 1,
            "sha256": "1" * 64,
        },
        graph_manifest_signature={
            "canonical_path": "/cpu-smoke/graph-manifest.json",
            "kind": "file",
            "bytes": 1,
            "sha256": "2" * 64,
        },
        graph_edges=123_456,
        source_sha256=str(parent_selection["staged_source"]["sha256"]),
        selection_sha256="3" * 64,
    )
    batch_size = int(config["optimizer"]["batch_size"])
    rows_gathered = SUCCESSFUL_UPDATES * batch_size
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "source_rows_gathered": rows_gathered,
        "destination_rows_gathered": rows_gathered,
        "host_prefetch_producer_batches": SUCCESSFUL_UPDATES + 1,
        "host_prefetch_consumer_batches": SUCCESSFUL_UPDATES,
    }
    accounting = {
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": 123_456,
    }
    mismatches = training_accounting_mismatches(
        accounting=accounting,
        runtime=runtime,
        expected_pipeline=config["execution"]["expected_pipeline_stamp"],
        graph_edges=123_456,
        batch_size=batch_size,
        profiler={"aborted": False},
        rate=TRAIN_MINIMUM_UPDATES_PER_S + 1.0,
    )
    if mismatches:
        raise RuntimeError(f"R0149 CPU smoke accounting failed: {mismatches}")
    receipt = seal({
        "schema": "round0149-drop-only-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parent_selection_arrays": arrays_signature,
        "selected_rows": len(selected["historical_positions"]),
        "selection_summary": summary,
        "training_config_sha256": config_sha,
        "paired_invariant_rows": config["paired_invariant"]["rows"],
        "row_universe": config["execution"]["expected_pipeline_stamp"]["row_universe"],
        "negative_sampling": config["execution"]["expected_pipeline_stamp"]["negative_sampling"],
        "accounting_mismatches": mismatches,
        "wall_seconds": time.monotonic() - started,
    })
    validate_seal(receipt, label="R0149 drop-only CPU smoke")
    return receipt


def prepare_round0149(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0149 release SHA must be one full commit")
    round_path, round_signature = _issued_round(release_sha)
    (
        review_inputs,
        r0147_decision_signature,
        r0147_panel_signature,
        r0147_selection_signature,
        r0147_universality,
    ) = _accepted_activation()
    r0147_selection, _ = _read_sealed(
        R0147_SELECTION, label="accepted R0147 selection"
    )
    _r0140, r0140_panel_signature = _read_sealed(
        r0147_prep.R0140_PANEL, label="accepted R0140 functional panel"
    )
    control = _r0140["cells"][CURRENT_GRAPH_CURRENT_HOST]
    control_train, control_inputs = r0147_prep._r0140_control(control)
    shared, shared_inputs = r0147_prep._shared_reference()
    inventory, inventory_signature, excluded, inventory_inputs = (
        r0147_prep._inventory_bundle()
    )
    (
        common_outputs,
        ood_control,
        dadabase,
        dadabase_texts,
        beir,
        universality_inputs,
    ) = r0147_prep._universality_inputs()

    queue_root = create_fresh_directory(queue_root, label="R0149 drop-only queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    pytest_path = os.path.join(preflight, "release-pytest.json")
    atomic_write_new_json(pytest_path, _pytest_smoke(release_sha=release_sha), immutable=True)
    pytest_signature = expected_input_signature(pytest_path)
    smoke_path = os.path.join(preflight, "drop-only-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path,
        _drop_only_cpu_smoke(parent_selection=r0147_selection),
        immutable=True,
    )
    smoke_signature = expected_input_signature(smoke_path)
    pipeline_smoke_path = os.path.join(preflight, "inherited-pipeline-cpu-smoke.json")
    pipeline_smoke = r0147_prep._cpu_smoke(
        inventory=inventory,
        excluded=excluded,
        model_signature=control_train["model"],
        output_root=preflight,
    )
    atomic_write_new_json(
        pipeline_smoke_path, pipeline_smoke, immutable=True
    )
    pipeline_smoke_signature = expected_input_signature(pipeline_smoke_path)

    staged_signature = expected_input_signature(
        r0147_selection["staged_source"]["canonical_path"]
    )
    if staged_signature != r0147_selection["staged_source"]:
        raise RuntimeError("accepted R0147 staged source changed")
    arrays_signature = expected_input_signature(
        r0147_selection["selection_arrays"]["canonical_path"]
    )
    if arrays_signature != r0147_selection["selection_arrays"]:
        raise RuntimeError("accepted R0147 selection arrays changed")
    external_inputs = _dedupe([
        round_signature,
        *review_inputs,
        r0147_decision_signature,
        r0147_panel_signature,
        r0147_selection_signature,
        arrays_signature,
        staged_signature,
        *r0147_universality.values(),
        r0140_panel_signature,
        inventory_signature,
        *inventory_inputs,
        expected_input_signature(TRAIN_PATH),
        *control_inputs,
        *shared_inputs,
        *universality_inputs,
        *[
            expected_input_signature(item["path"])
            for item in CENTROIDS.values()
        ],
        pytest_signature,
        smoke_signature,
        pipeline_smoke["published_checkpoint"],
        pipeline_smoke_signature,
    ])

    selection_output = os.path.join(artifacts, "drop-only-historical-selection")
    graph_output = os.path.join(artifacts, "current-graph-drop-only-historical")
    train_output = os.path.join(artifacts, TREATMENT, "train")
    functional_output = os.path.join(artifacts, "functional-drop-only-panel")
    drop_universality = os.path.join(artifacts, f"universality-{TREATMENT}")
    decision_output = os.path.join(artifacts, CAPABILITY)
    common_panel = {
        "source": expected_input_signature(TRAIN_PATH),
        "shared_reference_receipt": shared_inputs[0],
        "high_d_reference": dict(shared["high_d_reference"]),
        "query_truth": dict(shared["query_truth"]),
        "query_embeddings": dict(shared["query_embeddings"]),
        "centroids": {
            str(k): expected_input_signature(value["path"])
            for k, value in CENTROIDS.items()
        },
    }
    common_ood = {
        "common_outputs": common_outputs,
        "control_embeddings": ood_control,
        "dadabase": dadabase,
        "dadabase_texts": dadabase_texts,
        "beir": beir,
    }
    module = "experiments.round0149_nodes"
    jobs: list[dict[str, Any]] = [{
        "id": "materialize_drop_only_selection",
        "action": "materialize_selection",
        "r0147_selection_receipt": r0147_selection_signature,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [selection_output],
        "done_marker": os.path.join(artifacts, "materialize-selection.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 30.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
    }, {
        "id": "build_current_graph_drop_only_historical",
        "action": "build_graph",
        "selection_output": selection_output,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["materialize_drop_only_selection"],
        "outputs": [graph_output],
        "done_marker": os.path.join(artifacts, "build-treatment-graph.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 240.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "train_drop_only_historical_current_host",
        "action": "train",
        "selection_output": selection_output,
        "graph_output": graph_output,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["build_current_graph_drop_only_historical"],
        "outputs": [train_output],
        "done_marker": os.path.join(artifacts, "train-treatment.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 5_100.0,
        "node_policy": {"gpu_required": True, "training_performed": True},
    }, {
        "id": "score_functional_drop_only",
        "action": "functional_panel",
        "selection_output": selection_output,
        "train_output": train_output,
        "r0140_panel": r0140_panel_signature,
        **common_panel,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["train_drop_only_historical_current_host"],
        "outputs": [functional_output],
        "done_marker": os.path.join(artifacts, "functional-panel.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 120.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "score_universality_drop_only",
        "action": "universality_panel",
        "map_key": TREATMENT,
        "train_output": train_output,
        **common_ood,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["score_functional_drop_only"],
        "outputs": [drop_universality],
        "done_marker": os.path.join(artifacts, "universality-treatment.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 60.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "decide_drop_only_decomposition",
        "action": "decide",
        "selection_output": selection_output,
        "functional_output": functional_output,
        "drop_universality_output": drop_universality,
        "r0147_decision": r0147_decision_signature,
        "r0147_functional_panel": r0147_panel_signature,
        "r0147_universality": r0147_universality,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["score_universality_drop_only"],
        "outputs": [decision_output],
        "done_marker": os.path.join(artifacts, "drop-only-decision.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 60.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
    }]

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_path,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0149-drop-only-decomposition-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": ["0147"],
        "capability_dependencies": [
            "jina-2m-historical-row-policy-duplicate-control-v1"
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            str(job["id"]): float(job["p90_wall_s"])
            for job in jobs
            if job["node_policy"]["gpu_required"]
        },
        "scientific_contract": {
            "question": (
                "does excluding the same 10,367 ineligible rows without "
                "replacement preserve historical-row functional restoration?"
            ),
            "accepted_activation": {
                "r0147_decision": r0147_decision_signature,
                "required_outcome": "eligible-historical-row-policy-does-not-restore",
            },
            "cells": {
                CURRENT_GRAPH_CURRENT_HOST: "accepted R0140 raw historical 2M control",
                SIZE_PRESERVING_TREATMENT: "accepted R0147 size-preserving negative",
                TREATMENT: {
                    "rows": ROWS,
                    "row_universe": ROW_UNIVERSE,
                    "excluded_rows": RAW_PREFIX_EXCLUDED_ROWS,
                    "replacement_rows": 0,
                    "graph": "current R0104 graph rebuilt on drop-only population",
                    "trainer": "current R0104 host weighted pipeline, seed 42",
                    "successful_updates": SUCCESSFUL_UPDATES,
                },
            },
            "selector": {
                "metrics": list(RESTORATION_FLOORS),
                "floors": RESTORATION_FLOORS,
                "all_metrics_required": True,
                "density_diagnostic_only": True,
            },
            "universality": {
                "maps": [
                    CURRENT_GRAPH_CURRENT_HOST,
                    SIZE_PRESERVING_TREATMENT,
                    TREATMENT,
                ],
                "role": "diagnostic only; never selector input",
            },
            "causal_scope": (
                "drop-only population/cardinality/induced-graph package; no "
                "unique exclusion, cardinality, replacement, or graph claim"
            ),
            "claims_excluded": [
                "duplicate or replacement causality",
                "diverse 25M transfer",
                "density floor change",
                "map registry or publication state change",
            ],
            "cpu_smoke": smoke_signature,
            "inherited_train_seal_panel_smoke": pipeline_smoke_signature,
            "release_pytest": pytest_signature,
        },
    })
    queue["p90_gpu_seconds"]["total"] = sum(queue["p90_gpu_seconds"].values())
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0149(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
