#!/usr/bin/env python3
"""Prepare, but never launch, the R0158 drop-only seed-44/45 queue."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

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
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
)
from basemap.round0149_drop_only import ROWS, TREATMENT
from basemap.round0158_drop_seed_variance import (
    CAPABILITY,
    ROUND_ID,
    SEEDS,
    drop_seed_train_config,
)
from experiments import prepare_round0147_queue as r0147_prep
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _embedded_signatures, _frontmatter
from experiments.round0147_nodes import training_accounting_mismatches


ROUND_ROOT = "/data/latent-basemap/runs/round-0158"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0158-2026-08-02.md")
R0149_ROOT = "/data/latent-basemap/runs/round-0149/queue/artifacts"
R0149_SELECTION_ROOT = os.path.join(R0149_ROOT, "drop-only-historical-selection")
R0149_GRAPH_ROOT = os.path.join(R0149_ROOT, "current-graph-drop-only-historical")
R0140_PANEL = (
    "/data/latent-basemap/runs/round-0140/queue-attempt-2/artifacts/"
    "functional-panel/functional-bisection.json"
)
R0108_CALIBRATION = (
    "/data/latent-basemap/runs/round-0108/queue-attempt-3/artifacts/"
    "jina-density-calibration/jina-density-calibration.json"
)
REVIEW_FILES = (
    os.path.join(LAB_ROOT, "review-0108-2026-07-30.md"),
    os.path.join(LAB_ROOT, "review-0140-2026-08-01-01.md"),
    os.path.join(LAB_ROOT, "review-0149-2026-08-02.md"),
    os.path.join(LAB_ROOT, "review-0150-2026-08-02.md"),
)


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


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if frontmatter.get("status") != "issued":
        raise RuntimeError("R0158 round is not issued")
    if frontmatter.get("base_commit") != release_sha:
        raise RuntimeError("R0158 issued base_commit differs from release")
    return expected_input_signature(ROUND_FILE)


def _accepted_reviews() -> list[dict[str, Any]]:
    values = []
    for path in REVIEW_FILES:
        if _frontmatter(path).get("status") != "accepted":
            raise RuntimeError(f"R0158 required review is not accepted: {path}")
        values.append(expected_input_signature(path))
    return values


def _pytest_smoke(*, release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0158 pytest checkout is not at the requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0158_drop_seed_variance.py",
        "tests/test_round0154_seed_variance.py",
        "tests/test_round0153_density_forensics.py",
        "tests/test_round0150_seed_replay.py",
        "tests/test_round0149_drop_only.py",
        "tests/test_round0147_nodes.py",
        "tests/test_round0140_subsystem_bisection.py",
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
        "schema": "round0158-release-pytest-v1",
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
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0158 release pytest failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _seed_cpu_smoke(
    graph: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("R0158 CPU smoke requires CUDA hidden")
    reports: dict[str, Any] = {}
    for seed in SEEDS:
        config, digest = drop_seed_train_config(
            seed=seed,
            graph_signature=graph["graph"],
            graph_manifest_signature=graph["graph_manifest"],
            graph_edges=int(graph["graph_edges"]),
            source_sha256=str(selection["staged_source"]["sha256"]),
            selection_sha256=str(selection["selection_arrays"]["sha256"]),
        )
        batch = int(config["optimizer"]["batch_size"])
        expected_rows = SUCCESSFUL_UPDATES * batch
        runtime = {
            **config["execution"]["expected_pipeline_stamp"],
            "source_rows_gathered": expected_rows,
            "destination_rows_gathered": expected_rows,
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
            "n_pos_edges": int(graph["graph_edges"]),
        }
        mismatches = training_accounting_mismatches(
            accounting=accounting,
            runtime=runtime,
            expected_pipeline=config["execution"]["expected_pipeline_stamp"],
            graph_edges=int(graph["graph_edges"]),
            batch_size=batch,
            profiler={"aborted": False},
            rate=TRAIN_MINIMUM_UPDATES_PER_S + 1.0,
        )
        if mismatches or config["paired_invariant"]["seed"] != seed:
            raise RuntimeError(
                f"R0158 seed {seed} accounting smoke failed: {mismatches}"
            )
        reports[f"seed{seed}"] = {
            "config_sha256": digest,
            "seed": seed,
            "rows": config["paired_invariant"]["rows"],
            "pipeline": config["execution"]["expected_pipeline_stamp"],
            "accounting_mismatches": mismatches,
        }
    return seal({
        "schema": "round0158-drop-seed-config-accounting-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cells": reports,
    })


def prepare_round0158(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0158 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = _accepted_reviews()
    selection, selection_signature = _read_sealed(
        os.path.join(R0149_SELECTION_ROOT, "selection-receipt.json"),
        label="accepted R0149 selection",
    )
    graph, graph_signature = _read_sealed(
        os.path.join(R0149_GRAPH_ROOT, "receipt.json"),
        label="accepted R0149 drop-only graph",
    )
    if (
        selection.get("round_id") != "0149"
        or int(selection.get("target_rows", -1)) != ROWS
        or int(selection.get("replacement_rows", -1)) != 0
        or graph.get("round_id") != "0149"
        or int(graph.get("source_proof", {}).get("rows", -1)) != ROWS
    ):
        raise RuntimeError("R0158 accepted drop-only substrate changed")
    r0140_panel, r0140_panel_signature = _read_sealed(
        R0140_PANEL, label="accepted R0140 functional panel"
    )
    control = r0140_panel["cells"][CURRENT_GRAPH_CURRENT_HOST]
    control_train, control_inputs = r0147_prep._r0140_control(control)
    shared, shared_inputs = r0147_prep._shared_reference()
    inventory, inventory_signature, excluded, inventory_inputs = (
        r0147_prep._inventory_bundle()
    )
    calibration, calibration_signature = _read_sealed(
        R0108_CALIBRATION, label="accepted R0108 calibration"
    )

    queue_root = create_fresh_directory(queue_root, label="R0158 drop seed queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    pytest_path = os.path.join(preflight, "release-pytest.json")
    atomic_write_new_json(
        pytest_path, _pytest_smoke(release_sha=release_sha), immutable=True
    )
    seed_smoke_path = os.path.join(preflight, "seed-config-accounting-smoke.json")
    atomic_write_new_json(
        seed_smoke_path, _seed_cpu_smoke(graph, selection), immutable=True
    )
    pipeline_smoke = r0147_prep._cpu_smoke(
        inventory=inventory,
        excluded=excluded,
        model_signature=control_train["model"],
        output_root=preflight,
    )
    pipeline_smoke_path = os.path.join(preflight, "inherited-pipeline-cpu-smoke.json")
    atomic_write_new_json(pipeline_smoke_path, pipeline_smoke, immutable=True)

    inherited_calibration = [
        dict(calibration[key])
        for key in ("census_receipt", "census", "representative_reference", "arrays")
    ]
    external_inputs = _dedupe([
        round_signature,
        *reviews,
        selection_signature,
        *_embedded_signatures(selection),
        graph_signature,
        *_embedded_signatures(graph),
        r0140_panel_signature,
        calibration_signature,
        *inherited_calibration,
        inventory_signature,
        *inventory_inputs,
        expected_input_signature(TRAIN_PATH),
        *control_inputs,
        *shared_inputs,
        *[expected_input_signature(item["path"]) for item in CENTROIDS.values()],
        expected_input_signature(pytest_path),
        expected_input_signature(seed_smoke_path),
        pipeline_smoke["published_checkpoint"],
        expected_input_signature(pipeline_smoke_path),
    ])
    common_panel = {
        "r0140_panel": r0140_panel_signature,
        "selection_output": R0149_SELECTION_ROOT,
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
    train_outputs = {
        seed: os.path.join(artifacts, f"drop_only_historical_seed{seed}", "train")
        for seed in SEEDS
    }
    panel_outputs = {
        seed: os.path.join(artifacts, f"drop-seed{seed}-functional-panel")
        for seed in SEEDS
    }
    module = "experiments.round0158_nodes"
    jobs: list[dict[str, Any]] = []
    prior: str | None = None
    for seed in SEEDS:
        train_id = f"train_drop_only_seed{seed}"
        panel_id = f"score_drop_seed{seed}_functional"
        jobs.append({
            "id": train_id,
            "action": "train_drop_seed",
            "seed": seed,
            "selection_output": R0149_SELECTION_ROOT,
            "graph_output": R0149_GRAPH_ROOT,
            "handler_module": module,
            "handler_callable": "run_job",
            "deps": [] if prior is None else [prior],
            "outputs": [train_outputs[seed]],
            "done_marker": os.path.join(artifacts, f"train-drop-seed{seed}.done.json"),
            "expected_inputs": external_inputs,
            "p90_wall_s": 5_100.0,
            "node_policy": {"gpu_required": True, "training_performed": True},
        })
        jobs.append({
            "id": panel_id,
            "action": "score_drop_functional",
            "seed": seed,
            "train_output": train_outputs[seed],
            **common_panel,
            "handler_module": module,
            "handler_callable": "run_job",
            "deps": [train_id],
            "outputs": [panel_outputs[seed]],
            "done_marker": os.path.join(artifacts, f"drop-seed{seed}-functional.done.json"),
            "expected_inputs": external_inputs,
            "p90_wall_s": 180.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
        })
        prior = panel_id

    density_output = os.path.join(artifacts, "drop-seed-density-v2-panel")
    jobs.append({
        "id": "score_drop_seed_density_v2",
        "action": "score_drop_density_v2",
        "functional_panels": {str(seed): panel_outputs[seed] for seed in SEEDS},
        "r0108_calibration": calibration_signature,
        "cpu_workers": 4,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": [f"score_drop_seed{seed}_functional" for seed in SEEDS],
        "outputs": [density_output],
        "done_marker": os.path.join(artifacts, "drop-seed-density.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 180.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
    })
    decision_output = os.path.join(artifacts, CAPABILITY)
    jobs.append({
        "id": "seal_drop_seed_evidence",
        "action": "seal_drop_seed_evidence",
        "functional_panels": {str(seed): panel_outputs[seed] for seed in SEEDS},
        "density_output": density_output,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["score_drop_seed_density_v2"],
        "outputs": [decision_output],
        "done_marker": os.path.join(artifacts, "seed-evidence.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 30.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
    })

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=4.25,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0158-drop-seed44-45-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": ["0108", "0140", "0149", "0150"],
        "capability_dependencies": [
            "jina-density-v2-calibration-v1",
            "jina-2m-subsystem-bisection-v1",
            "jina-2m-historical-drop-only-decomposition-v1",
            "jina-2m-drop-only-seed-replication-v1",
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
            "question": "what is drop-only historical-row seed variation at seeds 44 and 45?",
            "seeds": list(SEEDS),
            "rows": ROWS,
            "selection": dict(selection["selection_arrays"]),
            "graph": dict(graph["graph"]),
            "graph_manifest": dict(graph["graph_manifest"]),
            "graph_reused_byte_exact": True,
            "graph_builds": 0,
            "successful_updates_per_seed": SUCCESSFUL_UPDATES,
            "full_functional_panel_per_seed": True,
            "density_v2_per_seed": True,
            "density_v2_floor": 0.17589389755990817,
            "density_diagnostic_only": True,
            "margin_or_floor_proposal_deferred_to_g3": True,
            "floor_changed": False,
            "release_pytest": expected_input_signature(pytest_path),
            "seed_config_accounting_smoke": expected_input_signature(seed_smoke_path),
            "inherited_train_seal_panel_smoke": expected_input_signature(
                pipeline_smoke_path
            ),
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
        "queue_manifest": prepare_round0158(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

