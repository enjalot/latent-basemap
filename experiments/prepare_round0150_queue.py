#!/usr/bin/env python3
"""Materialize, but never launch, the accepted-R0149 R0150 seed replay."""
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
from basemap.round0149_drop_only import (
    CAPABILITY as R0149_CAPABILITY,
    ROWS as DROP_ROWS,
    TREATMENT as DROP_ONLY,
)
from basemap.round0150_seed_replay import (
    CAPABILITY,
    ROUND_ID,
    SEED,
    drop_seed43_train_config,
    raw_seed43_train_config,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import (
    _accepted_review,
    _embedded_signatures,
    _frontmatter,
)
from experiments import prepare_round0147_queue as r0147_prep
from experiments.round0147_nodes import training_accounting_mismatches


ROUND_ROOT = "/data/latent-basemap/runs/round-0150"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0150-*.md")
R0140_GRAPH_ROOT = (
    "/data/latent-basemap/runs/round-0140/queue/artifacts/current-graph-fixed-rows"
)
R0134_PANEL = (
    "/data/latent-basemap/runs/round-0134/queue-attempt-3-exact-views/artifacts/"
    "functional-showdown/functional-showdown.json"
)
R0149_ROOT = "/data/latent-basemap/runs/round-0149/queue/artifacts"
R0149_DECISION = os.path.join(R0149_ROOT, R0149_CAPABILITY, "decision.json")
R0149_SELECTION_ROOT = os.path.join(R0149_ROOT, "drop-only-historical-selection")
R0149_GRAPH_ROOT = os.path.join(R0149_ROOT, "current-graph-drop-only-historical")

GPU_HOURS_MINIMUM = 2.35
GPU_HOURS_EXPECTED = 2.70
GPU_HOURS_P90 = 2.95
GPU_HOURS_MAXIMUM = 4.50


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
        raise RuntimeError(f"R0150 requires exactly one issued round; found {len(candidates)}")
    registered = str(_frontmatter(candidates[0]).get("base_commit") or "")
    if registered != release_sha:
        with open(candidates[0], encoding="utf-8") as handle:
            corrections = re.findall(
                r"Corrected execution release: `([0-9a-f]{40})`", handle.read()
            )
        if not corrections or corrections[-1] != release_sha:
            raise RuntimeError("R0150 issued release and correction addendum differ")
    return candidates[0], expected_input_signature(candidates[0])


def _accepted_activation() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    r0140_review = _accepted_review("0140", "jina-2m-subsystem-bisection-v1")
    r0149_review = _accepted_review("0149", R0149_CAPABILITY)
    decision, decision_signature = _read_sealed(
        R0149_DECISION, label="accepted R0149 decision"
    )
    selection, selection_signature = _read_sealed(
        os.path.join(R0149_SELECTION_ROOT, "selection-receipt.json"),
        label="accepted R0149 selection",
    )
    raw_graph, raw_graph_signature = _read_sealed(
        os.path.join(R0140_GRAPH_ROOT, "receipt.json"), label="accepted R0140 raw graph"
    )
    drop_graph, drop_graph_signature = _read_sealed(
        os.path.join(R0149_GRAPH_ROOT, "receipt.json"), label="accepted R0149 drop graph"
    )
    if (
        decision.get("round_id") != "0149"
        or decision.get("capability") != R0149_CAPABILITY
        or decision.get("outcome") != "drop-only-historical-row-policy-restores"
        or selection.get("round_id") != "0149"
        or selection.get("target_rows") != DROP_ROWS
        or selection.get("replacement_rows") != 0
        or raw_graph.get("round_id") != "0140"
        or drop_graph.get("round_id") != "0149"
        or drop_graph.get("source_proof", {}).get("rows") != DROP_ROWS
    ):
        raise RuntimeError("R0150 accepted activation changed")
    if decision.get("selection_receipt") != selection_signature:
        raise RuntimeError("R0149 decision/selection binding changed")
    return (
        r0140_review,
        r0149_review,
        decision_signature,
        selection_signature,
        raw_graph_signature,
        drop_graph_signature,
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
        raise RuntimeError("R0150 pytest checkout is not at the requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0150_seed_replay.py",
        "tests/test_round0149_drop_only.py",
        "tests/test_round0147_nodes.py",
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
        "schema": "round0150-release-pytest-v1",
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
        raise RuntimeError(f"R0150 release pytest failed:\n{completed.stdout}\n{completed.stderr}")
    return receipt


def _seed_cpu_smoke(
    *, raw_graph: Mapping[str, Any], drop_graph: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("R0150 CPU smoke requires CUDA_VISIBLE_DEVICES='' or '-1'")
    started = time.monotonic()
    raw_config, raw_digest = raw_seed43_train_config(
        graph_signature=raw_graph["graph"],
        graph_manifest_signature=raw_graph["graph_manifest"],
        graph_edges=int(raw_graph["graph_edges"]),
    )
    drop_config, drop_digest = drop_seed43_train_config(
        graph_signature=drop_graph["graph"],
        graph_manifest_signature=drop_graph["graph_manifest"],
        graph_edges=int(drop_graph["graph_edges"]),
        source_sha256=str(selection["staged_source"]["sha256"]),
        selection_sha256=str(selection["selection_arrays"]["sha256"]),
    )
    reports: dict[str, Any] = {}
    for key, config, graph_edges in (
        (CURRENT_GRAPH_CURRENT_HOST, raw_config, int(raw_graph["graph_edges"])),
        (DROP_ONLY, drop_config, int(drop_graph["graph_edges"])),
    ):
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
            "n_pos_edges": graph_edges,
        }
        mismatches = training_accounting_mismatches(
            accounting=accounting,
            runtime=runtime,
            expected_pipeline=config["execution"]["expected_pipeline_stamp"],
            graph_edges=graph_edges,
            batch_size=batch,
            profiler={"aborted": False},
            rate=TRAIN_MINIMUM_UPDATES_PER_S + 1.0,
        )
        if mismatches or config["paired_invariant"]["seed"] != SEED:
            raise RuntimeError(f"R0150 {key} CPU accounting smoke failed: {mismatches}")
        reports[key] = {
            "config_sha256": raw_digest if key == CURRENT_GRAPH_CURRENT_HOST else drop_digest,
            "rows": config["paired_invariant"]["rows"],
            "seed": config["paired_invariant"]["seed"],
            "row_universe": config["execution"]["expected_pipeline_stamp"]["row_universe"],
            "negative_sampling": config["execution"]["expected_pipeline_stamp"]["negative_sampling"],
            "accounting_mismatches": mismatches,
        }
    return seal({
        "schema": "round0150-paired-seed43-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": SEED,
        "cells": reports,
        "wall_seconds": time.monotonic() - started,
    })


def prepare_round0150(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0150 release SHA must be one full commit")
    round_path, round_signature = _issued_round(release_sha)
    (
        r0140_review,
        r0149_review,
        r0149_decision_signature,
        selection_signature,
        raw_graph_receipt_signature,
        drop_graph_receipt_signature,
    ) = _accepted_activation()
    r0149_decision, _ = _read_sealed(R0149_DECISION, label="accepted R0149 decision")
    selection, _ = _read_sealed(
        os.path.join(R0149_SELECTION_ROOT, "selection-receipt.json"),
        label="accepted R0149 selection",
    )
    raw_graph, _ = _read_sealed(
        os.path.join(R0140_GRAPH_ROOT, "receipt.json"), label="accepted R0140 raw graph"
    )
    drop_graph, _ = _read_sealed(
        os.path.join(R0149_GRAPH_ROOT, "receipt.json"), label="accepted R0149 drop graph"
    )
    r0140_panel, r0140_panel_signature = _read_sealed(
        r0147_prep.R0140_PANEL, label="accepted R0140 functional panel"
    )
    control_train, control_inputs = r0147_prep._r0140_control(
        r0140_panel["cells"][CURRENT_GRAPH_CURRENT_HOST]
    )
    shared, shared_inputs = r0147_prep._shared_reference()
    inventory, inventory_signature, excluded, inventory_inputs = r0147_prep._inventory_bundle()
    (
        common_outputs,
        ood_control,
        dadabase,
        dadabase_texts,
        beir,
        universality_inputs,
    ) = r0147_prep._universality_inputs()

    queue_root = create_fresh_directory(queue_root, label="R0150 seed replay queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    pytest_path = os.path.join(preflight, "release-pytest.json")
    atomic_write_new_json(pytest_path, _pytest_smoke(release_sha=release_sha), immutable=True)
    pytest_signature = expected_input_signature(pytest_path)
    seed_smoke_path = os.path.join(preflight, "paired-seed43-cpu-smoke.json")
    atomic_write_new_json(
        seed_smoke_path,
        _seed_cpu_smoke(raw_graph=raw_graph, drop_graph=drop_graph, selection=selection),
        immutable=True,
    )
    seed_smoke_signature = expected_input_signature(seed_smoke_path)
    pipeline_smoke_path = os.path.join(preflight, "inherited-pipeline-cpu-smoke.json")
    pipeline_smoke = r0147_prep._cpu_smoke(
        inventory=inventory,
        excluded=excluded,
        model_signature=control_train["model"],
        output_root=preflight,
    )
    atomic_write_new_json(pipeline_smoke_path, pipeline_smoke, immutable=True)
    pipeline_smoke_signature = expected_input_signature(pipeline_smoke_path)

    external_inputs = _dedupe([
        round_signature,
        *r0140_review,
        *r0149_review,
        r0149_decision_signature,
        *_embedded_signatures(r0149_decision),
        selection_signature,
        *_embedded_signatures(selection),
        raw_graph_receipt_signature,
        *_embedded_signatures(raw_graph),
        drop_graph_receipt_signature,
        *_embedded_signatures(drop_graph),
        r0140_panel_signature,
        expected_input_signature(R0134_PANEL),
        inventory_signature,
        *inventory_inputs,
        expected_input_signature(TRAIN_PATH),
        *control_inputs,
        *shared_inputs,
        *universality_inputs,
        *[expected_input_signature(item["path"]) for item in CENTROIDS.values()],
        pytest_signature,
        seed_smoke_signature,
        pipeline_smoke["published_checkpoint"],
        pipeline_smoke_signature,
    ])

    raw_train = os.path.join(artifacts, "raw_historical_seed43", "train")
    drop_train = os.path.join(artifacts, "drop_only_historical_seed43", "train")
    raw_panel = os.path.join(artifacts, "raw-seed43-functional-panel")
    paired_panel = os.path.join(artifacts, "paired-seed43-functional-panel")
    raw_ood = os.path.join(artifacts, f"universality-{CURRENT_GRAPH_CURRENT_HOST}")
    drop_ood = os.path.join(artifacts, f"universality-{DROP_ONLY}")
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
    module = "experiments.round0150_nodes"
    jobs: list[dict[str, Any]] = [{
        "id": "train_raw_historical_seed43",
        "action": "train_raw_seed43",
        "cell": CURRENT_GRAPH_CURRENT_HOST,
        "graph_kind": "current-fixed-row",
        "graph_output": R0140_GRAPH_ROOT,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [raw_train],
        "done_marker": os.path.join(artifacts, "train-raw-seed43.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 5_100.0,
        "node_policy": {"gpu_required": True, "training_performed": True},
    }, {
        "id": "train_drop_only_seed43",
        "action": "train_drop_seed43",
        "selection_output": R0149_SELECTION_ROOT,
        "graph_output": R0149_GRAPH_ROOT,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["train_raw_historical_seed43"],
        "outputs": [drop_train],
        "done_marker": os.path.join(artifacts, "train-drop-seed43.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 5_100.0,
        "node_policy": {"gpu_required": True, "training_performed": True},
    }, {
        "id": "score_raw_seed43_functional",
        "action": "score_raw_functional",
        "r0134_panel": expected_input_signature(R0134_PANEL),
        "train_outputs": {CURRENT_GRAPH_CURRENT_HOST: raw_train},
        "train_release_shas": {CURRENT_GRAPH_CURRENT_HOST: release_sha},
        **common_panel,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["train_raw_historical_seed43"],
        "outputs": [raw_panel],
        "done_marker": os.path.join(artifacts, "raw-functional.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 120.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "score_paired_seed43_functional",
        "action": "score_drop_functional",
        "selection_output": R0149_SELECTION_ROOT,
        "train_output": drop_train,
        "raw_panel_output": raw_panel,
        **common_panel,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["train_drop_only_seed43", "score_raw_seed43_functional"],
        "outputs": [paired_panel],
        "done_marker": os.path.join(artifacts, "paired-functional.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 120.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "score_raw_seed43_universality",
        "action": "score_raw_universality",
        "raw_train_output": raw_train,
        **common_ood,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["score_paired_seed43_functional"],
        "outputs": [raw_ood],
        "done_marker": os.path.join(artifacts, "raw-universality.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 60.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "score_drop_seed43_universality",
        "action": "score_drop_universality",
        "train_output": drop_train,
        **common_ood,
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["score_raw_seed43_universality"],
        "outputs": [drop_ood],
        "done_marker": os.path.join(artifacts, "drop-universality.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 60.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "decide_seed43_replay",
        "action": "decide_seed_replay",
        "r0149_decision": r0149_decision_signature,
        "raw_panel_output": raw_panel,
        "drop_panel_output": paired_panel,
        "universality_outputs": {
            CURRENT_GRAPH_CURRENT_HOST: raw_ood,
            DROP_ONLY: drop_ood,
        },
        "handler_module": module,
        "handler_callable": "run_job",
        "deps": ["score_drop_seed43_universality"],
        "outputs": [decision_output],
        "done_marker": os.path.join(artifacts, "decision.done.json"),
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
        "schema": "round0150-drop-only-seed-replay-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": ["0140", "0149"],
        "capability_dependencies": [
            "jina-2m-subsystem-bisection-v1",
            R0149_CAPABILITY,
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
            "question": "does raw-versus-drop-only restoration replicate at paired seed 43?",
            "accepted_activation": {
                "r0149_decision": r0149_decision_signature,
                "allowed_outcomes": ["drop-only-historical-row-policy-restores"],
            },
            "cells": {
                CURRENT_GRAPH_CURRENT_HOST: {
                    "rows": 2_000_000,
                    "graph": "byte-exact accepted R0140 current graph",
                    "seed": SEED,
                    "successful_updates": SUCCESSFUL_UPDATES,
                },
                DROP_ONLY: {
                    "rows": DROP_ROWS,
                    "graph": "byte-exact accepted R0149 drop-only graph",
                    "seed": SEED,
                    "successful_updates": SUCCESSFUL_UPDATES,
                },
            },
            "selector": {
                "metrics": list(RESTORATION_FLOORS),
                "floors": RESTORATION_FLOORS,
                "all_metrics_required_per_arm_and_seed": True,
                "scale_candidate_requires_raw_seed43_and_drop_only_both_seeds": True,
                "discordance": "inconclusive; no scale transfer",
                "density_diagnostic_only": True,
                "ood_diagnostic_only": True,
            },
            "claims_excluded": [
                "unique exclusion, cardinality, or graph causality",
                "25M transfer",
                "density floor change",
                "registry, production, or publishing state change",
            ],
            "paired_seed_cpu_smoke": seed_smoke_signature,
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
        "queue_manifest": prepare_round0150(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
