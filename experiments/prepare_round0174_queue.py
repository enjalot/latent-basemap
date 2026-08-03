#!/usr/bin/env python3
"""Prepare, but never launch, the R0174 historical-row fuzzy-k15 forensic."""
from __future__ import annotations

import argparse
import glob
import json
import os
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
from basemap.round0174_k15_forensic import CAPABILITY, CELL, GRAPH_K, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import (
    _accepted_review,
    _embedded_signatures,
    _frontmatter,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0174"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0174-2026-08-03.md")
SMOKE_PATH = os.path.join(ROUND_ROOT, "preflight", "release-cpu-smoke.json")
CORRECTION_SMOKE_PATH = os.path.join(
    ROUND_ROOT, "preflight", "release-cpu-smoke-correction-1.json"
)
FIRST_QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
FIRST_QUEUE_MANIFEST = os.path.join(FIRST_QUEUE_ROOT, "queue.json")
FIRST_TERMINAL = os.path.join(FIRST_QUEUE_ROOT, "runner-terminal.json")
FIRST_GRAPH_DONE = os.path.join(
    FIRST_QUEUE_ROOT, "artifacts", "build-current-k15-graph.done.json"
)
FIRST_TRAIN_FAILED = os.path.join(
    FIRST_QUEUE_ROOT, "artifacts", "train-k15-current-host.failed.json"
)
FIRST_GRAPH_OUTPUT = os.path.join(
    FIRST_QUEUE_ROOT, "artifacts", "current-k15-graph-fixed-rows"
)
R0037_SHARED = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/"
    "shared-reference/receipt.json"
)
R0134_PANEL = (
    "/data/latent-basemap/runs/round-0134/queue-attempt-3-exact-views/artifacts/"
    "functional-showdown/functional-showdown.json"
)
R0140_PANEL = (
    "/data/latent-basemap/runs/round-0140/queue-attempt-2/artifacts/"
    "functional-panel/functional-bisection.json"
)
R0171_EVALUATION = (
    "/data/latent-basemap/runs/round-0171/queue/artifacts/"
    "jina-document-english-8m-prompted-map-seed42-sharded-fp32-ivf-v1/"
    "scale-evaluation.json"
)
GPU_HOURS_MAXIMUM = 2.5


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0174 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0174_k15_forensic.py",
        "tests/test_round0140_subsystem_bisection.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
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
        "schema": "round0174-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": (
            "k15 config and runtime stamps -> train accounting -> seal -> "
            "checkpoint reload -> transform -> tiny panel -> selector"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0174 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def write_release_smoke(release_sha: str, *, path: str = SMOKE_PATH) -> str:
    preflight = ensure_data_directory(os.path.dirname(path))
    path = os.path.join(preflight, os.path.basename(path))
    atomic_write_new_json(path, _release_cpu_smoke(release_sha), immutable=True)
    return path


def _issued_round(release_sha: str) -> dict[str, Any]:
    if not os.path.isfile(ROUND_FILE):
        raise RuntimeError("R0174 issued round file is absent")
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0174 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def _accepted_negative_review_0171() -> list[dict[str, Any]]:
    matches: list[list[dict[str, Any]]] = []
    for path in sorted(glob.glob(os.path.join(LAB_ROOT, "review-0171-*.md"))):
        frontmatter = _frontmatter(path)
        if frontmatter.get("round_id") != "0171" or frontmatter.get("status") != "accepted":
            continue
        result_path = os.path.join(LAB_ROOT, frontmatter.get("result") or "")
        round_path = os.path.join(LAB_ROOT, frontmatter.get("round") or "")
        result_sig = expected_input_signature(result_path)
        round_sig = expected_input_signature(round_path)
        if (
            result_sig["sha256"] != frontmatter.get("result_sha256")
            or round_sig["sha256"] != frontmatter.get("round_sha256")
        ):
            raise RuntimeError("Review 0171 binding changed")
        result_frontmatter = _frontmatter(result_path)
        if result_frontmatter.get("outcome") != "prompted-english-8m-fixed-dose-not-qualified":
            raise RuntimeError("R0171 negative outcome changed")
        matches.append([round_sig, result_sig, expected_input_signature(path)])
    if len(matches) != 1:
        raise RuntimeError(
            "R0174 requires one accepted negative Review 0171; "
            f"found {len(matches)}"
        )
    evaluation_sig = expected_input_signature(R0171_EVALUATION)
    evaluation = _read_json(R0171_EVALUATION)
    validate_seal(evaluation, label="R0171 negative scale evaluation")
    if (
        evaluation.get("round_id") != "0171"
        or evaluation.get("capabilities") != []
        or (evaluation.get("decision") or {}).get("passed") is not False
    ):
        raise RuntimeError("R0171 is not the registered negative Q2 result")
    return [*matches[0], evaluation_sig]


def _review_inputs() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for round_id, capability in (
        ("0037", "jina-mrl-seed42-screen-v1"),
        ("0134", "jina-density-functional-showdown-v1"),
        ("0140", "jina-2m-subsystem-bisection-v1"),
    ):
        output.extend(_accepted_review(round_id, capability))
    output.extend(_accepted_negative_review_0171())
    return output


def _validated_smoke(
    release_sha: str, *, path: str = SMOKE_PATH
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    smoke = _read_json(path)
    validate_seal(smoke, label="R0174 release CPU smoke")
    if (
        smoke.get("schema") != "round0174-release-cpu-smoke-v1"
        or smoke.get("round_id") != ROUND_ID
        or smoke.get("release_sha") != release_sha
        or smoke.get("cuda_visible_devices") != ""
        or smoke.get("returncode") != 0
        or float(smoke.get("wall_seconds", 999.0)) >= 120.0
    ):
        raise RuntimeError("R0174 release CPU smoke is invalid")
    return signature


def _correction_inputs() -> tuple[list[dict[str, Any]], float]:
    signatures = [
        expected_input_signature(FIRST_QUEUE_MANIFEST),
        expected_input_signature(FIRST_TERMINAL),
        expected_input_signature(FIRST_GRAPH_DONE),
        expected_input_signature(FIRST_TRAIN_FAILED),
        expected_input_signature(
            os.path.join(FIRST_GRAPH_OUTPUT, "edges-k15-fuzzy.npz")
        ),
        expected_input_signature(
            os.path.join(FIRST_GRAPH_OUTPUT, "graph-manifest.json")
        ),
    ]
    terminal = _read_json(FIRST_TERMINAL)
    failed = _read_json(FIRST_TRAIN_FAILED)
    graph_manifest = _read_json(
        os.path.join(FIRST_GRAPH_OUTPUT, "graph-manifest.json")
    )
    if (
        terminal.get("round_id") != ROUND_ID
        or terminal.get("verdict") != "failed"
        or terminal.get("release_checkout", {}).get("head")
        != "a80259582bdfb762c7561f2bd7c44b180840dfe9"
        or terminal.get("completed_jobs") != ["build_current_k15_graph_fixed_rows"]
        or failed.get("node")
        != "train_historical_rows_current_graph_k15_current_host"
        or "actual loader plan cannot supply" not in str(failed.get("log_tail"))
        or graph_manifest.get("k") != GRAPH_K
        or graph_manifest.get("n_nodes") != 2_000_000
        or graph_manifest.get("n_edges") != 43_848_884
        or graph_manifest.get("graph_sha256") != signatures[4]["sha256"]
    ):
        raise RuntimeError("R0174 first-attempt correction evidence changed")
    prior_gpu_wall_s = float(terminal.get("gpu_wall_s", -1.0))
    if prior_gpu_wall_s <= 0 or prior_gpu_wall_s >= GPU_HOURS_MAXIMUM * 3_600:
        raise RuntimeError("R0174 first-attempt GPU accounting is invalid")
    return signatures, prior_gpu_wall_s


def prepare_round0174(
    *,
    release_sha: str,
    queue_root: str | None = None,
    correction: bool = False,
) -> str:
    if queue_root is None:
        queue_root = os.path.join(
            ROUND_ROOT, "queue-attempt-2" if correction else "queue"
        )
    round_signature = _issued_round(release_sha)
    review_inputs = _review_inputs()
    smoke_signature = _validated_smoke(
        release_sha,
        path=CORRECTION_SMOKE_PATH if correction else SMOKE_PATH,
    )
    correction_inputs: list[dict[str, Any]] = []
    prior_gpu_wall_s = 0.0
    if correction:
        correction_inputs, prior_gpu_wall_s = _correction_inputs()

    r0134_signature = expected_input_signature(R0134_PANEL)
    r0134 = _read_json(R0134_PANEL)
    validate_seal(r0134, label="R0134 functional panel")
    if (
        r0134.get("round_id") != "0134"
        or r0134.get("source") != expected_input_signature(TRAIN_PATH)
        or not isinstance(r0134.get("cells"), Mapping)
    ):
        raise RuntimeError("R0134 functional context changed")
    r0140_signature = expected_input_signature(R0140_PANEL)
    r0140 = _read_json(R0140_PANEL)
    validate_seal(r0140, label="accepted R0140 functional panel")
    if (
        r0140.get("round_id") != "0140"
        or "current_graph_current_host" not in (r0140.get("cells") or {})
    ):
        raise RuntimeError("R0140 k50 control changed")
    shared_signature = expected_input_signature(R0037_SHARED)
    shared = _read_json(R0037_SHARED)
    validate_seal(shared, label="R0037 shared reference")
    for key in ("high_d_reference", "query_truth", "query_embeddings"):
        if expected_input_signature(shared[key]["canonical_path"]) != shared[key]:
            raise RuntimeError(f"R0037 shared {key} changed")

    queue_root = create_fresh_directory(queue_root, label="R0174 k15 forensic queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe([
        round_signature,
        *review_inputs,
        smoke_signature,
        *correction_inputs,
        r0134_signature,
        r0140_signature,
        shared_signature,
        expected_input_signature(TRAIN_PATH),
        *[expected_input_signature(item["path"]) for item in CENTROIDS.values()],
        *[dict(shared[key]) for key in ("high_d_reference", "query_truth", "query_embeddings")],
        *_embedded_signatures(r0134),
    ])

    graph_output = (
        FIRST_GRAPH_OUTPUT
        if correction
        else os.path.join(artifacts, "current-k15-graph-fixed-rows")
    )
    train_output = os.path.join(artifacts, CELL, "train")
    panel_output = os.path.join(artifacts, "functional-panel")
    decision_output = os.path.join(artifacts, CAPABILITY)
    common_panel = {
        "source": expected_input_signature(TRAIN_PATH),
        "shared_reference_receipt": shared_signature,
        "high_d_reference": dict(shared["high_d_reference"]),
        "query_truth": dict(shared["query_truth"]),
        "query_embeddings": dict(shared["query_embeddings"]),
        "centroids": {
            str(k): expected_input_signature(value["path"])
            for k, value in CENTROIDS.items()
        },
        "r0134_panel": r0134_signature,
    }
    jobs: list[dict[str, Any]] = [{
        "id": "build_current_k15_graph_fixed_rows",
        "action": "build_current_graph",
        "handler_module": "experiments.round0174_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [graph_output],
        "done_marker": os.path.join(artifacts, "build-current-k15-graph.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 600.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "train_historical_rows_current_graph_k15_current_host",
        "action": "train_host",
        "cell": CELL,
        "graph_kind": "current-fixed-row",
        "graph_output": graph_output,
        "handler_module": "experiments.round0174_nodes",
        "handler_callable": "run_job",
        "deps": (
            [] if correction else ["build_current_k15_graph_fixed_rows"]
        ),
        "outputs": [train_output],
        "done_marker": os.path.join(artifacts, "train-k15-current-host.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 5_400.0,
        "node_policy": {"gpu_required": True, "training_performed": True},
    }, {
        "id": "score_k15_functional_panel",
        "action": "functional_panel",
        "handler_module": "experiments.round0174_nodes",
        "handler_callable": "run_job",
        "deps": ["train_historical_rows_current_graph_k15_current_host"],
        "train_outputs": {CELL: train_output},
        "train_release_shas": {CELL: release_sha},
        **common_panel,
        "outputs": [panel_output],
        "done_marker": os.path.join(artifacts, "score-k15-functional.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 300.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "decide_k15_forensic",
        "action": "decide",
        "handler_module": "experiments.round0174_nodes",
        "handler_callable": "run_job",
        "deps": ["score_k15_functional_panel"],
        "panel_output": panel_output,
        "r0140_panel": R0140_PANEL,
        "outputs": [decision_output],
        "done_marker": os.path.join(artifacts, "decide-k15-forensic.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 60.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
    }]
    if correction:
        jobs = jobs[1:]
    remaining_gpu_hours = GPU_HOURS_MAXIMUM - prior_gpu_wall_s / 3_600
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=remaining_gpu_hours,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": (
            "round0174-historical-row-k15-forensic-correction-queue-v1"
            if correction
            else "round0174-historical-row-k15-forensic-queue-v1"
        ),
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0037", "0134", "0140", "0171"],
        "capability_dependencies": [
            "jina-mrl-seed42-screen-v1",
            "jina-density-functional-showdown-v1",
            "jina-2m-subsystem-bisection-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            job["id"]: float(job["p90_wall_s"])
            for job in jobs if job["node_policy"]["gpu_required"]
        } | {"total": sum(
            float(job["p90_wall_s"])
            for job in jobs if job["node_policy"]["gpu_required"]
        )},
        "scientific_contract": {
            "question": "does fuzzy graph k15 alone break the R0140 historical-row restoration?",
            "cell": CELL,
            "paired_control": "accepted R0140 current_graph_current_host",
            "only_treatment": "fuzzy graph k 50 -> 15",
            "row_universe": "exact R0037 2M ordered rows",
            "graph_builder": "current GPU IVF-Flat/IP plus fuzzy_simplicial_set",
            "control_graph_k": 50,
            "treatment_graph_k": GRAPH_K,
            "trainer": "current R0104 host weighted sampler/runtime",
            "seed": 42,
            "successful_updates": 500_000,
            "restoration_floors": "accepted R0140 frozen five-metric stack",
            "density_diagnostic_only": True,
            "fixed_dose_estimand": True,
            "release_cpu_smoke": smoke_signature,
            "negative_q2_activation": expected_input_signature(R0171_EVALUATION),
            "correction": (
                {
                    "class": "setup-only-loader-loop-capacity",
                    "first_attempt_terminal": expected_input_signature(
                        FIRST_TERMINAL
                    ),
                    "reused_graph_output": FIRST_GRAPH_OUTPUT,
                    "prior_gpu_wall_s": prior_gpu_wall_s,
                    "cumulative_gpu_hard_cap": GPU_HOURS_MAXIMUM,
                    "remaining_gpu_hours_cap": remaining_gpu_hours,
                    "science_changed": False,
                }
                if correction
                else None
            ),
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--correction", action="store_true")
    args = parser.parse_args(argv)
    path = (
        write_release_smoke(
            args.release_sha,
            path=CORRECTION_SMOKE_PATH if args.correction else SMOKE_PATH,
        )
        if args.smoke_only
        else prepare_round0174(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
            correction=args.correction,
        )
    )
    print(json.dumps({"path": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
