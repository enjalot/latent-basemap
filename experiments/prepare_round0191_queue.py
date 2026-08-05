#!/usr/bin/env python3
"""Prepare, but never launch, the R0191 full-rung h4096 width contrast."""
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
from basemap.round0191_full_width_contrast import (
    ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
    CAPABILITY,
    HIDDEN_DIMENSION,
    HOST_RSS_LIMIT_GIB,
    MINIMUM_TRAIN_UPDATES_PER_S,
    RETAINED_ROWS,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    TARGET_GRAPH_EDGES,
    WARNING_TRAIN_UPDATES_PER_S,
    h4096_train_config,
)
from basemap.round0184_prompted_8m_dose_midpoint import (
    scale_train_config as h2048_train_config,
)
from basemap.round0188_composition_boundary_seed43 import train_checks_close
from basemap import round0113_prompt_contrast as prompt_contract
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0184_queue import _accepted_terminal_review
from experiments.prepare_round0188_queue import (
    PILE_QUERY_RECEIPT,
    R0165_POPULATION,
    R0171_GRAPH,
    R0187_COMMON_GRAPH,
    R0187_QUARTER_EVALUATION,
    R0187_QUARTER_POPULATION,
    R0187_SHARED_TRUTH,
    _accepted_lineage,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0191"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0191-2026-08-05.md")
HANDLER_MODULE = "experiments.round0191_nodes"
QUEUE_SCHEMA = "round0191-full-h4096-width-contrast-queue-v1"
GPU_HOURS_CAP = 8.0

R0184_QUEUE = "/data/latent-basemap/runs/round-0184/queue/queue.json"
R0184_TERMINAL = "/data/latent-basemap/runs/round-0184/queue/runner-terminal.json"
R0184_TRAIN = (
    "/data/latent-basemap/runs/round-0184/queue/artifacts/"
    "seed42-1m-update-train/train-receipt.json"
)
R0190_QUEUE = "/data/latent-basemap/runs/round-0190/queue/queue.json"
R0190_TERMINAL = "/data/latent-basemap/runs/round-0190/queue/runner-terminal.json"
R0190_SYNTHESIS = (
    "/data/latent-basemap/runs/round-0190/queue/artifacts/"
    "jina-composition-boundary-three-seed-synthesis-v1/"
    "three-seed-boundary-synthesis.json"
)

P90 = {
    "train_full_h4096": 28_500.0,
    "evaluate_full_h4096": 75.0,
    "evaluate_r0184_h2048": 75.0,
    "synthesize_width_contrast": 30.0,
}


def _read_json(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0191 round is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _terminal_evidence(
    *, round_id: str, queue_path: str, terminal_path: str
) -> list[dict[str, Any]]:
    queue_signature = expected_input_signature(queue_path)
    terminal_signature = expected_input_signature(terminal_path)
    terminal = _read_json(terminal_path, label=f"R{round_id} terminal")
    if (
        terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("boundary_problems") != []
    ):
        raise RuntimeError(f"R{round_id} terminal premise changed")
    return [queue_signature, terminal_signature]


def _embedded_signatures(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        if {"canonical_path", "bytes", "sha256"}.issubset(value):
            actual = expected_input_signature(str(value["canonical_path"]))
            if actual != dict(value):
                raise RuntimeError(f"embedded input changed: {value['canonical_path']}")
            output.append(actual)
        else:
            for child in value.values():
                _embedded_signatures(child, output)
    elif isinstance(value, list):
        for child in value:
            _embedded_signatures(child, output)


def _accepted_inputs() -> list[dict[str, Any]]:
    signatures = [
        *_accepted_lineage(),
        *_accepted_terminal_review("0184"),
        *_accepted_terminal_review("0190"),
        *_terminal_evidence(
            round_id="0184", queue_path=R0184_QUEUE, terminal_path=R0184_TERMINAL
        ),
        *_terminal_evidence(
            round_id="0190", queue_path=R0190_QUEUE, terminal_path=R0190_TERMINAL
        ),
        expected_input_signature(R0184_TRAIN),
        expected_input_signature(R0190_SYNTHESIS),
    ]
    train = prompt_contract.read_sealed(R0184_TRAIN, label="accepted R0184 train")
    accounting = train.get("train_accounting") or {}
    if (
        train.get("schema")
        != "round0184-prompted-8m-dose-midpoint-train-receipt-v1"
        or train.get("round_id") != "0184"
        or int(train.get("training_seed", -1)) != 42
        or int(train.get("optimizer_updates", -1)) != SUCCESSFUL_UPDATES
        or not train_checks_close(train.get("train_checks"))
        or int(accounting.get("positive_lr_optimizer_steps", -1))
        != SUCCESSFUL_UPDATES
        or int(accounting.get("pipeline_endpoint_gather_calls", -1))
        != SUCCESSFUL_UPDATES
        or int(accounting.get("pipeline_source_rows_gathered", -1))
        != SUCCESSFUL_UPDATES * prompt_contract.BATCH_SIZE
        or int(accounting.get("pipeline_destination_rows_gathered", -1))
        != SUCCESSFUL_UPDATES * prompt_contract.BATCH_SIZE
    ):
        raise RuntimeError("accepted R0184 train receipt does not close")
    _embedded_signatures(train, signatures)

    track_a = prompt_contract.read_sealed(
        R0190_SYNTHESIS, label="accepted R0190 synthesis"
    )
    decision = track_a.get("decision") or {}
    if (
        track_a.get("schema") != "round0190-three-seed-boundary-synthesis-v1"
        or track_a.get("round_id") != "0190"
        or decision.get("outcome") != "confirmed-2-of-3-seed-sensitive"
        or decision.get("capacity_sibling_activated") is not True
        or int(decision.get("positive_seed_count", -1)) != 2
    ):
        raise RuntimeError("accepted R0190 synthesis did not activate R0191")
    _embedded_signatures(track_a, signatures)
    return _dedupe(signatures)


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0191 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0191_full_width_contrast.py",
        "tests/test_round0184_prompted_8m_dose_midpoint.py",
        "tests/test_round0188_composition_boundary_seed43.py",
        "tests/test_round0166_cpu_smoke.py",
        "tests/test_round0171_prompted_8m.py",
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
    receipt = prompt_contract.seal({
        "schema": "round0191-release-cpu-smoke-v1",
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
            "R0191 h4096 width config -> shared tiny fit -> exact accounting -> "
            "seal -> checkpoint reload -> transform -> downstream tiny panel"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0191 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _config_smoke() -> dict[str, Any]:
    graph_signature = expected_input_signature(R0171_GRAPH)
    graph = prompt_contract.read_sealed(
        R0171_GRAPH, label="accepted R0171 graph manifest"
    )
    kwargs = {
        "graph_signature": graph["graph"],
        "graph_manifest_signature": graph_signature,
        "graph_edges": int(graph["directed_edge_count"]),
        "retained_rows": int(graph["retained_rows"]),
    }
    h2048, h2048_digest = h2048_train_config(**kwargs)
    h4096, h4096_digest = h4096_train_config(**kwargs)
    r0184 = prompt_contract.read_sealed(R0184_TRAIN, label="accepted R0184 train")
    if (
        graph["directed_edge_count"] != TARGET_GRAPH_EDGES
        or graph["retained_rows"] != RETAINED_ROWS
        or h2048_digest != r0184.get("production_config_sha256")
        or h2048["model"]["hidden_dimension"] != 2048
        or h4096["model"]["hidden_dimension"] != HIDDEN_DIMENSION
        or h4096["optimizer"] != h2048["optimizer"]
        or h4096["graph"] != h2048["graph"]
        or h4096["input"] != h2048["input"]
        or h4096["optimizer"]["successful_positive_lr_updates"]
        != SUCCESSFUL_UPDATES
        or h4096["execution"]["minimum_train_upd_s"]
        != MINIMUM_TRAIN_UPDATES_PER_S
        or h4096["execution"]["warning_train_upd_s"]
        != WARNING_TRAIN_UPDATES_PER_S
    ):
        raise RuntimeError("R0191 config smoke changed")
    return prompt_contract.seal({
        "schema": "round0191-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "h2048_r0184_config_sha256": h2048_digest,
        "h4096_config_sha256": h4096_digest,
        "hidden_dimension_change": [2048, HIDDEN_DIMENSION],
        "successful_updates": SUCCESSFUL_UPDATES,
        "achieved_positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
        "minimum_train_updates_per_s": MINIMUM_TRAIN_UPDATES_PER_S,
        "host_rss_hard_abort_gib": HOST_RSS_LIMIT_GIB,
    })


def prepare_round0191(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0191 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    lineage = _accepted_inputs()
    population_signature = expected_input_signature(R0165_POPULATION)

    queue_root = create_fresh_directory(queue_root, label="R0191 width queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    release_smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        release_smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    config_smoke_path = os.path.join(preflight, "config-smoke.json")
    atomic_write_new_json(config_smoke_path, _config_smoke(), immutable=True)
    common = _dedupe([
        round_signature,
        *lineage,
        expected_input_signature(release_smoke_path),
        expected_input_signature(config_smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    train_output = os.path.join(artifacts, "full-h4096-train")
    evaluation_outputs = {
        "h4096": os.path.join(artifacts, "full-h4096-common-core-evaluation"),
        "r0184_h2048": os.path.join(
            artifacts, "r0184-h2048-common-core-evaluation"
        ),
    }
    evaluator_common = {
        "rung": "full",
        "population_receipt": population_signature,
        "graph_manifest": R0171_GRAPH,
        "common_population_receipt_path": R0187_QUARTER_POPULATION,
        "common_graph_manifest": R0187_COMMON_GRAPH,
        "pile_query_receipt": PILE_QUERY_RECEIPT,
        "r0187_quarter_evaluation": R0187_QUARTER_EVALUATION,
        "shared_truth_path": R0187_SHARED_TRUTH,
        "r0184_train_receipt": R0184_TRAIN,
    }
    jobs = [
        {
            "id": "train_full_h4096",
            "action": "train_full_h4096",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [train_output],
            "done_marker": os.path.join(artifacts, "full-h4096-train.done.json"),
            "expected_inputs": common,
            "p90_wall_s": P90["train_full_h4096"],
            "rung": "full",
            "population_receipt": population_signature,
            "graph_manifest": R0171_GRAPH,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        },
        {
            "id": "evaluate_full_h4096",
            "action": "evaluate_width_arm",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["train_full_h4096"],
            "outputs": [evaluation_outputs["h4096"]],
            "done_marker": os.path.join(artifacts, "full-h4096-evaluation.done.json"),
            "expected_inputs": common,
            "p90_wall_s": P90["evaluate_full_h4096"],
            **evaluator_common,
            "width_arm": "h4096",
            "train_output": train_output,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
        {
            "id": "evaluate_r0184_h2048",
            "action": "evaluate_width_arm",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["evaluate_full_h4096"],
            "outputs": [evaluation_outputs["r0184_h2048"]],
            "done_marker": os.path.join(artifacts, "r0184-h2048-evaluation.done.json"),
            "expected_inputs": common,
            "p90_wall_s": P90["evaluate_r0184_h2048"],
            **evaluator_common,
            "width_arm": "r0184_h2048",
            "train_output": train_output,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
        {
            "id": "synthesize_width_contrast",
            "action": "synthesize_width_contrast",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["evaluate_full_h4096", "evaluate_r0184_h2048"],
            "outputs": [os.path.join(artifacts, CAPABILITY)],
            "done_marker": os.path.join(artifacts, "width-synthesis.done.json"),
            "expected_inputs": common,
            "p90_wall_s": P90["synthesize_width_contrast"],
            "r0190_synthesis": R0190_SYNTHESIS,
            "evaluation_outputs": evaluation_outputs,
            "train_output": train_output,
            "r0184_train_receipt": R0184_TRAIN,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": QUEUE_SCHEMA,
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0165", "0171", "0184", "0187", "0190"],
        "capability_dependencies": [
            "jina-document-english-first8m-frozen-prefix-population-v1",
            "jina-document-english-composition-controlled-nested-ladder-v1",
            "jina-composition-boundary-three-seed-synthesis-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            **{
                job["id"]: float(job["p90_wall_s"])
                for job in jobs
                if job["node_policy"]["gpu_required"]
            },
            "total": sum(
                float(job["p90_wall_s"])
                for job in jobs
                if job["node_policy"]["gpu_required"]
            ),
        },
        "scientific_contract": {
            "question": (
                "does h4096 recover the full-rung Pile FFR boundary at the exact "
                "R0184 seed-42 1M-update dose?"
            ),
            "only_model_treatment_relative_to_r0184": "hidden dimension 2048 -> 4096",
            "population": expected_input_signature(R0165_POPULATION),
            "graph": expected_input_signature(R0171_GRAPH),
            "population_graph_reused_byte_exact": True,
            "training": {
                "seed": 42,
                "successful_updates": SUCCESSFUL_UPDATES,
                "positive_draws_per_edge": ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
                "hidden_dimension": HIDDEN_DIMENSION,
                "same_cosine_schedule_sampler_optimizer_precision_as_r0184": True,
                "minimum_updates_per_s": MINIMUM_TRAIN_UPDATES_PER_S,
                "warning_updates_per_s": WARNING_TRAIN_UPDATES_PER_S,
                "host_rss_hard_abort_gib": HOST_RSS_LIMIT_GIB,
            },
            "evaluation": {
                "common_core_source_round": "0187",
                "mixed_and_per_corpus_panels": True,
                "disjoint_pile_reserve": True,
                "density_v2_transcribed": True,
                "r0184_h2048_rescored_on_same_core": True,
            },
            "decision": {
                "registered_metric": "pile_ffr",
                "recovery": "h4096 >= 0.97 * R0190 seed42 half h2048",
                "null": (
                    "absolute h4096-minus-R0184 h2048 Pile FFR delta <= R0190 "
                    "three-seed full-rung sample SD"
                ),
                "recovery_and_null_reported_independently": True,
                "other_metrics": "diagnostic deltas",
            },
            "scope": "one h4096 sibling only; no width ladder",
            "release_cpu_smoke": expected_input_signature(release_smoke_path),
            "config_cpu_smoke": expected_input_signature(config_smoke_path),
        },
    })
    if queue["p90_gpu_seconds"]["total"] > GPU_HOURS_CAP * 3600:
        raise RuntimeError("R0191 P90 exceeds the 8 GPU-hour queue cap")
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0191(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
