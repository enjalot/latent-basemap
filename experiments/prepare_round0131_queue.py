#!/usr/bin/env python3
"""Prepare, but never launch, conditional R0131 component localization."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0125_runtime_bridge import environment_freeze_receipt, validate_seal
from basemap.round0131_runtime_factorial import (
    ARMS,
    CAPABILITY,
    PIPELINES,
    POSITIVE_R0125_OUTCOMES,
    ROUND_ID,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0119_queue import _clean_terminal, _document, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0131"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
RUN_ENVIRONMENT_PREFIX = os.path.join(RELEASE_ROOT, ".venv")
RUN_PYTHON = os.path.join(RUN_ENVIRONMENT_PREFIX, "bin", "python")
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0131-*.md")
R0125_RELEASE_SHA = "ff5dfcde5632257aac355008a70bc330bab26bee"
R0125_CAPABILITY = "jina-fineweb-2m-runtime-path-density-bridge-v1"


def _require_dedicated_run_environment() -> None:
    if (
        os.path.abspath(sys.executable) != RUN_PYTHON
        or os.path.abspath(sys.prefix) != RUN_ENVIRONMENT_PREFIX
    ):
        raise RuntimeError(
            "R0131 queue preparation must use the dedicated run environment "
            f"{RUN_PYTHON}"
        )


def _issued_round() -> str:
    candidates = []
    for path in sorted(glob.glob(ROUND_FILE_GLOB)):
        frontmatter, _ = _document(path)
        if frontmatter.get("status") == "issued":
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0131 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _job(queue: Mapping[str, Any], *, action: str) -> dict[str, Any]:
    jobs = [value for value in queue.get("jobs") or [] if value.get("action") == action]
    if len(jobs) != 1:
        raise RuntimeError(f"R0125 queue lacks exactly one {action!r} job")
    return dict(jobs[0])


def _read_json(path: str, *, label: str, sealed: bool = False) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if sealed:
        validate_seal(value, label=label)
    return value


def _accepted_r0125(review_path: str) -> dict[str, Any]:
    review_signature = expected_input_signature(review_path)
    review, review_text = _document(review_path)
    result_name = str(review.get("result") or "")
    if (
        review.get("round_id") != "0125"
        or review.get("status") != "accepted"
        or review.get("verified_release_commit") != R0125_RELEASE_SHA
        or f"capability:{R0125_CAPABILITY}"
        not in _frontmatter_list(review, "releases", label="R0125 review")
        or os.path.basename(result_name) != result_name
        or re.fullmatch(r"result-0125-[0-9]{4}-[0-9]{2}-[0-9]{2}\.md", result_name)
        is None
    ):
        raise RuntimeError("R0125 review is not accepted and exact")
    result_path = os.path.join(os.path.dirname(review_path), result_name)
    result_signature = expected_input_signature(result_path)
    result, result_text = _document(result_path)
    queue_field = str(result.get("queue_manifest") or "")
    if not queue_field.startswith("gsv:/"):
        raise RuntimeError("R0125 result lacks a canonical queue path")
    queue_path = queue_field.removeprefix("gsv:")
    queue_signature = expected_input_signature(queue_path)
    queue = _read_json(queue_path, label="R0125 queue")
    terminal_path = os.path.join(os.path.dirname(queue_path), "runner-terminal.json")
    terminal = _clean_terminal(
        queue_path,
        terminal_path,
        round_id="0125",
        expected_release_sha=R0125_RELEASE_SHA,
    )
    if (
        result.get("round_id") != "0125"
        or result.get("status") != "complete"
        or result.get("release_commit") != R0125_RELEASE_SHA
        or review.get("result_sha256") != result_signature["sha256"]
        or result.get("queue_manifest_sha256") != queue_signature["sha256"]
        or R0125_CAPABILITY
        not in _frontmatter_list(
            result, "capabilities_produced", label="R0125 result"
        )
        or queue.get("round_id") != "0125"
        or queue.get("release_sha") != R0125_RELEASE_SHA
    ):
        raise RuntimeError("R0125 review/result/queue closure changed")
    decision_job = _job(queue, action="decide")
    panel_job = _job(queue, action="matched_density")
    decision_path = os.path.join(decision_job["outputs"][0], "decision.json")
    panel_path = os.path.join(panel_job["outputs"][0], "matched-density-panel.json")
    decision_signature = expected_input_signature(decision_path)
    panel_signature = expected_input_signature(panel_path)
    decision = _read_json(decision_path, label="R0125 decision", sealed=True)
    panel = _read_json(panel_path, label="R0125 panel", sealed=True)
    arrays_signature = dict(panel.get("arrays") or {})
    evidence_text = review_text + "\n" + result_text
    if (
        decision.get("schema") != "round0125-device-host-runtime-decision-v1"
        or decision.get("outcome") not in POSITIVE_R0125_OUTCOMES
        or (decision.get("selector") or {}).get("execution_valid") is not True
        or decision.get("capabilities_produced") != [R0125_CAPABILITY]
        or panel.get("schema") != "round0125-matched-runtime-density-panel-v1"
        or decision.get("matched_density_panel") != panel_signature
        or expected_input_signature(arrays_signature.get("canonical_path", ""))
        != arrays_signature
        or decision_signature["sha256"] not in evidence_text
        or panel_signature["sha256"] not in evidence_text
    ):
        raise RuntimeError("R0125 sealed positive evidence is not review-bound")
    inherited_inputs = []
    for signature in (queue.get("jobs") or [])[0].get("expected_inputs") or []:
        if not isinstance(signature, Mapping):
            raise RuntimeError("R0125 expected input is malformed")
        path = str(signature.get("canonical_path") or "")
        if not os.path.exists(path) or os.path.getsize(path) != signature.get("bytes"):
            raise RuntimeError("R0125 inherited input is missing/wrong size")
        inherited_inputs.append(dict(signature))
    return {
        "review": review_signature,
        "result": result_signature,
        "queue": queue_signature,
        "terminal": terminal["terminal"],
        "decision": decision_signature,
        "panel": panel_signature,
        "arrays": arrays_signature,
        "outcome": decision["outcome"],
        "shared_output": str((queue.get("jobs") or [])[0].get("shared_output")),
        "calibration": dict(panel_job.get("r0108_calibration") or {}),
        "inherited_inputs": inherited_inputs,
    }


def _smoke_receipt(path: str, expected_sha256: str, *, release_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signature = expected_input_signature(path)
    receipt = _read_json(path, label="R0131 CPU smoke", sealed=True)
    if (
        signature["sha256"] != expected_sha256
        or receipt.get("schema") != "round0131-cpu-preflight-v1"
        or receipt.get("release_sha") != release_sha
        or receipt.get("cuda_visible_devices") != ""
        or receipt.get("outcome") != "passed"
        or not all((receipt.get("checks") or {}).values())
    ):
        raise RuntimeError("R0131 CPU smoke did not pass on this release")
    sources = receipt.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("R0131 CPU smoke lacks source bindings")
    for source in sources:
        if expected_input_signature(source.get("canonical_path", "")) != source:
            raise RuntimeError("R0131 CPU smoke source changed")
    return signature, [dict(value) for value in sources]


def prepare_round0131(
    *,
    release_sha: str,
    r0125_review: str,
    cpu_smoke_receipt: str,
    cpu_smoke_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", release_sha) is None:
        raise ValueError("R0131 release SHA must be one full commit")
    _require_dedicated_run_environment()
    round_file = _issued_round()
    evidence = _accepted_r0125(r0125_review)
    smoke, smoke_sources = _smoke_receipt(
        cpu_smoke_receipt, cpu_smoke_sha256, release_sha=release_sha
    )
    if expected_input_signature(evidence["calibration"]["canonical_path"]) != evidence["calibration"]:
        raise RuntimeError("R0125-bound calibration bytes changed")
    common_inputs = _dedupe([
        expected_input_signature(round_file),
        smoke,
        *smoke_sources,
        evidence["review"],
        evidence["result"],
        evidence["queue"],
        evidence["terminal"],
        evidence["decision"],
        evidence["panel"],
        evidence["arrays"],
        evidence["calibration"],
        *evidence["inherited_inputs"],
    ])
    environment = environment_freeze_receipt()
    queue_root = create_fresh_directory(queue_root, label="R0131 component queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    train_outputs = {
        arm: os.path.join(artifacts, arm, "train") for arm in ARMS
    }
    panel_output = os.path.join(artifacts, "runtime-component-panel")
    decision_output = os.path.join(artifacts, "decision")

    def job(
        job_id: str,
        action: str,
        deps: list[str],
        output: str,
        p90_wall_s: float,
        *,
        gpu_required: bool = True,
        arm: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        value = {
            "id": job_id,
            "action": action,
            "handler_module": "experiments.round0131_nodes",
            "handler_callable": "run_job",
            "deps": deps,
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{job_id}.done.json"),
            "expected_inputs": common_inputs,
            "p90_wall_s": p90_wall_s,
            "node_policy": {
                "gpu_required": gpu_required,
                "training_performed": action == "train",
            },
            "shared_output": evidence["shared_output"],
            "r0125_decision": evidence["decision"],
            **extra,
        }
        if arm is not None:
            value["arm"] = arm
        return value

    jobs = [
        job(
            "train_numpy_device_fused",
            "train",
            [],
            train_outputs[ARMS[0]],
            5_200.0,
            arm=ARMS[0],
        ),
        job(
            "train_numpy_device_separate",
            "train",
            ["train_numpy_device_fused"],
            train_outputs[ARMS[1]],
            5_200.0,
            arm=ARMS[1],
        ),
        job(
            "score_runtime_components",
            "panel",
            ["train_numpy_device_separate"],
            panel_output,
            360.0,
            train_outputs=train_outputs,
            r0125_panel=evidence["panel"],
            r0108_calibration=evidence["calibration"],
        ),
        job(
            "decide_runtime_components",
            "decide",
            ["score_runtime_components"],
            decision_output,
            30.0,
            gpu_required=False,
            train_outputs=train_outputs,
            panel_output=panel_output,
        ),
    ]
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=4.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest.update({
        "schema": "round0131-runtime-component-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0125"],
        "capability_dependencies": [R0125_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "environment_freeze": environment,
        "scientific_contract": {
            "r0125_required_outcome": sorted(POSITIVE_R0125_OUTCOMES),
            "observed_trigger_outcome": evidence["outcome"],
            "path": [
                "host-numpy-host-fp16-fused",
                "host-numpy-device-fp16-fused",
                "host-numpy-device-fp16-separate",
                "torch-device-device-fp16-separate",
            ],
            "adjacent_treatments": [
                "feature residency/gather",
                "endpoint forward mode",
                "sampler RNG plus epoch-batching mechanism",
            ],
            "new_arm_pipelines": PIPELINES,
            "native_intermediate_quality_tested": False,
            "production_runtime_adopted": False,
        },
        "p90_gpu_seconds": {
            "numpy_device_fused_train": 5_200.0,
            "numpy_device_separate_train": 5_200.0,
            "matched_component_panel": 360.0,
            "total": 10_760.0,
        },
        "estimate_basis": {
            "expected_gpu_hours": 2.65,
            "p90_gpu_hours": 2.99,
            "hard_cap_gpu_hours": 4.0,
            "basis": "two R0125-class 500k-update Jina-768 trains plus one exact matched panel",
        },
        "jobs": jobs,
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0125-review", required=True)
    parser.add_argument("--cpu-smoke-receipt", required=True)
    parser.add_argument("--cpu-smoke-sha256", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    path = prepare_round0131(
        release_sha=args.release_sha,
        r0125_review=args.r0125_review,
        cpu_smoke_receipt=args.cpu_smoke_receipt,
        cpu_smoke_sha256=args.cpu_smoke_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

