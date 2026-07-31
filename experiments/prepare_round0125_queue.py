#!/usr/bin/env python3
"""Prepare, but never launch, the paired R0125 runtime-bridge queue."""
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
from basemap.round0125_runtime_bridge import (
    ARMS,
    CAPABILITY,
    DEVICE_ARM,
    HOST_ARM,
    R0104_RELEASE_SHA,
    R0104_SHARED_RECEIPT_SHA256,
    R0122_PANEL_SHA256,
    R0122_RELEASE_SHA,
    environment_freeze_receipt,
    validate_seal,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0119_queue import (
    _clean_terminal,
    _document,
    _frontmatter_list,
)


ROUND_ID = "0125"
ROUND_ROOT = "/data/latent-basemap/runs/round-0125"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
RUN_ENVIRONMENT_PREFIX = os.path.join(RELEASE_ROOT, ".venv")
RUN_PYTHON = os.path.join(RUN_ENVIRONMENT_PREFIX, "bin", "python")
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0125-*.md")

R0104_REVIEW_SHA256 = (
    "febc1033d4edcfdf75e48f77065d8236ef36dde261434d3f1bb557cab48b6cde"
)
R0104_RESULT_SHA256 = (
    "0f36830807dacfe7f679865ef49f49ee349d3c8e7ea883d11fd69bce7ba077ff"
)
R0122_REVIEW_SHA256 = (
    "30a34b3b7c931917e038042a602f7570e753f190bb5b728e6e59ca0f0ae7d1a8"
)
R0122_RESULT_SHA256 = (
    "ed9aacab4599928e34bf8bdbcb4fc626e23db82f2425c07a9be0c0859bbd0b73"
)
R0104_CAPABILITY = "jina-full768-host-int8-training-validation-v1"
R0122_CAPABILITY = "jina-density-provenance-representation-bridge-v1"

R0104_QUEUE = (
    "/data/latent-basemap/runs/round-0104/queue-attempt-3/queue.json"
)
R0104_TERMINAL = (
    "/data/latent-basemap/runs/round-0104/queue-attempt-3/"
    "runner-terminal.json"
)
R0104_SHARED = (
    "/data/latent-basemap/runs/round-0104/queue-attempt-3/artifacts/shared"
)
R0104_SHARED_RECEIPT = os.path.join(R0104_SHARED, "receipt.json")

R0122_QUEUE = "/data/latent-basemap/runs/round-0122/queue/queue.json"
R0122_TERMINAL = (
    "/data/latent-basemap/runs/round-0122/queue/runner-terminal.json"
)
R0122_PANEL = (
    "/data/latent-basemap/runs/round-0122/queue/artifacts/"
    "density-provenance-bridge/density-bridge-panel.json"
)


def _require_dedicated_run_environment() -> None:
    observed_python = os.path.abspath(sys.executable)
    observed_prefix = os.path.abspath(sys.prefix)
    if (
        observed_python != RUN_PYTHON
        or observed_prefix != RUN_ENVIRONMENT_PREFIX
    ):
        raise RuntimeError(
            "R0125 queue preparation must use the dedicated run "
            f"environment: python={RUN_PYTHON}, "
            f"prefix={RUN_ENVIRONMENT_PREFIX}; observed "
            f"python={observed_python}, prefix={observed_prefix}"
        )


def _issued_round() -> str:
    candidates = []
    for path in sorted(glob.glob(ROUND_FILE_GLOB)):
        frontmatter, _ = _document(path)
        if frontmatter.get("status") == "issued":
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0125 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _accepted_evidence(
    path: str,
    *,
    round_id: str,
    review_sha256: str,
    result_sha256: str,
    release_sha: str,
    capability: str,
    queue_path: str,
) -> dict[str, Any]:
    review_signature = expected_input_signature(path)
    review_frontmatter, review_text = _document(path)
    result_name = review_frontmatter.get("result") or ""
    if (
        review_signature["sha256"] != review_sha256
        or review_frontmatter.get("round_id") != round_id
        or review_frontmatter.get("status") != "accepted"
        or review_frontmatter.get("verified_release_commit") != release_sha
        or f"capability:{capability}"
        not in _frontmatter_list(
            review_frontmatter,
            "releases",
            label=f"R{round_id} review",
        )
        or os.path.basename(result_name) != result_name
        or re.fullmatch(
            rf"result-{round_id}-[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\.md",
            result_name,
        )
        is None
    ):
        raise RuntimeError(f"R{round_id} review is not exact and accepted")
    result_path = os.path.join(os.path.dirname(path), result_name)
    result_signature = expected_input_signature(result_path)
    result_frontmatter, result_text = _document(result_path)
    queue_signature = expected_input_signature(queue_path)
    if (
        result_signature["sha256"] != result_sha256
        or review_frontmatter.get("result_sha256") != result_sha256
        or result_frontmatter.get("round_id") != round_id
        or result_frontmatter.get("status") != "complete"
        or result_frontmatter.get("release_commit") != release_sha
        or result_frontmatter.get("queue_manifest") != f"gsv:{queue_path}"
        or result_frontmatter.get("queue_manifest_sha256")
        != queue_signature["sha256"]
        or capability
        not in _frontmatter_list(
            result_frontmatter,
            "capabilities_produced",
            label=f"R{round_id} result",
        )
    ):
        raise RuntimeError(
            f"R{round_id} accepted review does not close to result/release"
        )
    return {
        "review": review_signature,
        "result": result_signature,
        "queue": queue_signature,
        "evidence_text": review_text + "\n" + result_text,
    }


def _exact_shared_inputs(evidence_text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shared_signature = expected_input_signature(R0104_SHARED_RECEIPT)
    if shared_signature["sha256"] != R0104_SHARED_RECEIPT_SHA256:
        raise RuntimeError("accepted R0104 shared receipt bytes changed")
    with open(R0104_SHARED_RECEIPT, encoding="utf-8") as handle:
        shared = json.load(handle)
    validate_seal(shared, label="accepted R0104 shared receipt")
    required = [
        dict(shared[key])
        for key in (
            "graph",
            "graph_manifest",
            "high_d_reference",
            "query_truth",
        )
    ]
    for signature in required:
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise RuntimeError("accepted R0104 shared artifact bytes changed")
    if shared_signature["sha256"] not in evidence_text:
        raise RuntimeError("accepted R0104 evidence does not bind shared receipt")
    return shared_signature, required


def _one_job(queue_path: str, *, action: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(queue_path, encoding="utf-8") as handle:
        queue = json.load(handle)
    jobs = [job for job in queue.get("jobs") or [] if job.get("action") == action]
    if len(jobs) != 1:
        raise RuntimeError(f"{queue_path} does not contain one {action!r} job")
    return queue, jobs[0]


def _accepted_job_inputs(
    queue_path: str,
    *,
    action: str,
    expected_release_sha: str,
    expected_output: str,
) -> list[dict[str, Any]]:
    queue, job = _one_job(queue_path, action=action)
    inputs = job.get("expected_inputs")
    if (
        queue.get("release_sha") != expected_release_sha
        or os.path.realpath(str(queue.get("repo_root") or ""))
        != os.path.realpath(RELEASE_ROOT)
        or job.get("outputs") != [expected_output]
        or not isinstance(inputs, list)
        or not inputs
    ):
        raise RuntimeError(f"accepted {action} job contract changed")
    exact = []
    for signature in inputs:
        if (
            not isinstance(signature, Mapping)
            or expected_input_signature(signature.get("canonical_path", ""))
            != dict(signature)
        ):
            raise RuntimeError(f"accepted {action} input bytes changed")
        exact.append(dict(signature))
    return exact


def _r0122_matched_inputs(
    evidence_text: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    queue, score_job = _one_job(
        R0122_QUEUE, action="score_density_provenance_bridge"
    )
    if queue.get("release_sha") != R0122_RELEASE_SHA:
        raise RuntimeError("accepted R0122 queue release changed")
    panel_signature = expected_input_signature(R0122_PANEL)
    if panel_signature["sha256"] != R0122_PANEL_SHA256:
        raise RuntimeError("accepted R0122 panel bytes changed")
    with open(R0122_PANEL, encoding="utf-8") as handle:
        panel = json.load(handle)
    validate_seal(panel, label="accepted R0122 density panel")
    calibration = dict(score_job.get("r0108_calibration") or {})
    if (
        panel.get("schema")
        != "round0122-jina-density-provenance-bridge-panel-v1"
        or expected_input_signature(calibration.get("canonical_path", ""))
        != calibration
        or panel_signature["sha256"] not in evidence_text
    ):
        raise RuntimeError("accepted R0122 matched-density lineage changed")
    with open(calibration["canonical_path"], encoding="utf-8") as handle:
        calibration_receipt = json.load(handle)
    validate_seal(calibration_receipt, label="accepted R0108 calibration")
    nested = [
        dict(calibration_receipt[key])
        for key in (
            "arrays",
            "census",
            "census_receipt",
            "representative_reference",
        )
    ]
    for signature in nested:
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise RuntimeError("accepted R0108 calibration input bytes changed")
    return panel_signature, calibration, nested


def _cpu_smoke_receipt(
    path: str, expected_sha256: str, *, release_sha: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError("R0125 CPU preflight receipt bytes changed")
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label="R0125 CPU preflight")
    if (
        receipt.get("schema") != "round0125-cpu-preflight-v1"
        or receipt.get("release_sha") != release_sha
        or receipt.get("cuda_visible_devices") != ""
        or receipt.get("outcome") != "passed"
        or not all((receipt.get("checks") or {}).values())
    ):
        raise RuntimeError("R0125 CPU preflight did not pass on this release")
    source_files = receipt.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise RuntimeError("R0125 CPU preflight lacks source-file bindings")
    for source in source_files:
        if (
            not isinstance(source, Mapping)
            or expected_input_signature(source.get("canonical_path", ""))
            != dict(source)
        ):
            raise RuntimeError("R0125 CPU preflight source bytes changed")
    return signature, [dict(source) for source in source_files]


def prepare_round0125(
    *,
    release_sha: str,
    r0104_review: str,
    r0122_review: str,
    cpu_smoke_receipt: str,
    cpu_smoke_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", release_sha) is None:
        raise ValueError("R0125 release SHA must be one full commit")
    _require_dedicated_run_environment()
    round_file = _issued_round()
    evidence = {
        "0104": _accepted_evidence(
            r0104_review,
            round_id="0104",
            review_sha256=R0104_REVIEW_SHA256,
            result_sha256=R0104_RESULT_SHA256,
            release_sha=R0104_RELEASE_SHA,
            capability=R0104_CAPABILITY,
            queue_path=R0104_QUEUE,
        ),
        "0122": _accepted_evidence(
            r0122_review,
            round_id="0122",
            review_sha256=R0122_REVIEW_SHA256,
            result_sha256=R0122_RESULT_SHA256,
            release_sha=R0122_RELEASE_SHA,
            capability=R0122_CAPABILITY,
            queue_path=R0122_QUEUE,
        ),
    }
    runtime = {
        "0104": _clean_terminal(
            R0104_QUEUE,
            R0104_TERMINAL,
            round_id="0104",
            expected_release_sha=R0104_RELEASE_SHA,
        ),
        "0122": _clean_terminal(
            R0122_QUEUE,
            R0122_TERMINAL,
            round_id="0122",
            expected_release_sha=R0122_RELEASE_SHA,
        ),
    }
    shared, shared_artifacts = _exact_shared_inputs(
        evidence["0104"]["evidence_text"]
    )
    r0104_queue_inputs = _accepted_job_inputs(
        R0104_QUEUE,
        action="build_shared",
        expected_release_sha=R0104_RELEASE_SHA,
        expected_output=R0104_SHARED,
    )
    r0122_panel, calibration, calibration_inputs = _r0122_matched_inputs(
        evidence["0122"]["evidence_text"]
    )
    r0122_queue_inputs = _accepted_job_inputs(
        R0122_QUEUE,
        action="score_density_provenance_bridge",
        expected_release_sha=R0122_RELEASE_SHA,
        expected_output=os.path.dirname(R0122_PANEL),
    )
    smoke, smoke_sources = _cpu_smoke_receipt(
        cpu_smoke_receipt, cpu_smoke_sha256, release_sha=release_sha
    )
    environment = environment_freeze_receipt()
    common_inputs = _dedupe([
        expected_input_signature(round_file),
        smoke,
        *smoke_sources,
        *[
            dict(evidence[round_id][field])
            for round_id in ("0104", "0122")
            for field in ("review", "result", "queue")
        ],
        *[
            dict(runtime[round_id][field])
            for round_id in ("0104", "0122")
            for field in ("queue", "terminal")
        ],
        shared,
        *shared_artifacts,
        *r0104_queue_inputs,
        r0122_panel,
        calibration,
        *calibration_inputs,
        *r0122_queue_inputs,
    ])

    queue_root = create_fresh_directory(
        queue_root, label="R0125 paired runtime bridge queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    outputs = {
        arm: {
            "train": os.path.join(artifacts, arm, "train"),
            "transform": os.path.join(artifacts, arm, "native-transform"),
            "score": os.path.join(artifacts, arm, "native-score"),
        }
        for arm in ARMS
    }
    matched_output = os.path.join(artifacts, "matched-density")
    decision_output = os.path.join(artifacts, "decision")

    def job(
        job_id: str,
        action: str,
        deps: list[str],
        output: str,
        p90_wall_s: float,
        *,
        arm: str | None = None,
        gpu_required: bool = True,
        **extra: Any,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": job_id,
            "action": action,
            "handler_module": "experiments.round0125_nodes",
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
            **extra,
        }
        if arm is not None:
            value["arm"] = arm
        return value

    jobs = [
        job(
            "train_device_treatment", "train", [],
            outputs[DEVICE_ARM]["train"], 4_500.0,
            arm=DEVICE_ARM, shared_output=R0104_SHARED,
        ),
        job(
            "train_host_control", "train", ["train_device_treatment"],
            outputs[HOST_ARM]["train"], 4_500.0,
            arm=HOST_ARM, shared_output=R0104_SHARED,
        ),
        job(
            "transform_device_treatment", "transform", ["train_host_control"],
            outputs[DEVICE_ARM]["transform"], 240.0,
            arm=DEVICE_ARM, shared_output=R0104_SHARED,
            train_output=outputs[DEVICE_ARM]["train"],
        ),
        job(
            "score_device_treatment", "native_score",
            ["transform_device_treatment"], outputs[DEVICE_ARM]["score"],
            180.0, arm=DEVICE_ARM, shared_output=R0104_SHARED,
            train_output=outputs[DEVICE_ARM]["train"],
            transform_output=outputs[DEVICE_ARM]["transform"],
        ),
        job(
            "transform_host_control", "transform", ["score_device_treatment"],
            outputs[HOST_ARM]["transform"], 240.0,
            arm=HOST_ARM, shared_output=R0104_SHARED,
            train_output=outputs[HOST_ARM]["train"],
        ),
        job(
            "score_host_control", "native_score", ["transform_host_control"],
            outputs[HOST_ARM]["score"], 180.0,
            arm=HOST_ARM, shared_output=R0104_SHARED,
            train_output=outputs[HOST_ARM]["train"],
            transform_output=outputs[HOST_ARM]["transform"],
        ),
        job(
            "score_matched_density", "matched_density",
            ["score_device_treatment", "score_host_control"], matched_output,
            300.0, shared_output=R0104_SHARED,
            train_outputs={arm: outputs[arm]["train"] for arm in ARMS},
            r0108_calibration=calibration, r0122_panel=r0122_panel,
        ),
        job(
            "decide_runtime_bridge", "decide", ["score_matched_density"],
            decision_output, 30.0, gpu_required=False,
            train_outputs={arm: outputs[arm]["train"] for arm in ARMS},
            native_score_outputs={arm: outputs[arm]["score"] for arm in ARMS},
            matched_output=matched_output,
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
        "schema": "round0125-device-host-runtime-bridge-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0104", "0122"],
        "capability_dependencies": [R0104_CAPABILITY, R0122_CAPABILITY],
        "capabilities_produced": [
            CAPABILITY
        ],
        "training_performed": True,
        "environment_freeze": environment,
        "scientific_contract": {
            "design": (
                "same-round paired seed-42 R0104 replay; complete legacy "
                "device runtime bundle versus complete accepted host runtime bundle"
            ),
            "arm_order": list(ARMS),
            "only_treatment": "registered execution-path bundle",
            "identical_between_arms": [
                "exact first-2M R0103 FineWeb fp16 rows",
                "exact R0104 fuzzy k50 graph",
                "seed and initial model state",
                "model, optimizer, schedule, and 500000 successful-update dose",
                "Python environment freeze",
            ],
            "actual_execution_required": [
                "selected pipeline and sampler class",
                "positive and negative sampling semantics",
                "feature residency and conversion",
                "endpoint accounting including non-divisible epoch boundary",
                "first-eight-live-batch source and destination ID digests",
            ],
            "native_panel": list(
                (
                    "ffr", "density", "recall_at_10",
                    "oos_proj_ffr", "oos_proj_recall_at_10",
                )
            ),
            "matched_panel": (
                "exact accepted R0122/R0108 R0040 representative universe, "
                "anchors, high-D radii, density-v2 scorer, and unchanged floor"
            ),
            "paired_interval": "same-anchor 1000-draw 99% percentile bootstrap",
            "causal_scope": "single-seed complete-path-bundle evidence only",
            "sampler_only_cause_claimed": False,
            "residency_only_cause_claimed": False,
            "production_runtime_adopted": False,
        },
        "p90_gpu_seconds": {
            "device_train": 4_500.0,
            "host_train": 4_500.0,
            "native_transforms": 480.0,
            "native_scores": 360.0,
            "matched_density": 300.0,
            "total": 10_140.0,
        },
        "estimate_basis": {
            "expected_gpu_hours": 2.65,
            "p90_gpu_hours": 2.82,
            "hard_cap_gpu_hours": 4.0,
            "basis": (
                "two measured 500k-update jina-768 R0104-class trains plus "
                "two native panels and one two-model matched density replay"
            ),
        },
        "jobs": jobs,
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0104-review", required=True)
    parser.add_argument("--r0122-review", required=True)
    parser.add_argument("--cpu-smoke-receipt", required=True)
    parser.add_argument("--cpu-smoke-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0125(
        release_sha=args.release_sha,
        r0104_review=args.r0104_review,
        r0122_review=args.r0122_review,
        cpu_smoke_receipt=args.cpu_smoke_receipt,
        cpu_smoke_sha256=args.cpu_smoke_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
