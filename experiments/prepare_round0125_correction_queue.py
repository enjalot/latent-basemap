#!/usr/bin/env python3
"""Prepare R0125's eval-only setup correction queue; never launch it."""
from __future__ import annotations

import argparse
import json
import math
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
    ORIGINAL_RELEASE_SHA,
    R0104_QUERY_TRUTH_KEY,
    R0104_QUERY_TRUTH_PRODUCER_BACKEND,
    R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256,
    R0104_QUERY_TRUTH_SHA256,
    environment_freeze_receipt,
    validate_seal,
)
from experiments.prepare_round0020_0022_queues import _base_manifest, _dedupe
from experiments.prepare_round0119_queue import _clean_terminal
from experiments.prepare_round0125_queue import (
    R0104_CAPABILITY,
    R0104_QUEUE,
    R0104_RELEASE_SHA,
    R0104_REVIEW_SHA256,
    R0104_RESULT_SHA256,
    R0104_SHARED,
    R0104_TERMINAL,
    R0122_CAPABILITY,
    R0122_PANEL,
    R0122_QUEUE,
    R0122_RELEASE_SHA,
    R0122_REVIEW_SHA256,
    R0122_RESULT_SHA256,
    R0122_TERMINAL,
    ROUND_ROOT,
    _accepted_evidence,
    _accepted_job_inputs,
    _exact_shared_inputs,
    _issued_round,
    _r0122_matched_inputs,
    _require_dedicated_run_environment,
)


PRIOR_QUEUE = os.path.join(ROUND_ROOT, "queue", "queue.json")
PRIOR_TERMINAL = os.path.join(ROUND_ROOT, "queue", "runner-terminal.json")
PRIOR_QUEUE_SHA256 = (
    "9b7e966420e6723fc840a16c8f1477c6fd7ca8fbbc1b7b4009ec388d97321bcf"
)
PRIOR_TERMINAL_SHA256 = (
    "4095fbdc66b89ac721bb12518cc16375e92cf29f54b7d559ee37e2136704e835"
)
PRIOR_GPU_WALL_S = 9_246.523752104957
ROUND_GPU_CAP_S = 4.0 * 3_600.0
RESIDUAL_GPU_CAP_HOURS = (ROUND_GPU_CAP_S - PRIOR_GPU_WALL_S) / 3_600.0
PRIOR_ARTIFACT_ROOT = os.path.join(ROUND_ROOT, "queue", "artifacts")
PRIOR_TRAIN_OUTPUTS = {
    arm: os.path.join(PRIOR_ARTIFACT_ROOT, arm, "train") for arm in ARMS
}
PRIOR_DEVICE_TRANSFORM = os.path.join(
    PRIOR_ARTIFACT_ROOT, DEVICE_ARM, "native-transform"
)
PRIOR_DONE_MARKERS = {
    "train_device_treatment": os.path.join(
        PRIOR_ARTIFACT_ROOT, "train_device_treatment.done.json"
    ),
    "train_host_control": os.path.join(
        PRIOR_ARTIFACT_ROOT, "train_host_control.done.json"
    ),
    "transform_device_treatment": os.path.join(
        PRIOR_ARTIFACT_ROOT, "transform_device_treatment.done.json"
    ),
}
PRIOR_FAILED_MARKER = os.path.join(
    PRIOR_ARTIFACT_ROOT, "score_device_treatment.failed.json"
)
PRIOR_EXPECTED_SHA256 = {
    PRIOR_DONE_MARKERS["train_device_treatment"]: (
        "1ed1c8993f61b57cec9b363ad17138b55cb93bdb82eec8ae5dcff79294115126"
    ),
    PRIOR_DONE_MARKERS["train_host_control"]: (
        "b000844d7c6e386c0d9b55e230db266fda627b356ee9ab9dc87a1f6d4d5f324e"
    ),
    PRIOR_DONE_MARKERS["transform_device_treatment"]: (
        "2fab371abb53262bf6f1f493c6192deae5b8e0b64fe4ed98ecdf7a9d43eca476"
    ),
    PRIOR_FAILED_MARKER: (
        "fa5b4b3a0f864a816d3ceb802441671e6cec5311676347728dbbec3bedbc84af"
    ),
    os.path.join(PRIOR_TRAIN_OUTPUTS[DEVICE_ARM], "production-config.json"): (
        "63fb2b70fe265f0260cb7d6000850e0ad9f9ff8e84df62158c17c11a0118d180"
    ),
    os.path.join(PRIOR_TRAIN_OUTPUTS[DEVICE_ARM], "model.pt"): (
        "030d9508356d023b0ea2cf38d8da7165dd78f0c3712da6d13c70805f1a9e6ae8"
    ),
    os.path.join(PRIOR_TRAIN_OUTPUTS[DEVICE_ARM], "train-receipt.json"): (
        "27789f13cd179d540548f9bad620bee7979155ae90b38ab2bdf103226007117f"
    ),
    os.path.join(PRIOR_TRAIN_OUTPUTS[HOST_ARM], "production-config.json"): (
        "90d546c799d36c4bb6c48007b9c72f55d32726ce6f010d1dd0c13d5286e63f1e"
    ),
    os.path.join(PRIOR_TRAIN_OUTPUTS[HOST_ARM], "model.pt"): (
        "36a7fb86784b6a891f7c73b83d008aead320a7729eea913efc117e4bcd5b3e08"
    ),
    os.path.join(PRIOR_TRAIN_OUTPUTS[HOST_ARM], "train-receipt.json"): (
        "230046e8468f565632136009798034575a839be5a2bd194bee45492972f10d77"
    ),
    os.path.join(PRIOR_DEVICE_TRANSFORM, "coordinates.npy"): (
        "0ebf483e55697388bf8d5d7e5e54fa0fce6bf70c86edd4c5dc549e49d7188c04"
    ),
    os.path.join(PRIOR_DEVICE_TRANSFORM, "oos-query-coordinates.npy"): (
        "d049efd4509cdc5283529b1ba8fc2556ec2699defa87e0cb61aee9ad9d89976d"
    ),
    os.path.join(PRIOR_DEVICE_TRANSFORM, "transform-receipt.json"): (
        "e4001f7f510c762d299a402ce6bd20e6a1dcbacc9d5c012b082b1db117e4cf12"
    ),
}


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object at {path}")
    return value


def _prior_exact_signature(path: str) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != PRIOR_EXPECTED_SHA256.get(path):
        raise RuntimeError(f"R0125 prior artifact bytes changed: {path}")
    return signature


def _prior_attempt_inputs() -> list[dict[str, Any]]:
    queue_signature = expected_input_signature(PRIOR_QUEUE)
    terminal_signature = expected_input_signature(PRIOR_TERMINAL)
    queue = _read_json(PRIOR_QUEUE)
    terminal = _read_json(PRIOR_TERMINAL)
    if (
        queue_signature["sha256"] != PRIOR_QUEUE_SHA256
        or terminal_signature["sha256"] != PRIOR_TERMINAL_SHA256
        or queue.get("round_id") != "0125"
        or queue.get("release_sha") != ORIGINAL_RELEASE_SHA
        or float(queue.get("gpu_hours_cap", -1.0)) != 4.0
        or terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != "0125"
        or terminal.get("verdict") != "failed"
        or terminal.get("queue_manifest_sha256") != PRIOR_QUEUE_SHA256
        or terminal.get("release_checkout", {}).get("head")
        != ORIGINAL_RELEASE_SHA
        or terminal.get("gpu_wall_accounting_complete") is not True
        or not math.isclose(
            float(terminal.get("gpu_wall_s", -1.0)),
            PRIOR_GPU_WALL_S,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or terminal.get("completed_jobs")
        != [
            "train_device_treatment",
            "train_host_control",
            "transform_device_treatment",
        ]
    ):
        raise RuntimeError("R0125 failed-attempt terminal/queue identity changed")

    signatures: list[dict[str, Any]] = [queue_signature, terminal_signature]
    for node, path in PRIOR_DONE_MARKERS.items():
        marker = _read_json(path)
        if (
            marker.get("schema") != "slim-runner-done-v2"
            or marker.get("node") != node
            or marker.get("returncode") != 0
            or marker.get("queue_manifest_sha256") != PRIOR_QUEUE_SHA256
            or marker.get("release_sha") != ORIGINAL_RELEASE_SHA
        ):
            raise RuntimeError(f"R0125 prior done marker {node} changed")
        signatures.append(_prior_exact_signature(path))
    failed = _read_json(PRIOR_FAILED_MARKER)
    if (
        failed.get("schema") != "slim-runner-failed-v2"
        or failed.get("node") != "score_device_treatment"
        or failed.get("queue_manifest_sha256") != PRIOR_QUEUE_SHA256
        or failed.get("release_sha") != ORIGINAL_RELEASE_SHA
        or "query truth implementation/backend identity mismatch"
        not in str(failed.get("log_tail") or "")
    ):
        raise RuntimeError("R0125 setup-class failed marker changed")
    signatures.append(_prior_exact_signature(PRIOR_FAILED_MARKER))

    artifact_files = [
        *[
            os.path.join(PRIOR_TRAIN_OUTPUTS[arm], name)
            for arm in ARMS
            for name in (
                "production-config.json",
                "model.pt",
                "train-receipt.json",
            )
        ],
        os.path.join(PRIOR_DEVICE_TRANSFORM, "coordinates.npy"),
        os.path.join(PRIOR_DEVICE_TRANSFORM, "oos-query-coordinates.npy"),
        os.path.join(PRIOR_DEVICE_TRANSFORM, "transform-receipt.json"),
    ]
    for path in artifact_files:
        signatures.append(_prior_exact_signature(path))

    for arm in ARMS:
        receipt = _read_json(
            os.path.join(PRIOR_TRAIN_OUTPUTS[arm], "train-receipt.json")
        )
        validate_seal(receipt, label=f"prior R0125 {arm} train")
        train_checks = receipt.get("train_checks")
        if (
            receipt.get("schema") != "round0125-runtime-arm-train-receipt-v1"
            or receipt.get("round_id") != "0125"
            or receipt.get("arm") != arm
            or receipt.get("release_sha") != ORIGINAL_RELEASE_SHA
            or not isinstance(train_checks, Mapping)
            or set(train_checks)
            != {
                "exact_update_closure",
                "zero_numerical_skips",
                "no_pipeline_stamp_drift",
                "endpoint_rows_match_registered_path",
                "bounded_stream_trace_complete",
                "initial_model_state_stamped",
            }
            or not all(train_checks.values())
            or (receipt.get("train_accounting") or {}).get(
                "positive_lr_optimizer_steps"
            )
            != 500_000
        ):
            raise RuntimeError(f"R0125 prior {arm} train evidence changed")
    transform = _read_json(
        os.path.join(PRIOR_DEVICE_TRANSFORM, "transform-receipt.json")
    )
    validate_seal(transform, label="prior R0125 device transform")
    if (
        transform.get("schema") != "round0125-native-transform-receipt-v1"
        or transform.get("round_id") != "0125"
        or transform.get("arm") != DEVICE_ARM
        or transform.get("release_sha") != ORIGINAL_RELEASE_SHA
        or transform.get("finite") is not True
    ):
        raise RuntimeError("R0125 prior device transform evidence changed")
    return signatures


def _query_truth_smoke_inputs(
    path: str, expected_sha256: str, *, release_sha: str
) -> list[dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError("R0125 corrected query-truth smoke bytes changed")
    receipt = _read_json(path)
    validate_seal(receipt, label="R0125 corrected query-truth smoke")
    policy = receipt.get("query_truth_producer_policy") or {}
    if (
        receipt.get("schema")
        != "round0125-accepted-query-truth-cpu-smoke-v1"
        or receipt.get("release_sha") != release_sha
        or receipt.get("cuda_visible_devices") != ""
        or receipt.get("torch_threads") != 1
        or receipt.get("outcome") != "passed"
        or not all((receipt.get("checks") or {}).values())
        or receipt.get("query_truth", {}).get("sha256")
        != R0104_QUERY_TRUTH_SHA256
        or receipt.get("query_truth_key") != R0104_QUERY_TRUTH_KEY
        or policy.get("implementation_sha256")
        != R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256
        or policy.get("candidate_compute_backend")
        != R0104_QUERY_TRUTH_PRODUCER_BACKEND
    ):
        raise RuntimeError("R0125 corrected query-truth smoke is not exact")
    source_files = receipt.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise RuntimeError("R0125 corrected smoke lacks source bindings")
    for item in source_files:
        if (
            not isinstance(item, Mapping)
            or expected_input_signature(item.get("canonical_path", ""))
            != dict(item)
        ):
            raise RuntimeError("R0125 corrected smoke source bytes changed")
    return [signature, *[dict(item) for item in source_files]]


def prepare_correction_queue(
    *,
    release_sha: str,
    r0104_review: str,
    r0122_review: str,
    query_truth_smoke: str,
    query_truth_smoke_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue-attempt-2"),
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", release_sha) is None:
        raise ValueError("R0125 correction release SHA must be one commit")
    if release_sha == ORIGINAL_RELEASE_SHA:
        raise ValueError("R0125 correction must use a new release commit")
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
            R0104_QUEUE, R0104_TERMINAL,
            round_id="0104", expected_release_sha=R0104_RELEASE_SHA,
        ),
        "0122": _clean_terminal(
            R0122_QUEUE, R0122_TERMINAL,
            round_id="0122", expected_release_sha=R0122_RELEASE_SHA,
        ),
    }
    shared, shared_artifacts = _exact_shared_inputs(
        evidence["0104"]["evidence_text"]
    )
    r0104_inputs = _accepted_job_inputs(
        R0104_QUEUE,
        action="build_shared",
        expected_release_sha=R0104_RELEASE_SHA,
        expected_output=R0104_SHARED,
    )
    r0122_panel, calibration, calibration_inputs = _r0122_matched_inputs(
        evidence["0122"]["evidence_text"]
    )
    r0122_inputs = _accepted_job_inputs(
        R0122_QUEUE,
        action="score_density_provenance_bridge",
        expected_release_sha=R0122_RELEASE_SHA,
        expected_output=os.path.dirname(R0122_PANEL),
    )
    common_inputs = _dedupe([
        expected_input_signature(round_file),
        *_query_truth_smoke_inputs(
            query_truth_smoke,
            query_truth_smoke_sha256,
            release_sha=release_sha,
        ),
        *_prior_attempt_inputs(),
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
        *r0104_inputs,
        r0122_panel,
        calibration,
        *calibration_inputs,
        *r0122_inputs,
    ])

    queue_root = create_fresh_directory(
        queue_root, label="R0125 eval-only setup correction queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    device_score = os.path.join(artifacts, DEVICE_ARM, "native-score")
    host_transform = os.path.join(artifacts, HOST_ARM, "native-transform")
    host_score = os.path.join(artifacts, HOST_ARM, "native-score")
    matched = os.path.join(artifacts, "matched-density")
    decision = os.path.join(artifacts, "decision")

    def job(
        job_id: str,
        action: str,
        deps: list[str],
        output: str,
        p90_wall_s: float,
        *,
        gpu_required: bool = True,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
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
                "training_performed": False,
            },
            **extra,
        }

    jobs = [
        job(
            "score_device_treatment", "native_score", [], device_score, 180.0,
            arm=DEVICE_ARM,
            shared_output=R0104_SHARED,
            train_output=PRIOR_TRAIN_OUTPUTS[DEVICE_ARM],
            transform_output=PRIOR_DEVICE_TRANSFORM,
            train_release_sha=ORIGINAL_RELEASE_SHA,
            transform_release_sha=ORIGINAL_RELEASE_SHA,
        ),
        job(
            "transform_host_control", "transform", ["score_device_treatment"],
            host_transform, 240.0,
            arm=HOST_ARM,
            shared_output=R0104_SHARED,
            train_output=PRIOR_TRAIN_OUTPUTS[HOST_ARM],
            train_release_sha=ORIGINAL_RELEASE_SHA,
        ),
        job(
            "score_host_control", "native_score", ["transform_host_control"],
            host_score, 180.0,
            arm=HOST_ARM,
            shared_output=R0104_SHARED,
            train_output=PRIOR_TRAIN_OUTPUTS[HOST_ARM],
            transform_output=host_transform,
            train_release_sha=ORIGINAL_RELEASE_SHA,
        ),
        job(
            "score_matched_density", "matched_density",
            ["score_device_treatment", "score_host_control"], matched, 300.0,
            shared_output=R0104_SHARED,
            train_outputs=PRIOR_TRAIN_OUTPUTS,
            train_release_sha=ORIGINAL_RELEASE_SHA,
            r0108_calibration=calibration,
            r0122_panel=r0122_panel,
        ),
        job(
            "decide_runtime_bridge", "decide", ["score_matched_density"],
            decision, 30.0, gpu_required=False,
            train_outputs=PRIOR_TRAIN_OUTPUTS,
            train_release_sha=ORIGINAL_RELEASE_SHA,
            native_score_outputs={
                DEVICE_ARM: device_score,
                HOST_ARM: host_score,
            },
            matched_output=matched,
        ),
    ]
    manifest = _base_manifest(
        round_id="0125",
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=RESIDUAL_GPU_CAP_HOURS,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest.update({
        "schema": "round0125-eval-only-setup-correction-queue-v1",
        "queue_class": "gpu-research",
        "required_reviews": ["0104", "0122"],
        "capability_dependencies": [R0104_CAPABILITY, R0122_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "environment_freeze": environment_freeze_receipt(),
        "correction_attempt": {
            "class": "setup-only-query-truth-producer-compatibility",
            "prior_queue": expected_input_signature(PRIOR_QUEUE),
            "prior_terminal": expected_input_signature(PRIOR_TERMINAL),
            "prior_release_sha": ORIGINAL_RELEASE_SHA,
            "prior_gpu_wall_s": PRIOR_GPU_WALL_S,
            "round_gpu_cap_s": ROUND_GPU_CAP_S,
            "residual_gpu_cap_s": ROUND_GPU_CAP_S - PRIOR_GPU_WALL_S,
            "reused_nodes": list(PRIOR_DONE_MARKERS),
            "retraining_permitted": False,
        },
        "scientific_contract": {
            "science_and_thresholds_unchanged": True,
            "only_code_change": (
                "authenticate the immutable accepted R0104 query truth against "
                "its exact historical producer implementation and CUDA backend"
            ),
            "historical_truth_relabelled_as_current": False,
            "original_train_receipts_reused": True,
            "original_device_transform_reused": True,
            "training_performed_in_correction": False,
        },
        "p90_gpu_seconds": {
            "remaining_eval_only": 900.0,
            "prior_attempt": PRIOR_GPU_WALL_S,
            "cumulative_p90": PRIOR_GPU_WALL_S + 900.0,
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
    parser.add_argument("--query-truth-smoke", required=True)
    parser.add_argument("--query-truth-smoke-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue-attempt-2")
    )
    args = parser.parse_args(argv)
    path = prepare_correction_queue(
        release_sha=args.release_sha,
        r0104_review=args.r0104_review,
        r0122_review=args.r0122_review,
        query_truth_smoke=args.query_truth_smoke,
        query_truth_smoke_sha256=args.query_truth_smoke_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
