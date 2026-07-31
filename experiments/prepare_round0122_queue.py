#!/usr/bin/env python3
"""Prepare, but never launch, the no-training R0122 density bridge."""
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
from basemap.round0108_evaluation import validate_seal
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
from experiments.round0119_nodes import CALIBRATION_SCHEMA
from experiments.round0122_nodes import (
    DECISION_SCHEMA,
    NEW_CELL_ORDER,
    R0104_MODEL_SHA256,
    R0119_DECISION_SCHEMA,
    R0119_DECISION_SHA256,
    R0119_PANEL_SHA256,
    R0119_RELEASE_SHA,
    R0119_SCORE_SCHEMA,
    REGISTERED_FLOOR,
    SCORE_SCHEMA,
)


ROUND_ID = "0122"
ROUND_ROOT = "/data/latent-basemap/runs/round-0122"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
RUN_ENVIRONMENT_PREFIX = os.path.join(RELEASE_ROOT, ".venv")
RUN_PYTHON = os.path.join(RUN_ENVIRONMENT_PREFIX, "bin", "python")
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0122-*.md")

R0104_RELEASE_SHA = "2b1b51746d4aeb01e9dd88b19aa6dc80ccbb8329"
R0104_QUEUE = (
    "/data/latent-basemap/runs/round-0104/queue-attempt-3/queue.json"
)
R0104_TERMINAL = (
    "/data/latent-basemap/runs/round-0104/queue-attempt-3/"
    "runner-terminal.json"
)
R0104_ROOT = (
    "/data/latent-basemap/runs/round-0104/queue-attempt-3/artifacts"
)
R0104_BUNDLE_PATHS = {
    "r0104_fp16_seed42_full_transform": (
        "fp16_control",
        os.path.join(R0104_ROOT, "fp16_control", "train"),
    ),
    "r0104_int8_seed42_full_transform": (
        "int8_treatment",
        os.path.join(R0104_ROOT, "int8_treatment", "train"),
    ),
}

R0119_QUEUE = "/data/latent-basemap/runs/round-0119/queue/queue.json"
R0119_TERMINAL = (
    "/data/latent-basemap/runs/round-0119/queue/runner-terminal.json"
)
R0119_PANEL = (
    "/data/latent-basemap/runs/round-0119/queue/artifacts/"
    "density-localization-panel/density-localization-panel.json"
)
R0119_DECISION = (
    "/data/latent-basemap/runs/round-0119/queue/artifacts/"
    "density-localization-decision/density-localization-decision.json"
)
REQUIRED_EVIDENCE = {
    "0104": (
        "capability:jina-full768-host-int8-training-validation-v1",
        R0104_RELEASE_SHA,
        R0104_QUEUE,
    ),
    "0119": (
        "capability:jina-density-failure-localization-v1",
        R0119_RELEASE_SHA,
        R0119_QUEUE,
    ),
}


def _require_dedicated_run_environment() -> None:
    """Fail before queue creation unless preparation uses the run venv."""
    observed_python = os.path.abspath(sys.executable)
    observed_prefix = os.path.abspath(sys.prefix)
    if (
        observed_python != RUN_PYTHON
        or observed_prefix != RUN_ENVIRONMENT_PREFIX
    ):
        raise RuntimeError(
            "R0122 queue preparation must use the dedicated run "
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
            f"R0122 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _accepted_evidence(
    path: str,
    expected_sha256: str,
    *,
    round_id: str,
) -> dict[str, Any]:
    capability, expected_release, expected_queue = REQUIRED_EVIDENCE[round_id]
    signature = expected_input_signature(path)
    frontmatter, review_text = _document(path)
    if (
        signature["sha256"] != expected_sha256
        or frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
        or capability
        not in _frontmatter_list(
            frontmatter,
            "releases",
            label=f"R{round_id} review",
        )
        or frontmatter.get("verified_release_commit") != expected_release
    ):
        raise RuntimeError(f"R{round_id} review is not exact and accepted")

    result_name = frontmatter.get("result") or ""
    if (
        os.path.basename(result_name) != result_name
        or re.fullmatch(
            rf"result-{round_id}-[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\.md",
            result_name,
        )
        is None
    ):
        raise RuntimeError(f"R{round_id} review result binding is invalid")
    result_path = os.path.join(os.path.dirname(path), result_name)
    result_signature = expected_input_signature(result_path)
    result_frontmatter, result_text = _document(result_path)
    queue_uri = f"gsv:{expected_queue}"
    if (
        result_signature["sha256"] != frontmatter.get("result_sha256")
        or result_frontmatter.get("round_id") != round_id
        or result_frontmatter.get("status") != "complete"
        or result_frontmatter.get("release_commit") != expected_release
        or result_frontmatter.get("queue_manifest") != queue_uri
        or capability.removeprefix("capability:")
        not in _frontmatter_list(
            result_frontmatter,
            "capabilities_produced",
            label=f"R{round_id} result",
        )
    ):
        raise RuntimeError(
            f"R{round_id} accepted review does not close to result/release"
        )
    queue_signature = expected_input_signature(expected_queue)
    if (
        result_frontmatter.get("queue_manifest_sha256")
        != queue_signature["sha256"]
    ):
        raise RuntimeError(f"R{round_id} result queue binding changed")
    return {
        "review": signature,
        "result": result_signature,
        "release_commit": expected_release,
        "queue": queue_signature,
        "evidence_text": review_text + "\n" + result_text,
    }


def _r0104_bundles(
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bundles = []
    text = str(evidence["evidence_text"])
    for key in NEW_CELL_ORDER[:2]:
        arm, root = R0104_BUNDLE_PATHS[key]
        bundle = {
            "key": key,
            "arm": arm,
            "train_receipt": expected_input_signature(
                os.path.join(root, "train-receipt.json")
            ),
            "production_config": expected_input_signature(
                os.path.join(root, "production-config.json")
            ),
            "model": expected_input_signature(
                os.path.join(root, "model.pt")
            ),
        }
        if bundle["model"]["sha256"] != R0104_MODEL_SHA256[key]:
            raise RuntimeError(f"{key} model identity changed")
        for field in ("train_receipt", "production_config", "model"):
            if bundle[field]["sha256"] not in text:
                raise RuntimeError(
                    f"R0104 accepted evidence does not bind {key} {field}"
                )
        bundles.append(bundle)
    return bundles


def _r0119_inputs(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    queue_signature = evidence["queue"]
    with open(R0119_QUEUE, encoding="utf-8") as handle:
        queue = json.load(handle)
    jobs = queue.get("jobs") or []
    score_jobs = [
        job
        for job in jobs
        if job.get("action") == "score_density_localization"
    ]
    if (
        queue.get("round_id") != "0119"
        or queue.get("release_sha") != R0119_RELEASE_SHA
        or len(score_jobs) != 1
    ):
        raise RuntimeError("R0119 accepted queue identity changed")
    score_job = score_jobs[0]
    all_bundles = score_job.get("model_bundles")
    if not isinstance(all_bundles, list):
        raise RuntimeError("R0119 model bundles are missing")
    by_key = {item.get("key"): item for item in all_bundles}
    replay_keys = ("current_2m_seed42", "current_2m_seed43")
    if any(key not in by_key for key in replay_keys):
        raise RuntimeError("R0119 direct replay bundles are missing")
    replay_bundles = [by_key[key] for key in replay_keys]

    panel_signature = expected_input_signature(R0119_PANEL)
    decision_signature = expected_input_signature(R0119_DECISION)
    if (
        panel_signature["sha256"] != R0119_PANEL_SHA256
        or decision_signature["sha256"] != R0119_DECISION_SHA256
    ):
        raise RuntimeError("R0119 panel/decision bytes changed")
    with open(R0119_PANEL, encoding="utf-8") as handle:
        panel = json.load(handle)
    with open(R0119_DECISION, encoding="utf-8") as handle:
        decision = json.load(handle)
    validate_seal(panel, label="R0119 density localization panel")
    validate_seal(decision, label="R0119 density localization decision")
    lineage = panel.get("lineage")
    scorer = panel.get("scorer")
    if (
        panel.get("schema") != R0119_SCORE_SCHEMA
        or panel.get("round_id") != "0119"
        or panel.get("release_sha") != R0119_RELEASE_SHA
        or not isinstance(lineage, Mapping)
        or lineage.get("registered_floor") != REGISTERED_FLOOR
        or not isinstance(scorer, Mapping)
        or scorer.get("registered_floor") != REGISTERED_FLOOR
        or scorer.get("transform_batch_rows") != 8_192
        or scorer.get("k") != 15
        or scorer.get("low_dim_search") != "exact"
        or decision.get("schema") != R0119_DECISION_SCHEMA
        or decision.get("round_id") != "0119"
        or decision.get("release_sha") != R0119_RELEASE_SHA
        or decision.get("score") != panel_signature
        or decision.get("outcome") != "failure-not-unique-to-25m-tuple"
    ):
        raise RuntimeError("R0119 accepted density contract changed")
    calibration = lineage.get("r0108_calibration")
    if not isinstance(calibration, Mapping):
        raise RuntimeError("R0119 calibration binding is missing")
    with open(calibration["canonical_path"], encoding="utf-8") as handle:
        calibration_receipt = json.load(handle)
    validate_seal(
        calibration_receipt, label="R0108 density calibration"
    )
    if (
        calibration_receipt.get("schema") != CALIBRATION_SCHEMA
        or expected_input_signature(calibration["canonical_path"])
        != dict(calibration)
    ):
        raise RuntimeError("R0119 R0108 calibration bytes changed")

    text = str(evidence["evidence_text"])
    for signature in (
        queue_signature,
        panel_signature,
        decision_signature,
        *[
            dict(bundle[field])
            for bundle in replay_bundles
            for field in ("train_receipt", "production_config", "model")
        ],
    ):
        if signature["sha256"] not in text:
            raise RuntimeError(
                "R0119 accepted evidence does not bind a required input"
            )
    return {
        "panel": panel_signature,
        "decision": decision_signature,
        "lineage": dict(lineage),
        "calibration": dict(calibration),
        "replay_bundles": replay_bundles,
    }


def prepare_round0122(
    *,
    release_sha: str,
    r0104_review: tuple[str, str],
    r0119_review: tuple[str, str],
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", release_sha) is None:
        raise ValueError("R0122 release SHA must be one full commit")
    _require_dedicated_run_environment()
    round_file = _issued_round()
    evidence = {
        "0104": _accepted_evidence(
            r0104_review[0],
            r0104_review[1],
            round_id="0104",
        ),
        "0119": _accepted_evidence(
            r0119_review[0],
            r0119_review[1],
            round_id="0119",
        ),
    }
    runtime = {
        "0104": _clean_terminal(
            R0104_QUEUE,
            R0104_TERMINAL,
            round_id="0104",
            expected_release_sha=R0104_RELEASE_SHA,
        ),
        "0119": _clean_terminal(
            R0119_QUEUE,
            R0119_TERMINAL,
            round_id="0119",
            expected_release_sha=R0119_RELEASE_SHA,
        ),
    }
    r0104_bundles = _r0104_bundles(evidence["0104"])
    r0119 = _r0119_inputs(evidence["0119"])
    lineage_signatures = [
        dict(value)
        for value in r0119["lineage"].values()
        if isinstance(value, Mapping)
        and set(("canonical_path", "sha256", "bytes", "kind"))
        <= set(value)
    ]
    common_inputs = _dedupe([
        expected_input_signature(round_file),
        *[
            dict(evidence[round_id][field])
            for round_id in ("0104", "0119")
            for field in ("review", "result", "queue")
        ],
        *[
            dict(runtime[round_id][field])
            for round_id in ("0104", "0119")
            for field in ("queue", "terminal")
        ],
        r0119["panel"],
        r0119["decision"],
        *lineage_signatures,
        *[
            dict(bundle[field])
            for bundle in (
                *r0104_bundles,
                *r0119["replay_bundles"],
            )
            for field in ("train_receipt", "production_config", "model")
        ],
    ])

    queue_root = create_fresh_directory(
        queue_root, label="R0122 density provenance bridge queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    score_output = os.path.join(artifacts, "density-provenance-bridge")
    decision_output = os.path.join(
        artifacts, "density-provenance-bridge-decision"
    )
    jobs = [
        {
            "id": "score_density_provenance_bridge",
            "action": "score_density_provenance_bridge",
            "handler_module": "experiments.round0122_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [score_output],
            "done_marker": os.path.join(
                artifacts, "score_density_provenance_bridge.done.json"
            ),
            "expected_inputs": common_inputs,
            "p90_wall_s": 180.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "r0108_calibration": r0119["calibration"],
            "r0119_panel": r0119["panel"],
            "r0119_decision": r0119["decision"],
            "r0104_model_bundles": r0104_bundles,
            "r0119_replay_model_bundles": r0119["replay_bundles"],
        },
        {
            "id": "decide_density_provenance_bridge",
            "action": "decide_density_provenance_bridge",
            "handler_module": "experiments.round0122_nodes",
            "handler_callable": "run_job",
            "deps": ["score_density_provenance_bridge"],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts, "decide_density_provenance_bridge.done.json"
            ),
            "expected_inputs": _dedupe([
                expected_input_signature(round_file),
                evidence["0104"]["review"],
                evidence["0104"]["result"],
                evidence["0119"]["review"],
                evidence["0119"]["result"],
                r0119["panel"],
                r0119["decision"],
            ]),
            "p90_wall_s": 30.0,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
            "score_output": score_output,
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.25,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0122-jina-density-provenance-bridge-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0104", "0119"],
        "capability_dependencies": [
            "jina-full768-host-int8-training-validation-v1",
            "jina-density-failure-localization-v1",
        ],
        "capabilities_produced": [
            "jina-density-provenance-representation-bridge-v1",
        ],
        "training_performed": False,
        "scientific_contract": {
            "purpose": (
                "localize the R0119 matched-density failure across "
                "evaluation path and the R0104-to-R0115 bundled transition"
            ),
            "universe_floor_and_scorer": (
                "exact accepted R0119 R0040 universe, anchors, high-D "
                "radii, family filter, density-v2 scorer, and unchanged "
                "0.17589389755990817 floor"
            ),
            "new_cells": list(NEW_CELL_ORDER),
            "new_cell_transform_path": (
                "transform all 2000000 R0040 source rows in exact 8192-row "
                "batches, then select exact R0040 representatives"
            ),
            "reused_r0119_cells": [
                "historical_2m_seed42",
                "historical_2m_seed43",
                "current_2m_seed42",
                "current_2m_seed43",
            ],
            "branch_order": [
                (
                    "if either R0115/R0117 replay floor classification "
                    "changes from its R0119 direct cell: "
                    "evaluation-path-material and stop"
                ),
                (
                    "otherwise, if R0104 fp16 passes while both raw replays "
                    "fail: failure enters after R0104 within the bundled "
                    "fresh-native8192/representative/graph/sampler transition"
                ),
                (
                    "otherwise, if R0104 fp16 fails: failure was already "
                    "present before R0115"
                ),
            ],
            "storage_diagnostic": (
                "fp16/int8 floor-classification disagreement is "
                "storage-sensitive diagnostic evidence only; the boundary "
                "conclusion is always tied to fp16"
            ),
            "single_factor_cause_localized": False,
            "native_r0115_density_context": {
                "reported_density": 0.2304,
                "numerically_clears_r0119_floor": True,
                "same_matched_universe_and_scorer": False,
                "role": (
                    "this is calibration/representation-transfer "
                    "localization, not proof of bad native training geometry"
                ),
            },
            "training_performed": False,
            "map_decision": False,
            "production_transfer": False,
        },
        "p90_gpu_seconds": {
            "score_density_provenance_bridge": 180.0,
            "total": 180.0,
        },
        "estimate_basis": {
            "expected_gpu_hours": 0.01,
            "p90_gpu_hours": 0.05,
            "hard_cap_gpu_hours": 0.25,
            "basis": (
                "four full-source transforms plus exact 2D density scoring; "
                "R0119 completed six related cells in 29.46 seconds"
            ),
        },
        "jobs": jobs,
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0104-review", required=True)
    parser.add_argument("--r0104-review-sha256", required=True)
    parser.add_argument("--r0119-review", required=True)
    parser.add_argument("--r0119-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0122(
        release_sha=args.release_sha,
        r0104_review=(
            args.r0104_review,
            args.r0104_review_sha256,
        ),
        r0119_review=(
            args.r0119_review,
            args.r0119_review_sha256,
        ),
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
