#!/usr/bin/env python3
"""Prepare the frozen-protocol evaluation of the R0111 seed-44 map."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ID = "0118"
ROUND_ROOT = "/data/latent-basemap/runs/round-0118"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0118-*.md")

R0108_ROOT = "/data/latent-basemap/runs/round-0108/queue-attempt-3"
R0108_QUEUE = os.path.join(R0108_ROOT, "queue.json")
R0108_TERMINAL = os.path.join(R0108_ROOT, "runner-terminal.json")
R0108_SELECTION = os.path.join(
    R0108_ROOT, "inputs", "registered-selections.npz"
)
R0108_ARTIFACTS = os.path.join(R0108_ROOT, "artifacts")
R0108_CALIBRATION_OUTPUT = os.path.join(
    R0108_ARTIFACTS, "jina-density-calibration"
)

R0110_ROOT = "/data/latent-basemap/runs/round-0110/queue"
R0110_QUEUE = os.path.join(R0110_ROOT, "queue.json")
R0110_TERMINAL = os.path.join(R0110_ROOT, "runner-terminal.json")
R0110_ARTIFACTS = os.path.join(R0110_ROOT, "artifacts")
R0110_MATCHED_OUTPUT = os.path.join(
    R0110_ARTIFACTS, "matched-calibration-density"
)
R0110_DECISION_OUTPUT = os.path.join(
    R0110_ARTIFACTS, "two-seed-decision"
)

R0111_ROOT = "/data/latent-basemap/runs/round-0111/queue"
R0111_QUEUE = os.path.join(R0111_ROOT, "queue.json")
R0111_TERMINAL = os.path.join(R0111_ROOT, "runner-terminal.json")
R0111_TRAIN_OUTPUT = os.path.join(
    R0111_ROOT, "artifacts", "train-diverse-jina-25m-seed44"
)

# R0108 attempt 3 measured 37.4s / 85.1s / 99.8s for transform, core,
# and OOD. R0110 measured 37.8s / 90.8s / 111.1s / 13.8s for transform,
# core, OOD, and the *two-model* matched-density cell. These bounds remain
# deliberately >2x the slower observed walls without retaining the original
# multi-hour estimates that obscured queue planning.
TRANSFORM_P90_S = 90.0
CORE_P90_S = 240.0
OOD_P90_S = 300.0
MATCHED_DENSITY_P90_S = 60.0
DECISION_P90_S = 60.0
GPU_P90_S = (
    TRANSFORM_P90_S
    + CORE_P90_S
    + OOD_P90_S
    + MATCHED_DENSITY_P90_S
)


def _frontmatter(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.startswith("---\n"):
        raise RuntimeError(f"missing frontmatter: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError(f"unterminated frontmatter: {path}")
    values: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0118 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_accepted_review(
    path: str,
    *,
    expected_sha256: str,
    round_id: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"Review {round_id} bytes changed")
    frontmatter = _frontmatter(path)
    if (
        frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
    ):
        raise RuntimeError(f"Review {round_id} is not accepted")
    return signature


def _require_clean_terminal(path: str, *, round_id: str) -> dict[str, Any]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal.get("completed_jobs") != terminal.get("required_jobs")
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("queue_manifest_unchanged") is not True
    ):
        raise RuntimeError(f"R{round_id} terminal is not a clean success")
    return signature


def _load_queue(
    path: str,
    *,
    schema: str,
    round_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        queue = json.load(handle)
    if queue.get("schema") != schema or queue.get("round_id") != round_id:
        raise RuntimeError(f"R{round_id} queue identity changed")
    return queue, signature


def _jobs_by_action(
    queue: dict[str, Any],
    *,
    required: set[str],
) -> dict[str, dict[str, Any]]:
    jobs = queue.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("source queue jobs are missing")
    result = {str(job.get("action")): job for job in jobs}
    if not required.issubset(result):
        raise RuntimeError("source scorer jobs are incomplete")
    return result


def _job_template(source: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "id",
        "action",
        "handler_module",
        "handler_callable",
        "deps",
        "outputs",
        "done_marker",
        "expected_inputs",
        "p90_wall_s",
        "train_output",
        "release_sha",
    }
    return {
        key: value for key, value in source.items() if key not in excluded
    }


def prepare_round0118(
    *,
    release_sha: str,
    r0110_review_path: str,
    r0110_review_sha256: str,
    r0111_review_path: str,
    r0111_review_sha256: str,
    r0108_queue_path: str = R0108_QUEUE,
    r0110_queue_path: str = R0110_QUEUE,
    r0111_queue_path: str = R0111_QUEUE,
    r0111_train_output: str = R0111_TRAIN_OUTPUT,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0118 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = [
        _require_accepted_review(
            r0110_review_path,
            expected_sha256=r0110_review_sha256,
            round_id="0110",
        ),
        _require_accepted_review(
            r0111_review_path,
            expected_sha256=r0111_review_sha256,
            round_id="0111",
        ),
    ]

    r0108_root = os.path.dirname(os.path.realpath(r0108_queue_path))
    r0108_queue, r0108_queue_signature = _load_queue(
        r0108_queue_path,
        schema="round0108-diverse-jina-evaluation-queue-v1",
        round_id="0108",
    )
    r0108_terminal = _require_clean_terminal(
        os.path.join(r0108_root, "runner-terminal.json"),
        round_id="0108",
    )
    r0108_jobs = _jobs_by_action(
        r0108_queue,
        required={
            "calibrate_jina_density",
            "transform_retained_map",
            "score_core_geometry",
            "score_ood",
        },
    )

    r0110_root = os.path.dirname(os.path.realpath(r0110_queue_path))
    r0110_queue, r0110_queue_signature = _load_queue(
        r0110_queue_path,
        schema="round0110-diverse-jina-seed43-evaluation-queue-v2",
        round_id="0110",
    )
    r0110_terminal = _require_clean_terminal(
        os.path.join(r0110_root, "runner-terminal.json"),
        round_id="0110",
    )
    r0110_artifacts = os.path.join(r0110_root, "artifacts")
    r0110_matched_output = os.path.join(
        r0110_artifacts, "matched-calibration-density"
    )
    r0110_decision_output = os.path.join(
        r0110_artifacts, "two-seed-decision"
    )
    r0110_fixed = [
        expected_input_signature(
            os.path.join(r0110_matched_output, "matched-density.json")
        ),
        expected_input_signature(
            os.path.join(
                r0110_matched_output, "matched-density-arrays.npz"
            )
        ),
        expected_input_signature(
            os.path.join(r0110_decision_output, "two-seed-decision.json")
        ),
    ]

    r0111_root = os.path.dirname(os.path.realpath(r0111_queue_path))
    r0111_queue, r0111_queue_signature = _load_queue(
        r0111_queue_path,
        schema="round0111-diverse-jina-seed44-training-queue-v1",
        round_id="0111",
    )
    r0111_terminal = _require_clean_terminal(
        os.path.join(r0111_root, "runner-terminal.json"),
        round_id="0111",
    )
    r0111_train = [
        expected_input_signature(os.path.join(r0111_train_output, name))
        for name in (
            "train-receipt.json",
            "production-config.json",
            "model.pt",
        )
    ]

    r0108_selection = os.path.join(
        r0108_root, "inputs", "registered-selections.npz"
    )
    r0108_artifacts = os.path.join(r0108_root, "artifacts")
    r0108_calibration_output = os.path.join(
        r0108_artifacts, "jina-density-calibration"
    )
    r0108_fixed = [
        expected_input_signature(r0108_selection),
        expected_input_signature(
            os.path.join(
                r0108_calibration_output,
                "jina-density-calibration.json",
            )
        ),
        expected_input_signature(
            os.path.join(
                r0108_calibration_output,
                "jina-density-calibration-arrays.npz",
            )
        ),
    ]
    common = _dedupe([
        expected_input_signature(round_file),
        *reviews,
        r0108_queue_signature,
        r0108_terminal,
        r0110_queue_signature,
        r0110_terminal,
        r0111_queue_signature,
        r0111_terminal,
        *r0110_fixed,
        *r0111_train,
    ])

    transform_source = r0108_jobs["transform_retained_map"]
    core_source = r0108_jobs["score_core_geometry"]
    ood_source = r0108_jobs["score_ood"]
    calibration_source = r0108_jobs["calibrate_jina_density"]
    with open(
        calibration_source["census_receipt"], encoding="utf-8"
    ) as handle:
        census_receipt = json.load(handle)
    source_signature = expected_input_signature(
        str((census_receipt.get("source") or {}).get("canonical_path") or "")
    )
    if source_signature != census_receipt.get("source"):
        raise RuntimeError("R0040 Jina calibration source bytes changed")

    queue_root = create_fresh_directory(
        queue_root, label="R0118 seed-44 frozen evaluation queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    transform_output = os.path.join(artifacts, "coordinates-seed44")
    core_output = os.path.join(artifacts, "core-geometry-seed44")
    ood_output = os.path.join(artifacts, "ood-seed44")
    matched_output = os.path.join(
        artifacts, "matched-calibration-density-seed44"
    )
    decision_output = os.path.join(artifacts, "three-seed-decision")
    # Keep the conventional sibling name: scan_projection_maps derives the
    # base-map sample-ID path from coordinates/../semantic-renders.
    render_output = os.path.join(artifacts, "semantic-renders")

    jobs = [
        {
            **_job_template(transform_source),
            "id": "transform_seed44",
            "action": "transform_seed44",
            "handler_module": "experiments.round0118_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [transform_output],
            "done_marker": os.path.join(
                artifacts, "transform_seed44.done.json"
            ),
            "expected_inputs": _dedupe([
                *common,
                *transform_source["expected_inputs"],
            ]),
            "p90_wall_s": TRANSFORM_P90_S,
            "train_output": r0111_train_output,
            "release_sha": release_sha,
        },
        {
            **_job_template(core_source),
            "id": "score_seed44_core",
            "action": "score_seed44_core",
            "handler_module": "experiments.round0118_nodes",
            "handler_callable": "run_job",
            "deps": ["transform_seed44"],
            "outputs": [core_output],
            "done_marker": os.path.join(
                artifacts, "score_seed44_core.done.json"
            ),
            "expected_inputs": _dedupe([
                *common,
                *core_source["expected_inputs"],
                *r0108_fixed,
            ]),
            "p90_wall_s": CORE_P90_S,
            "train_output": r0111_train_output,
            "transform_output": transform_output,
            "calibration_output": r0108_calibration_output,
            "selection": r0108_selection,
            "release_sha": release_sha,
        },
        {
            **_job_template(ood_source),
            "id": "score_seed44_ood",
            "action": "score_seed44_ood",
            "handler_module": "experiments.round0118_nodes",
            "handler_callable": "run_job",
            "deps": ["transform_seed44"],
            "outputs": [ood_output],
            "done_marker": os.path.join(
                artifacts, "score_seed44_ood.done.json"
            ),
            "expected_inputs": _dedupe([
                *common,
                *ood_source["expected_inputs"],
                *r0108_fixed,
            ]),
            "p90_wall_s": OOD_P90_S,
            "train_output": r0111_train_output,
            "transform_output": transform_output,
            "selection": r0108_selection,
            "release_sha": release_sha,
        },
        {
            "id": "score_seed44_matched_fineweb_density",
            "action": "score_seed44_matched_fineweb_density",
            "handler_module": "experiments.round0118_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [matched_output],
            "done_marker": os.path.join(
                artifacts,
                "score_seed44_matched_fineweb_density.done.json",
            ),
            "expected_inputs": _dedupe([
                *common,
                *calibration_source["expected_inputs"],
                *r0108_fixed,
                expected_input_signature(
                    calibration_source["census_receipt"]
                ),
                expected_input_signature(
                    calibration_source["representative_reference"]
                ),
                source_signature,
            ]),
            "p90_wall_s": MATCHED_DENSITY_P90_S,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "seed44_train_output": r0111_train_output,
            "graph_manifest": transform_source["graph_manifest"],
            "graph_manifest_sha256": transform_source[
                "graph_manifest_sha256"
            ],
            "census_receipt": calibration_source["census_receipt"],
            "census_receipt_sha256": calibration_source[
                "census_receipt_sha256"
            ],
            "representative_reference": calibration_source[
                "representative_reference"
            ],
            "representative_reference_sha256": calibration_source[
                "representative_reference_sha256"
            ],
            "calibration_output": r0108_calibration_output,
            "r0110_matched_density": os.path.join(
                r0110_matched_output, "matched-density.json"
            ),
            "release_sha": release_sha,
        },
        {
            "id": "decide_three_seed_stability_and_publish_registry",
            "action": (
                "decide_three_seed_stability_and_publish_registry"
            ),
            "handler_module": "experiments.round0118_nodes",
            "handler_callable": "run_job",
            "deps": [
                "score_seed44_core",
                "score_seed44_ood",
                "score_seed44_matched_fineweb_density",
            ],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts,
                "decide_three_seed_stability_and_publish_registry.done.json",
            ),
            "expected_inputs": common,
            "p90_wall_s": DECISION_P90_S,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
            "r0110_decision": os.path.join(
                r0110_decision_output, "two-seed-decision.json"
            ),
            "transform_output": transform_output,
            "core_output": core_output,
            "ood_output": ood_output,
            "matched_density_output": matched_output,
            "render_output": render_output,
            "release_sha": release_sha,
        },
    ]

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=1.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0118-diverse-jina-seed44-evaluation-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-research"
    queue["required_reviews"] = ["0110", "0111"]
    queue["capability_dependencies"] = [
        "jina-diverse-25m-full768-trained-map-seed44-v1",
        "jina-diverse-25m-map-registry-v1",
    ]
    queue["capabilities_produced"] = [
        "jina-diverse-25m-seed44-atlas-quality-v1",
        "jina-diverse-25m-three-seed-quality-v1",
        "jina-diverse-25m-three-seed-matched-fineweb-qualified-atlas-v1",
        "jina-diverse-25m-seed44-map-registry-v1",
    ]
    queue["training_performed"] = False
    queue["scientific_contract"] = {
        "role": (
            "seed-44 replay of R0108's frozen native evaluation plus an "
            "exact extension of R0110's matched-FineWeb density cell"
        ),
        "seed44_training_round": "0111",
        "seed44": 44,
        "selection": expected_input_signature(r0108_selection),
        "absolute_native_core_gate": "identical to R0108",
        "absolute_headline_polish_ood_gate": "identical to R0108",
        "strict_three_seed_release_rule": (
            "R0110's strict two-seed native quality must already pass and "
            "seed44 must independently pass the same frozen native core and "
            "Polish OOD gates"
        ),
        "matched_fineweb_density": {
            "prior": expected_input_signature(
                os.path.join(
                    r0110_matched_output, "matched-density.json"
                )
            ),
            "universe": (
                "exact R0040 representative FineWeb calibration universe"
            ),
            "anchors": (
                "exact R0040 10k calibration anchors and high-D radii"
            ),
            "floor": "unchanged R0108 registered Jina floor",
            "role": (
                "separate matched-FineWeb qualification only; cannot "
                "override or rescue native diverse-universe density"
            ),
        },
        "three_seed_metric_ranges": "diagnostic-only",
        "projection_ffr": "diagnostic-only",
        "thresholds_tunable_after_any_seed": False,
        "production_document_prompt_transfer_resolved": False,
        "production_readiness_claimed": False,
        "registry_and_probe_publication": True,
        "map_decision": True,
        "timing_basis": {
            "r0108_measured_gpu_wall_s": {
                "transform": 37.43145924899727,
                "core": 85.1386641911231,
                "ood": 99.79384239483625,
            },
            "r0110_measured_gpu_wall_s": {
                "transform": 37.78170580789447,
                "core": 90.80119433626533,
                "ood": 111.14488627016544,
                "matched_density_two_models": 13.835361725185066,
            },
            "p90_gpu_wall_s": GPU_P90_S,
        },
    }
    queue["jobs"] = jobs
    queue["p90_gpu_seconds"] = {
        "transform_seed44": TRANSFORM_P90_S,
        "score_seed44_core": CORE_P90_S,
        "score_seed44_ood": OOD_P90_S,
        "score_seed44_matched_fineweb_density": (
            MATCHED_DENSITY_P90_S
        ),
        "total": GPU_P90_S,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0110-review", required=True)
    parser.add_argument("--r0110-review-sha256", required=True)
    parser.add_argument("--r0111-review", required=True)
    parser.add_argument("--r0111-review-sha256", required=True)
    parser.add_argument("--r0108-queue", default=R0108_QUEUE)
    parser.add_argument("--r0110-queue", default=R0110_QUEUE)
    parser.add_argument("--r0111-queue", default=R0111_QUEUE)
    parser.add_argument("--r0111-train-output", default=R0111_TRAIN_OUTPUT)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0118(
        release_sha=args.release_sha,
        r0110_review_path=args.r0110_review,
        r0110_review_sha256=args.r0110_review_sha256,
        r0111_review_path=args.r0111_review,
        r0111_review_sha256=args.r0111_review_sha256,
        r0108_queue_path=args.r0108_queue,
        r0110_queue_path=args.r0110_queue,
        r0111_queue_path=args.r0111_queue,
        r0111_train_output=args.r0111_train_output,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
