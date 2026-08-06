#!/usr/bin/env python3
"""Prepare, but never launch, the reviewed width-factorial synthesis."""
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
from basemap.round0207_width_factorial import (
    FACTORIAL_CAPABILITY,
    MEMO_CAPABILITY,
    ROUND_ID,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0207"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0207-2026-08-06.md")
CAMPAIGN_FILE = os.path.join(
    LAB_ROOT, "campaign-2026-08-06-width-scaling-and-v0-ship.md"
)

R0202_LADDER = (
    "/data/latent-basemap/runs/round-0202/queue/artifacts/"
    "h4096-nested-ladder-synthesis/h4096-ladder-summary.json"
)
R0203_LADDER = (
    "/data/latent-basemap/runs/round-0203/queue/artifacts/"
    "h2048-low-dose-ladder-synthesis/h2048-low-dose-ladder-summary.json"
)
R0190_SYNTHESIS = (
    "/data/latent-basemap/runs/round-0190/queue/artifacts/"
    "jina-composition-boundary-three-seed-synthesis-v1/"
    "three-seed-boundary-synthesis.json"
)
R0191_DECISION = (
    "/data/latent-basemap/runs/round-0191/queue/artifacts/"
    "jina-document-english-8m-h4096-width-contrast-v1/width-decision.json"
)
R0201_LOCALIZATION = (
    "/data/latent-basemap/runs/round-0201/queue/artifacts/"
    "jina-mixed-pile-boundary-loss-localization-v1/pile-loss-localization.json"
)
R0168_U12 = (
    "/data/latent-basemap/runs/round-0168/queue/artifacts/"
    "prompted-diverse-u12/prompted-u12-manifest.json"
)
R0171_GRAPH = (
    "/data/latent-basemap/runs/round-0171/queue/artifacts/"
    "fuzzy-k50-graph-and-reference/graph-manifest.json"
)
R0173_OOD_AUDIT = (
    "/data/latent-basemap/runs/round-0173/queue/artifacts/"
    "jina-prompted-u12-ood-probe-pack-v1/audit.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    if not os.path.isfile(ROUND_FILE):
        raise RuntimeError("R0207 issued round file is absent")
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0207 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def _current_accepted_inputs(
    round_id: str, *, required_release: str | None = None
) -> list[dict[str, Any]]:
    """Select the one accepted review still bound to current append-only docs."""
    matches: list[list[dict[str, Any]]] = []
    for review_path in sorted(
        glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))
    ):
        review = _frontmatter(review_path)
        if review.get("round_id") != round_id or review.get("status") != "accepted":
            continue
        releases = _frontmatter_list(review, "releases")
        if required_release is not None and required_release not in releases:
            continue
        round_path = os.path.join(LAB_ROOT, review.get("round") or "")
        result_path = os.path.join(LAB_ROOT, review.get("result") or "")
        try:
            round_signature = expected_input_signature(round_path)
            result_signature = expected_input_signature(result_path)
        except (FileNotFoundError, OSError):
            continue
        if (
            round_signature["sha256"] != review.get("round_sha256")
            or result_signature["sha256"] != review.get("result_sha256")
            or _frontmatter(result_path).get("release_commit")
            != review.get("verified_release_commit")
        ):
            # Historical additive correction reviews can leave an earlier
            # accepted review whose same-named result no longer has its old
            # bytes. Only the review bound to current durable bytes is usable.
            continue
        matches.append([
            round_signature,
            result_signature,
            expected_input_signature(review_path),
        ])
    if len(matches) != 1:
        raise RuntimeError(
            f"R0207 requires one current accepted Review {round_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def prepare_round0207(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0207 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    accepted = _dedupe([
        *_current_accepted_inputs(
            "0168",
            required_release=(
                "capability:jina-document-diverse-r0132-u12-host-fp16-v1"
            ),
        ),
        *_current_accepted_inputs("0171"),
        *_current_accepted_inputs("0173"),
        *_current_accepted_inputs(
            "0184",
            required_release=(
                "capability:jina-document-english-8m-prompted-dose-midpoint-readout-v1"
            ),
        ),
        *_current_accepted_inputs(
            "0187",
            required_release=(
                "capability:jina-document-english-composition-controlled-nested-ladder-v1"
            ),
        ),
        *_current_accepted_inputs(
            "0190",
            required_release=(
                "capability:jina-composition-boundary-three-seed-synthesis-v1"
            ),
        ),
        *_current_accepted_inputs(
            "0191",
            required_release=(
                "capability:jina-document-english-8m-h4096-width-contrast-v1"
            ),
        ),
        *_current_accepted_inputs(
            "0201",
            required_release=(
                "capability:jina-mixed-pile-boundary-loss-localization-v1"
            ),
        ),
        *_current_accepted_inputs(
            "0202",
            required_release=(
                "capability:jina-document-english-h4096-composition-nested-dose-ladder-v1"
            ),
        ),
        *_current_accepted_inputs(
            "0203",
            required_release=(
                "capability:jina-document-english-h2048-composition-nested-low-dose-ladder-v1"
            ),
        ),
    ])
    sources = {
        "h2048": expected_input_signature(R0203_LADDER),
        "h4096": expected_input_signature(R0202_LADDER),
        "r0190": expected_input_signature(R0190_SYNTHESIS),
        "r0191": expected_input_signature(R0191_DECISION),
        "r0201": expected_input_signature(R0201_LOCALIZATION),
        "u12": expected_input_signature(R0168_U12),
        "graph": expected_input_signature(R0171_GRAPH),
        "ood_audit": expected_input_signature(R0173_OOD_AUDIT),
    }
    queue_root = create_fresh_directory(queue_root, label="R0207 synthesis queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(CAMPAIGN_FILE),
        *accepted,
        *sources.values(),
    ])
    factorial_output = os.path.join(
        artifacts, "jina-width-by-n-factorial-capacity-economics-v1"
    )
    memo_output = os.path.join(
        artifacts, "jina-prompted-diverse-u12-next-rung-design-v1"
    )
    jobs = [
        {
            "id": "synthesize_width_by_n_factorial",
            "action": "factorial",
            "handler_module": "experiments.round0207_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "ladders": {"h2048": sources["h2048"], "h4096": sources["h4096"]},
            "r0190": sources["r0190"],
            "r0191": sources["r0191"],
            "r0201": sources["r0201"],
            "outputs": [factorial_output],
            "done_marker": os.path.join(artifacts, "width-factorial.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 30.0,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
        {
            "id": "write_prompted_u12_design_memo",
            "action": "u12_memo",
            "handler_module": "experiments.round0207_nodes",
            "handler_callable": "run_job",
            "deps": ["synthesize_width_by_n_factorial"],
            "factorial_output": factorial_output,
            "u12_manifest": sources["u12"],
            "graph_precedent": sources["graph"],
            "ood_audit": sources["ood_audit"],
            "outputs": [memo_output],
            "done_marker": os.path.join(artifacts, "prompted-u12-memo.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 30.0,
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
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    queue.update({
        "schema": "round0207-width-factorial-and-u12-memo-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-synthesis",
        "required_reviews": [
            "0168", "0171", "0173", "0184", "0187",
            "0190", "0191", "0201", "0202", "0203",
        ],
        "capability_dependencies": [
            "jina-document-diverse-r0132-u12-host-fp16-v1",
            "jina-document-english-h4096-composition-nested-dose-ladder-v1",
            "jina-document-english-h2048-composition-nested-low-dose-ladder-v1",
        ],
        "capabilities_produced": [FACTORIAL_CAPABILITY, MEMO_CAPABILITY],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {"total": 0.0},
        "scientific_contract": {
            "question": (
                "does width flatten the fixed-dose composition-controlled N "
                "response, at what cost, and what U12 design follows?"
            ),
            "factorial": "2 widths x 3 nested N rungs; seed 42; fixed dose",
            "registered_metric": "Pile FFR",
            "retention_floor": 0.97,
            "density_v2_and_projection_ffr": "diagnostic",
            "economics": "actual train wall, update rate, and quality/GPU-h",
            "u12_rows": 12_474_331,
            "u12_output_role": "memo only; no GPU launch authority",
            "training_performed": False,
            "map_registry_state_changed": False,
            "production_or_publishing": False,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    path = prepare_round0207(
        release_sha=args.release_sha,
        queue_root=args.queue_root,
    )
    print(json.dumps({"path": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
