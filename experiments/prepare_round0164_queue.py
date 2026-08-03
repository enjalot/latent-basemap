#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0164 population redecision."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
from basemap.round0162_prompted_english_staging import VIEW_ROWS
from basemap.round0164_prompted_population import CAPABILITY, HOST_CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0163_queue import (
    LAYOUT_PATH,
    R0113_MAPPING,
    VIEW_PATH,
    _dedupe_signatures,
)
from experiments.round0163_nodes import _document_slices, _read_sealed, _source_layouts


ROUND_ROOT = "/data/latent-basemap/runs/round-0164"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0164-2026-08-03.md")
RESULT_0163 = os.path.join(LAB_ROOT, "result-0163-2026-08-03.md")
REVIEWS = {
    "0113": os.path.join(LAB_ROOT, "review-0113-2026-07-30.md"),
    "0162": os.path.join(LAB_ROOT, "review-0162-2026-08-03.md"),
    "0163": os.path.join(LAB_ROOT, "review-0163-2026-08-03.md"),
}
R0163_CENSUS = (
    "/data/latent-basemap/runs/round-0163/queue-correction-1/artifacts/"
    "prompted-english-8m-representatives/representative-census.json"
)
R0163_MAPPING = (
    "/data/latent-basemap/runs/round-0163/queue-correction-1/artifacts/"
    "prompted-english-8m-representatives/compact-to-canonical.i64.npy"
)


def _accepted_review(round_id: str) -> dict[str, Any]:
    path = REVIEWS[round_id]
    frontmatter = _frontmatter(path)
    if frontmatter.get("status") != "accepted" or frontmatter.get("round_id") != round_id:
        raise RuntimeError(f"R0164 required Review {round_id} is not accepted")
    return expected_input_signature(path)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0164 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def prepare_round0164(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0164 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = [_accepted_review(round_id) for round_id in ("0113", "0162", "0163")]
    result0163 = expected_input_signature(RESULT_0163)
    census0163 = expected_input_signature(R0163_CENSUS)
    mapping0163 = expected_input_signature(R0163_MAPPING)
    r0113_mapping = expected_input_signature(R0113_MAPPING)

    layout_signature = expected_input_signature(LAYOUT_PATH)
    view_signature = expected_input_signature(VIEW_PATH)
    layout = _read_sealed(layout_signature, label="accepted R0162 layout")
    view = _read_sealed(view_signature, label="accepted R0162 first8m view")
    if layout.get("first8m_view") != view_signature or int(view.get("rows", -1)) != VIEW_ROWS:
        raise RuntimeError("R0164 accepted R0162 view lineage changed")
    source_manifests = layout.get("source_manifests") or {}
    r0116_signature = dict(source_manifests["0116"])
    r0120_signature = dict(source_manifests["0120"])
    r0116 = _read_sealed(r0116_signature, label="accepted R0116 manifest")
    r0120 = _read_sealed(r0120_signature, label="accepted R0120 manifest")
    text_layout, _raw_slices, _source_inputs = _source_layouts(r0116, r0120)
    document_slices = _document_slices(view)
    payload_inputs = _dedupe_signatures([
        *[dict(item["text"]) for item in text_layout],
        *[item.signature for item in document_slices],
    ])

    queue_root = create_fresh_directory(queue_root, label="R0164 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, "prompted-english-8m-population-v2")
    expected_inputs = _dedupe([
        round_signature,
        *reviews,
        result0163,
        census0163,
        mapping0163,
        r0113_mapping,
        layout_signature,
        view_signature,
        r0116_signature,
        r0120_signature,
        *payload_inputs,
    ])
    job = {
        "id": "decide_prompted_only_population",
        "action": "decide_prompted_only_population",
        "handler_module": "experiments.round0164_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "prompted-population.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 900.0,
        "canonical_layout": layout_signature,
        "first8m_view": view_signature,
        "r0116_manifest": r0116_signature,
        "r0120_manifest": r0120_signature,
        "payload_inputs": payload_inputs,
        "r0163_census": census0163,
        "r0163_mapping": mapping0163,
        "r0113_compact_mapping": r0113_mapping,
        "accepted_reviews": reviews,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    }
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
        "schema": "round0164-prompted-population-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0113", "0162", "0163"],
        "capability_dependencies": [
            "jina-document-english-first8m-view-v1",
            "claim:R0163-prompted-English-population-confound",
        ],
        "capabilities_produced": [CAPABILITY, HOST_CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "source_rows": VIEW_ROWS,
            "selection_rule": (
                "union exact source-text UTF-8 and Document: stored-fp16 families; "
                "keep lowest canonical row per transitive component"
            ),
            "raw_unprompted_embedding_relation_used": False,
            "r0163_mapping_must_be_strict_subset": True,
            "accepted_r0113_first2m_prefix_must_remain_exact": True,
            "prompted_cross_source_text_family_count_must_be_zero": True,
            "retained_source_text_and_prompted_family_count_required": 0,
            "host_contiguous_prompted_fp16_required": True,
            "multiplicity_is_metadata": True,
            "no_graph": True,
            "no_training": True,
            "negative_outcome_releases_no_q2_population_capability": True,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0164(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

